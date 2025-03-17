import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from einops import rearrange

class CrossAttention(nn.Module):
    """Cross-Attention Layer for fusing L3 label features with image features."""
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_heads = num_heads

    def forward(self, img_features, l3_features):
        """
        img_features: (Batch, Time, 144) (B, C_img, T, H, W) -> reshape to (T, B, C_img * H * W)
        l3_features: (B, Time, 144)
        """
        
        B, T, C_img = img_features.shape
        attn_output = torch.zeros(B, T, C_img).to(img_features.device)
        attn_weights = torch.zeros(B, T, C_img).to(img_features.device)

        # Convert to (Time, batch, 144)
        key_val = img_features#.permute(1, 0, 2)
        query = l3_features#.permute(1, 0, 2)

        attn_out, attn_weight = self.cross_attn(query = query, key=key_val, value=key_val)
        '''
        attn_out = [4, 8, 64]
        attn_weight = [8, 4, 576]
        '''

        return attn_out, attn_weight

class Chomp2d(nn.Module):
    """Removes padding symmetrically from both sides of a tensor."""
    def __init__(self, chomp_size_h, chomp_size_w):
        """
        Args:
        - chomp_size_h: Amount of padding to remove from height.
        - chomp_size_w: Amount of padding to remove from width.
        """
        super(Chomp2d, self).__init__()
        self.chomp_size_h = chomp_size_h
        self.chomp_size_w = chomp_size_w

    def forward(self, x):
        # Check the input size to avoid index out of range errors
        if x.shape[2] <= self.chomp_size_h or x.shape[3] <= self.chomp_size_w:
            raise ValueError(f"Chomp size too large for input tensor shape: {x.shape}")
        
        # Remove padding from both height and width
        return x[:, :, :-self.chomp_size_h, :-self.chomp_size_w].contiguous()
    
class TemporalBlock2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size=3, stride=2, dilation=1, dropout=0.2):
        padding = (kernel_size - 1) // 2 #* dilation # padding 설정
        super(TemporalBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size=kernel_size, stride=stride, padding=(padding, padding))#, dilation=dilation)
        self.chomp1 = Chomp2d(padding, padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size=kernel_size, stride=1, padding=(padding, padding))#, dilation=dilation)
        self.chomp2 = Chomp2d(padding, padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, kernel_size=1, stride=stride) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #print(f"Input to TemporalBlock2D: {x.shape}")
        out = self.net(x)
        #print(f"Output from TemporalBlock2D: {out.shape}")
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2D(in_channels, out_channels, kernel_size, stride=2, dilation=dilation_size,
                                       dropout=dropout)]
            # if i == num_levels - 1:  # 마지막 레이어에서 더 줄이기
            #     layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MustafaNet2DTCN(nn.Module):
    def __init__(self, num_classes=8, l3_feature_dim=64, window_size=4, num_l3_classes = 48):
        super(MustafaNet2DTCN, self).__init__()

        # 2D TCN layers to process each image frame independently and then fuse
        self.tcn_local = TemporalConvNet2D(
            num_inputs=3,  # Assuming RGB images as input
            num_channels=[16, 32, 64],  # Incremental channel sizes
            kernel_size=3,
            dropout=0.2
        )

        # Pooling layer to reduce H x W dimension to 1 x 1, resulting in [B, T, 64]
        self.spatial_pool = nn.AdaptiveAvgPool2d((9, 16))  # (64, 1, 1)

        # Linear layer to convert the pooled output to a desired embedding dimension (64 in this case)
        self.flatten = nn.Flatten(start_dim=2)  # (B, T, 64)
        
        # Embedding layers for L3 labels
        self.l3_embed = nn.Embedding(num_embeddings=num_l3_classes, embedding_dim=l3_feature_dim)
        self.positional_embedding_l3 = self.sinusoidal_positional_encoding(window_size, l3_feature_dim)
        self.positional_embedding_image = self.sinusoidal_positional_encoding(window_size, l3_feature_dim)

        # Cross-Attention layer
        self.cross_attention = CrossAttention(embed_dim=l3_feature_dim, num_heads=16)#nn.MultiheadAttention(embed_dim=64, num_heads=4)

        # Final layer to output class predictions
        self.fc_out = nn.Linear(in_features=l3_feature_dim, out_features=num_classes)
    
    def sinusoidal_positional_encoding(self, seq_len, emb_dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pos_embed = torch.zeros(seq_len, emb_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term) # apply sine to even indices
        pos_embed[:, 1::2] = torch.cos(position * div_term) # apply cosine to odd indices
        return pos_embed
    
    def forward(self, x, l3_labels):
        B, T, C_img, H, W = x.shape  # Input shape: [batch_size, time_steps, channels, height, width]
        # Step 1: Pass each frame through TCN layers independently
        x = x.view(B * T, C_img, H, W)  # [B*T, C, H, W]
        x = self.tcn_local(x)  # Output shape: [B*T, 64, H, W]

        # Step 2: Apply pooling to reduce H x W -> 1 x 1 (spatial pooling)
        x = self.spatial_pool(x)  # Output shape: [B*T, 64, 9, 16]
        B_T, C_emb, H_new, W_new = x.shape
        #print(x.shape)

        # Step 3: Flatten and reshape to [B, T, 64]
        x = x.view(B, T, C_emb, H_new * W_new)
        pos_embed_image = self.positional_embedding_image.unsqueeze(1).expand(T, B, -1)
        pos_embed_image = pos_embed_image.permute(1, 0, 2).unsqueeze(3).expand(B, T, -1, H_new*W_new)
        x_with_pos = x + pos_embed_image.to(x.device)
        x_with_pos = rearrange(x_with_pos, 'b t c hw -> (t hw) b c')

        
        l3_embed = self.l3_embed(l3_labels)  # [B, T, 64]
        l3_embed = l3_embed.permute(1, 0, 2)
        pos_embed_l3 = self.positional_embedding_l3.unsqueeze(1).expand(T, B, -1)
        l3_embed_with_pos = l3_embed + pos_embed_l3.to(l3_embed.device)

        x, attn_weights = self.cross_attention(x_with_pos, l3_embed_with_pos)
        visualize = x

        x = x.mean(dim=0)  # [B, 64]
        x = self.fc_out(x)  # [B, 8]

        return x, visualize, attn_weights

# Function for attention visualization
def visualize_attention(attn_weights, l3_labels, time_steps):
    attn_weights = attn_weights.detach().cpu().numpy()
    plt.figure(figsize=(12, 6))
    sns.heatmap(attn_weights.squeeze(), cmap='viridis', annot=True, xticklabels=[f"L3 Label {l}" for l in l3_labels],
                yticklabels=[f"Timestep {i}" for i in range(time_steps)])
    plt.show()