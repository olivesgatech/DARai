# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns

# class CrossAttention(nn.Module):
#     """Cross-Attention Layer for fusing L3 label features with image features."""
#     def __init__(self, embed_dim, num_heads=4):
#         super(CrossAttention, self).__init__()
#         self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)

#     def forward(self, img_features, l3_features):
#         """
#         img_features: (B, C_img, T, H, W) -> reshape to (T, B, C_img * H * W)
#         l3_features: (B, C_l3)
#         """
#         B, C_img, T, H, W = img_features.shape

#         # 8, 64, 4, 128
#         img_features = img_features.permute(2, 0, 1, 3, 4).contiguous()  # (T, B, C_img, H, W)
#         img_features = img_features.view(T*H*W, B, C_img)

#         # img_features = img_features.view(B, C_img, T, H * W).permute(2, 0, 3, 1).contiguous()  # (T, B, H*W, C_img) # (1, 8, 1280*720, 128)
#         # print(img_features.shape)
#         # img_features = img_features.view(T, B, -1)  # [T, B, C_img * H * W]
#         # print(img_features.shape)

#         # L3 feature shape: [B, C_l3] -> [1, B, C_l3]
#         #l3_features = l3_features.unsqueeze(0)  # [1, B, C_l3]
#         l3_features = l3_features.permute(1, 0, 2)
#         # Check for invalid indices
#         if torch.max(l3_features) >= l3_features.size(-1):
#             print(f"Error: L3 feature index out of range! max index = {torch.max(l3_features)}")


#         # Perform cross-attention: Query=img_features, Key/Value=L3 features
#         attn_output, attn_weights = self.cross_attn(query=l3_features, key=img_features, value=img_features)
#         #attn_output, attn_weights = self.cross_attn(query=img_features, key=l3_features, value=l3_features)
        
#         attn_output = attn_output.view(T, B, H, W, C_img).permute(1, 4, 0, 2, 3)  # Reshape back to [B, C_img, T, H, W]

#         return attn_output, attn_weights

# class Chomp2d(nn.Module):
#     '''Removing padding'''
#     def __init__(self, chomp_size):
#         super(Chomp2d, self).__init__()
#         self.chomp_size = chomp_size

#     def forward(self, x):
#         return x[:, :, :-self.chomp_size, :-self.chomp_size].contiguous()

# class TemporalBlock2D(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock2D, self).__init__()
#         self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size,
#                                stride=stride, padding=padding, dilation=dilation)
#         self.chomp1 = Chomp2d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout2d(dropout)

#         self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size,
#                                stride=stride, padding=padding, dilation=dilation)
#         self.chomp2 = Chomp2d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout2d(dropout)

#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
#         self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)

# class TemporalConvNet2D(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
#         super(TemporalConvNet2D, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock2D(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)


# class MustafaNet2DTCN(nn.Module):
#     """Model using 3D Convolution to handle sequences of images"""
#     def __init__(self, l3_feature_dim=64):
#         super(MustafaNet2DTCN, self).__init__()
#         # 3D convolution: input channels=3 (RGB), output channels=16, kernel=(4, 3, 3), padding=(0, 1, 1)
#         self.conv3d = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(1, 3, 3), padding=(0, 1, 1))
#         self.relu = nn.ReLU()
        
#         # 2D TCN layers after the 3D convolution (output channels=16 from the conv3d)
#         self.tcn_local = nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

#         # Separate embedding layers for L2 and L3
#         self.l3_embed = nn.Embedding(num_embeddings=48, embedding_dim=l3_feature_dim)  # Assuming 15 unique L3 classes

#         # Cross-Attention layer
#         self.cross_attention = CrossAttention(embed_dim=64, num_heads=4)

#         # Final layer to output class predictions (27 classes in your case)
#         self.conv_out = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1)

#     def forward(self, x, l3_labels):
#         # Input shape: [batch_size, window_size, channels, height, width]
#         # Permute to [batch_size, channels, window_size, height, width] for Conv3D
#         x = x.permute(0, 2, 1, 3, 4)  # (batch_size, channels, window_size, height, width)
#         x = self.conv3d(x)  # Apply 3D convolution # (8, 8, 4, 720, 1280)
#         x = self.relu(x) # (8, 8, 4, 720, 1280)

#         B, C, T, H, W = x.shape
#         x = x.view(B*T, C, H, W) # (32, 8, 720, 1280)
#         # Apply the 2D TCN layers
#         x = self.tcn_local(x) # (32, 64, 720, 1280)
#         x = x.view(B, T, 64, H, W).permute(0, 2, 1, 3, 4) # (8, 64, 4, 720, 1280)

#         l3_embed = self.l3_embed(l3_labels) # (8, 64) -> (8, 4, 64)
#         x, attn_weights = self.cross_attention(x, l3_embed)

#         x = x.mean(dim=2)
#         # Output layer
#         x = self.conv_out(x)

#         # Perform global average pooling over height and width (reduce to [batch_size, num_classes])
#         x = x.mean(dim=(2, 3))
#         return x, attn_weights

# def visualize_attention(attn_weights, l3_labels, time_steps):
#     """
#     Visualize the cross-attention weights between L3 labels and time sequence.
#     Args:
#     - attn_weights: Attention weights from cross-attention layer (T, B, 1).
#     - l3_labels: Actual L3 labels used during cross-attention.
#     - time_steps: Number of time steps in the input sequence.
#     """
#     # Squeeze the attention weights to shape [T, B]
#     attn_weights = attn_weights.squeeze().detach().cpu().numpy()

#     # Plot the heatmap for attention weights
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(attn_weights, cmap='viridis', annot=True, xticklabels=[f"L3 Label {l}" for l in l3_labels], yticklabels=[f"Timestep {i}" for i in range(time_steps)])
#     plt.xlabel("L3 Labels")
#     plt.ylabel("Time Steps")
#     plt.title("Cross-Attention between L3 Labels and Time Sequence")
#     plt.show()


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
        #print(attn_out.shape, attn_weight.shape)

        # for i in range(B):
        #     key_val = img_features[i].permute(2, 0, 1)
        #     query = l3_features[i].unsqueeze(0)
        #     #print(key_val.shape, query.shape)
        #     attn_out, attn_weight = self.cross_attn(query=query, key=key_val, value=key_val)
        #     # attn_out = (1, 4, 64)
        #     # attn_weight = (4, 1, 3600)
        #     attn_output[i] = attn_out
        #     #attn_weight = attn_weight.squeeze(1)
        #     attn_weights[i] = attn_weight.squeeze(1).unsqueeze(0)
        
        

        # # Change (B, C_img, T, H, W) to (T, B, C_img * H * W) to align with attention requirements
        # #img_features = img_features.permute(1, 0, 2, 3, 4).contiguous()  # (T, B, C_img, H, W)
        # #img_features = img_features.view(T, B, C_img, -1)  # (T, B, C_img * H * W)
        # img_features = img_features.permute(1, 0, 2)#.contiguous()
        # # Reshape l3_features to match the format for cross-attention
        # l3_features = l3_features.permute(1, 0, 2)  # (T, B, C_l3)

        # # Perform cross-attention: Query=L3 features, Key/Value=Image features
        # attn_output, attn_weights = self.cross_attn(query=l3_features, key=img_features, value=img_features)
        #print(f"Attention Output shape: {attn_output.shape}")  # Expected: (Time, Batch, 64)
        #print(f"Attention Weights shape: {attn_weights.shape}")  # Expected: (Batch, num_heads, Time, Time)
         # 5. Reshape attn_output back to match (B, T, H, W, C_l3)
        #attn_output = attn_output.permute(1, 2, 0, 3).view(B, T, H, W, -1)  # (B, T, H, W, C_l3)

        # 6. Reshape attn_weights to match (B, T, H, W)
        #attn_weights = attn_weights.view(T, B, H, W, -1).permute(1, 2, 3, 0).contiguous()  # (B, H, W, T)

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
    def __init__(self, num_classes=48, l3_feature_dim=64, window_size=4, num_l3_classes = 48):
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
    
    def forward(self, x, l3_labels, l2_labels):
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

        # x = x.view(B, T * H_new * W_new, C_emb)
        # x = x.permute(1, 0, 2)
        # pos_embed = self.positional_embedding.unsqueeze(1).expand(T, B, -1)

        # pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, H_new * W_new, 1)  # (T, B, 1, embedding) -> (T, B, H*W, embedding)

        # 2. Reshape to match key, value shape
        # pos_embed_expanded: (T, B, H*W, embedding) -> (T * H * W, B, embedding)
        # pos_embed = pos_embed.view(T * H_new * W_new, B, -1)
        # pos_embed = pos_embed.to(x.device)
        # x = x + pos_embed
        
        # x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        # x = self.conv3d(x)  # [B, 8, T, H, W]
        # x = x.mean(dim=2)  # Compress time steps to single spatial representation
        # x = self.tcn_local(x)  # [B, 64, H', W']

        # # Convert (H, W) to embedding vector size using Conv layers
        # x = x.view(B, 64, -1)  # [B, 64, H*W]
        # x = x.permute(2, 0, 1)  # [H*W, B, 64]
        random_l3_labels = torch.randint(0, self.l3_embed.num_embeddings, (B, T), device=x.device)
        l3_embed = self.l3_embed(l2_labels)  # [B, T, 64]
        l3_embed = l3_embed.permute(1, 0, 2)
        pos_embed_l3 = self.positional_embedding_l3.unsqueeze(1).expand(T, B, -1)
        l3_embed_with_pos = l3_embed + pos_embed_l3.to(l3_embed.device)

        x, attn_weights = self.cross_attention(x_with_pos, l3_embed_with_pos)
        visualize = x
        #print(x.shape, attn_weights.shape)
        #np.savetxt("debug.txt", visualize[0,0].detach().cpu().numpy(), fmt='%.4f')
        # Average pooling over spatial dimensions
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
