import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math

class CrossAttention(nn.Module):
    """Cross-Attention Layer for fusing L3 label features with TCN image features."""
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_heads = num_heads

    def forward(self, img_features, l3_features):
        """
        img_features: (Batch, Time, Channels) -> reshape to (Time, Batch, Channels)
        l3_features: (Batch, Time, Channels)
        """
        img_features = img_features.permute(1, 0, 2)  # Convert to (T, B, C)
        l3_features = l3_features.permute(1, 0, 2)  # Convert to (T, B, C)
        
        attn_out, attn_weights = self.cross_attn(query=l3_features, key=img_features, value=img_features)
        
        return attn_out.permute(1, 0, 2), attn_weights  # Return to (B, T, C)


class Chomp1d(nn.Module):
    """Remove padding in 2D for temporal convolutions."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock1D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock1D, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet1D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet1D, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock1D(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size - 1) * dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MustafaNet1DTCN(nn.Module):
    def __init__(self, num_classes=15, anticipated_frames=5, num_l3_classes=42, feature_dim=256, num_heads=4, window_size=5):
        super(MustafaNet1DTCN, self).__init__()
        self.anticipated_frames = anticipated_frames
        self.num_classes = num_classes
        # TemporalConvNet2D: takes 2048-channel features over temporal window as input
        self.tcn_local = TemporalConvNet1D(num_inputs=2048, num_channels=[256, 512, 512, feature_dim], kernel_size=3, dropout=0.2)
        self.l3_embed = nn.Embedding(num_embeddings=num_l3_classes, embedding_dim=feature_dim)
        # Cross-Attention layer
        self.cross_attention = CrossAttention(embed_dim=feature_dim, num_heads=num_heads)
        
        # Positional embeddings
        self.positional_embedding_l3 = self.sinusoidal_positional_encoding(window_size, feature_dim)
        self.positional_embedding_tcn = self.sinusoidal_positional_encoding(window_size, feature_dim)
        
        # Final linear layer to output class predictions
        self.fc_out = nn.Linear(feature_dim, num_classes * anticipated_frames)
        # Final regression layer
        self.regression = nn.Conv1d(in_channels=feature_dim, out_channels=num_classes * anticipated_frames, kernel_size=1)

    
    def sinusoidal_positional_encoding(self, seq_len, emb_dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pos_embed = torch.zeros(seq_len, emb_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        return pos_embed
        
    def forward(self, x, l3):
        # Expected input shape: [batch, window, 2048]
        B, T, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.tcn_local(x).permute(0, 2, 1)  # Apply 2D TCN
        pos_embed_tcn = self.positional_embedding_tcn.unsqueeze(0).expand(B, T, -1).to(x.device)
        x_with_pos = x + pos_embed_tcn

        l3_embed = self.l3_embed(l3)  # [B, T, l3_feature_dim]
        pos_embed_l3 = self.positional_embedding_l3.unsqueeze(0).expand(B, T, -1).to(l3_embed.device)
        l3_embed_with_pos = l3_embed + pos_embed_l3

        # Step 3: Cross-Attention between TCN features and L3 label features
        x, attn_weights = self.cross_attention(x_with_pos, l3_embed_with_pos)
        
        # Step 4: Pass through the final classification layer and reshape
        x = self.fc_out(x)  # Shape: [B, T, num_classes * anticipated_frames]
        x = x.mean(dim=1)
        x = x.view(B, self.anticipated_frames, self.num_classes)  # Shape: [B, T, anticipated_frames, num_classes]
        return x, attn_weights


        #x = self.regression(x)  # Apply final regression layer to get class scores
        #x = x.view(x.size(0), self.anticipated_frames, -1, x.size(2))
        #return x.mean(dim=3)  # Output shape: [batch, num_classes]
