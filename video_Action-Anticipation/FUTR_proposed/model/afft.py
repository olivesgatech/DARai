import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import pdb
from einops import repeat, rearrange
from model.extras.transformer import Transformer
from model.extras.position import PositionalEncoding
import copy
from model.extras.transformerblock import Block

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class CMFuser(nn.Module):
    """SA-Fuser """
    def __init__(self, dim, depth=1, num_heads=4, mlp_ratio=4., qkv_bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.embd_drop = nn.Dropout(0.1)

        self.modality_token = nn.Parameter(torch.randn(1, 1, 1, dim))  # (1, 1, 2048)
        self.projection = nn.Linear(dim, dim)

    @staticmethod
    def generate_cross_attention_mask(sz):
        mask = torch.eye(sz)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, modal_feats):
        B, T, C = next(iter(modal_feats.values())).shape
        M = len(modal_feats)
        modality_token = self.modality_token.repeat(B, T, M, 1)
        attn_mask = self.generate_cross_attention_mask(2).to('cuda')

        for_fusion_feats = torch.cat([
            modal_feats['rgb'].unsqueeze(2),  # (B, T, 1, C)
            modal_feats['depth'].unsqueeze(2)  # (B, T, 1, C)
        ], dim=2)  # (B, T, M, C)

        for_fusion_feats = for_fusion_feats + modality_token
        for_fusion_feats = for_fusion_feats.view(B*T, M, C)

        x = self.embd_drop(for_fusion_feats)
        attn_weights=[]
        #x_res = x
        for blk in self.blocks:
            x, attn_weight = blk(x, attn_mask)
            attn_weights.append(attn_weight.view(B, T, *attn_weight.shape[1:]))
            #x = F.dropout(x, p=0.1, training=self.training)

        #x = x + x_res
        x = self.norm(x)
        feats_fusion = torch.mean(x, dim=1)
        feats_fusion = feats_fusion.view(B, T, C)

        return feats_fusion, torch.stack(attn_weights).transpose(0,1)

        # x_res = x
        # x = self.projection(x)
        # x = x + x_res
        # x = x.view(B, T, M + 1, C)
        
        # return x.mean(dim=2)

class FUTR(nn.Module):
    # 50salads: query_num: 19
    # Breakfast: query_num: 49
    # Darai: query_num: 48
    def __init__(self, n_class, hidden_dim, src_pad_idx, device, args, n_query=8, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, query_num=49):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.query_pad_idx = query_num - 1
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Linear(args.input_dim, hidden_dim)
        self.transformer = Transformer(hidden_dim, n_head, num_encoder_layers, num_decoder_layers,
                                        hidden_dim*4, normalize_before=False)
        self.n_query = n_query
        self.args = args
        nn.init.xavier_uniform_(self.input_embed.weight)
        self.l3_attention = nn.MultiheadAttention(hidden_dim, n_head, batch_first=True)
        self.query_attention = nn.MultiheadAttention(hidden_dim, n_head, batch_first=True)
        self.query_embed = nn.Embedding(self.n_query, hidden_dim) #(8, 128)
        self.fuser = CMFuser(dim=hidden_dim, depth=1, num_heads=n_head)

        if args.seg :
            self.fc_seg = nn.Linear(hidden_dim, n_class) #except SOS, EOS
            nn.init.xavier_uniform_(self.fc_seg.weight)

        if args.anticipate :
            self.fc = nn.Linear(hidden_dim, n_class)
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc_len = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_len.weight)

        self.fc_l3 = nn.Linear(hidden_dim, query_num)

        max_seq_len = args.max_pos_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.xavier_uniform_(self.pos_embedding)
        # Sinusoidal position encoding
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.pos_enc_depth = PositionalEncoding(hidden_dim)
        self.positional_embedding_l3 = self.sinusoidal_positional_encoding(max_seq_len, hidden_dim)
        self.positional_embedding_l3 = self.positional_embedding_l3.to(self.device)

        self.depth_projection = nn.Linear(224 * 224, hidden_dim)  # (1, 224, 224) → (hidden_dim)
        #self.depth_projection = nn.Linear(160 * 120, hidden_dim)  # (1, 224, 224) → (hidden_dim)
        
        nn.init.xavier_uniform_(self.depth_projection.weight)
        self.depth_layernorm = nn.LayerNorm(hidden_dim)  # 추가된 LayerNorm


        if args.input_type =='gt':
            self.gt_emb = nn.Embedding(n_class+2, self.hidden_dim, padding_idx=n_class+1)
            nn.init.xavier_uniform_(self.gt_emb.weight)


    def sinusoidal_positional_encoding(self, seq_len, emb_dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pos_embed = torch.zeros(seq_len, emb_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term) # apply sine to even indices
        pos_embed[:, 1::2] = torch.cos(position * div_term) # apply cosine to odd indices
        return pos_embed


    def forward(self, inputs, depth_features, mode='train', epoch=0, idx=0):
        if mode == 'train' :
            src, src_label = inputs
            tgt_key_padding_mask = None
            src_key_padding_mask = get_pad_mask(src_label, self.src_pad_idx).to(self.device)
            memory_key_padding_mask = src_key_padding_mask.clone().to(self.device)
        else :
            
            src = inputs
            src_key_padding_mask = None
            memory_key_padding_mask = None
            tgt_key_padding_mask = None

        tgt_mask = None
        if self.args.input_type == 'i3d_transcript':
            B, S, C = src.size()
            src = self.input_embed(src) #[B, S, C]
        elif self.args.input_type == 'gt':
            B, S = src.size()
            src = self.gt_emb(src)
        src = F.relu(src)
        
        #src = self.pos_enc(src)
       
        #pos_embed_l3 = self.positional_embedding_l3.unsqueeze(0) # (1, 2000, 128)
        #pos_embed_l3 = pos_embed_l3[:, :S,] # (1, 537, 128)
        
        pos = self.pos_embedding[:, :S,].repeat(B, 1, 1)
        
        
        #B, S, H, W = depth_features.shape  # (batch, sequence_length, 1, 224, 224)
        depth_inputs = depth_features.view(B, S, -1)  # (B, S, 50176)
        depth_inputs = self.depth_projection(depth_inputs)  # (B, S, hidden_dim)
        depth_inputs = self.depth_layernorm(depth_inputs)  # LayerNorm 적용
        depth_inputs = F.relu(depth_inputs)

        fused_features, _ = self.fuser({'rgb': src, 'depth': depth_inputs})
        
        #########################

        # pos = rearrange(pos, 'b t c -> t b c')
        # action_query = self.query_embed.weight
        # action_query = action_query.unsqueeze(0).repeat(B, 1, 1)#(8, 8, 128)
        
        # action_query = rearrange(action_query, 'b t c -> t b c') #(8, 8, 128)
        # tgt = torch.zeros_like(action_query)

        # src, tgt = self.transformer(src=fused_features, tgt=tgt, mask=src_key_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=None, query_embed=action_query, pos_embed=pos, tgt_pos_embed=None, epoch=epoch, idx=idx)

        # tgt = rearrange(tgt, 't b c -> b t c') # (8, 655, 128) -> [4, 5, 128]
        # src = rearrange(src, 't b c -> b t c')
        tgt = fused_features
        pooled_tgt = F.adaptive_avg_pool1d(tgt.permute(0, 2, 1), self.n_query).permute(0, 2, 1)

        
        output = dict()
        
        if self.args.anticipate :
            # action anticipation
            output_class = self.fc(pooled_tgt) 
            duration = self.fc_len(pooled_tgt) #[B, T, 1]
            duration = duration.squeeze(2) #[B, T]
            output['duration'] = duration
            output['action'] = output_class

        if False:#self.args.seg :
            # action segmentation
            tgt_seg = self.fc_seg(src)
            #tgt_seg = self.fc_l3(src)
            output['seg'] = tgt_seg

        #############ADDING ################
        #l3_logits = self.fc_l3(l3_logits)
        #l3_logits = self.fc_seg(l3_logits)
        #output['l3'] = l3_logits

        return output


def get_pad_mask(seq, pad_idx):
    return (seq ==pad_idx)



