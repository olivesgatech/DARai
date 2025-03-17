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
import torchvision.models as models

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

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
        
        resnet50 = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])  # remove the last FC layer
        self.feature_extractor.eval()
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = True
        
        
        nn.init.xavier_uniform_(self.input_embed.weight)
        self.l3_attention = nn.MultiheadAttention(hidden_dim, n_head, batch_first=True)
        self.query_attention = nn.MultiheadAttention(hidden_dim, n_head, batch_first=True)
        

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

        self.depth_projection = nn.Linear(160 * 120, hidden_dim)
        #self.depth_projection = nn.Linear(224 * 224, hidden_dim)  # (1, 224, 224) → (hidden_dim)
        nn.init.xavier_uniform_(self.depth_projection.weight)
        self.depth_layernorm = nn.LayerNorm(hidden_dim)  # 추가된 LayerNorm


        if args.input_type =='gt':
            self.gt_emb = nn.Embedding(n_class+2, self.hidden_dim, padding_idx=n_class+1)
            nn.init.xavier_uniform_(self.gt_emb.weight)

    def extract_features(self, src):
        B, S, C, H, W = src.shape
        #print(src.shape)
        src = src.view(B * S, C, H, W)
        src = self.feature_extractor(src)  # (B*S, 2048, 1, 1)
        src = src.view(B, S, 2048)  # (B, S, 2048)
        return src

    def sinusoidal_positional_encoding(self, seq_len, emb_dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pos_embed = torch.zeros(seq_len, emb_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term) # apply sine to even indices
        pos_embed[:, 1::2] = torch.cos(position * div_term) # apply cosine to odd indices
        return pos_embed


    def forward(self, inputs, depth_features, mode='train', epoch=0, idx=0):
        #print(inputs[0].shape, depth_features.shape)
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
        src = src.to(self.device)
        if self.args.input_type == 'i3d_transcript':
            #print(src.shape)
            src = src.permute(0, 1, 4, 2, 3)
            src = self.extract_features(src) # (B, S, 2048)
            B, S, C = src.size()
            src = self.input_embed(src) #[B, S, C]
        elif self.args.input_type == 'gt':
            B, S = src.size()
            src = self.gt_emb(src)
        src = F.relu(src)
        src = self.pos_enc(src)
       
        #pos_embed_l3 = self.positional_embedding_l3.unsqueeze(0) # (1, 2000, 128)
        #pos_embed_l3 = pos_embed_l3[:, :S,] # (1, 537, 128)
        
        pos = self.pos_embedding[:, :S,].repeat(B, 1, 1)
        src = rearrange(src, 'b t c -> t b c')
        
        B, S, H, W = depth_features.shape  # (batch, sequence_length, 1, 224, 224)
        depth_inputs = depth_features.view(B, S, -1)  # (B, S, 50176)
        depth_inputs = self.depth_projection(depth_inputs)  # (B, S, hidden_dim)
        depth_inputs = self.depth_layernorm(depth_inputs)  # LayerNorm 적용
        depth_inputs = F.relu(depth_inputs)

        action_query = self.pos_enc_depth(depth_inputs) #pos_embed_l3.to(self.device) + depth_inputs
        #########################


        ######################### Multi-modal ###########################
        #action_query = l3_logits# + rearrange(src, 't b c -> b t c')
        #action_query = F.adaptive_avg_pool1d(action_query.permute(0, 2, 1), self.n_query).permute(0, 2, 1)
        #################################################################

        pos = rearrange(pos, 'b t c -> t b c')
        action_query = rearrange(action_query, 'b t c -> t b c') #(8, 8, 128)
        tgt = torch.zeros_like(action_query)

        src, tgt = self.transformer(src=src, tgt=tgt, mask=src_key_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=None, query_embed=action_query, pos_embed=pos, tgt_pos_embed=None, epoch=epoch, idx=idx)

        tgt = rearrange(tgt, 't b c -> b t c') # (8, 655, 128) -> [4, 5, 128]
        src = rearrange(src, 't b c -> b t c')
        
        pooled_tgt = F.adaptive_avg_pool1d(tgt.permute(0, 2, 1), self.n_query).permute(0, 2, 1)

        
        output = dict()
        if self.args.anticipate :
            # action anticipation
            output_class = self.fc(pooled_tgt) 
            duration = self.fc_len(pooled_tgt) #[B, T, 1]
            duration = duration.squeeze(2) #[B, T]
            output['duration'] = duration
            output['action'] = output_class

        if self.args.seg :
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



