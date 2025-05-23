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

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class FUTR(nn.Module):

    # Breakfast: query_num: 48
    # Darai: query_num: 48
    # 50salads: query_num: 19
    def __init__(self, n_class, hidden_dim, src_pad_idx, device, args, n_query=8, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, query_num=48):
        super().__init__()
        self.query_mask = query_num - 1
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Linear(args.input_dim, hidden_dim)
        self.transformer = Transformer(hidden_dim, n_head, num_encoder_layers, num_decoder_layers,
                                        hidden_dim*4, normalize_before=False)
        self.n_query = n_query
        self.args = args
        nn.init.xavier_uniform_(self.input_embed.weight)
        self.query_embed = nn.Embedding(query_num, hidden_dim) #(8, 128)
        #self.linear = nn.Linear(863,8)
        

        if args.seg :
            self.fc_seg = nn.Linear(hidden_dim, n_class-1) #except SOS, EOS
            nn.init.xavier_uniform_(self.fc_seg.weight)

        if args.anticipate :
            self.fc = nn.Linear(hidden_dim, n_class)
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc_len = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_len.weight)

        #if args.pos_emb:
        #pos embedding
        max_seq_len = args.max_pos_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.xavier_uniform_(self.pos_embedding)
        # Sinusoidal position encoding
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.positional_embedding_l3 = self.sinusoidal_positional_encoding(max_seq_len, hidden_dim)
        self.positional_embedding_l3 = self.positional_embedding_l3.to(self.device)

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


    def forward(self, inputs, query, mode='train', epoch=0, idx=0):
        #query_min, query_max = query.min(), query.max()
        #normalized_query = ((query - query_min) * (self.n_query - 1) / (query_max - query_min)).long().to(self.device) #(8, 1142)
        query = query.long().to(self.device)
        
        query_mask = get_pad_mask(query, self.query_mask).to(self.device)
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

        #print(query, src_label)
        tgt_mask = None
        #src = src.to(self.device)
        #src = src.squeeze(1)
        if self.args.input_type == 'i3d_transcript':
            B, S, C = src.size()
            src = self.input_embed(src) #[B, S, C]
        elif self.args.input_type == 'gt':
            B, S = src.size()
            src = self.gt_emb(src)
        src = F.relu(src)

        # action query embedding
        #action_query = self.query_embed.weight # (8, 128)
        action_query = self.query_embed(query) # (8, 1142, 128)
        pos_embed_l3 = self.positional_embedding_l3.unsqueeze(0) # (1, 2000, 128)
        pos_embed_l3 = pos_embed_l3[:, :S,] # (1, 537, 128)
        action_query = pos_embed_l3.to(self.device) + action_query
        #action_query = action_query.unsqueeze(0).repeat(B, 1, 1)#(8, 8, 128)
        tgt = torch.zeros_like(action_query)

        # pos embedding
        pos = self.pos_embedding[:, :S,].repeat(B, 1, 1) ## self.pos_embedding : (1, 2000, 128)
        src = rearrange(src, 'b t c -> t b c')
        tgt = rearrange(tgt, 'b t c -> t b c')
        pos = rearrange(pos, 'b t c -> t b c')
        action_query = rearrange(action_query, 'b t c -> t b c') #(8, 8, 128)
        #query_mask = query_mask.T.unsqueeze(-1).expand_as(action_query)
        #action_query_mask = action_query.masked_fill(query_mask, 1000.0)

        
        src, tgt = self.transformer(src, tgt, src_key_padding_mask, tgt_mask, None, action_query, pos, None, epoch=epoch, idx=idx)
        

        tgt = rearrange(tgt, 't b c -> b t c') # (8, 655, 128) -> [4, 5, 128]
        src = rearrange(src, 't b c -> b t c')

        pooled_tgt = F.adaptive_avg_pool1d(tgt.permute(0, 2, 1), self.n_query).permute(0, 2, 1)

        output = dict()
        if self.args.anticipate :
            # action anticipation
            output_class = self.fc(pooled_tgt) #[T, B, C] # output_class: (8, 655, 11)
            duration = self.fc_len(pooled_tgt) #[B, T, 1]
            duration = duration.squeeze(2) #[B, T]
            output['duration'] = duration
            output['action'] = output_class

        if self.args.seg :
            # action segmentation
            tgt_seg = self.fc_seg(src)
            output['seg'] = tgt_seg

        #output['supcon'] = rearrange(action_query, 't b c -> b t c')
        return output


def get_pad_mask(seq, pad_idx):
    return (seq ==pad_idx)



