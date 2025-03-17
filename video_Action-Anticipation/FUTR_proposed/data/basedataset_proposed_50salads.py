import torch
import numpy as np
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from numpy.random import randint
from utils import *
import pdb
import random

action_mapping = {
    "cut_and_mix_ingredients": [
        "peel_cucumber",
        "cut_cucumber",
        "place_cucumber_into_bowl",
        "cut_tomato",
        "place_tomato_into_bowl",
        "cut_cheese",
        "place_cheese_into_bowl",
        "cut_lettuce",
        "place_lettuce_into_bowl",
        "mix_ingredients"
    ],
    "prepare_dressing": [
        "add_oil",
        "add_vinegar",
        "add_salt",
        "add_pepper",
        "mix_dressing"
    ],
    "serve_salad": [
        "serve_salad_onto_plate",
        "add_dressing"
    ],
    "action_end": [
        "action_end"
    ],
    "action_start": [
        "action_start"
    ]
}

def change_query_dict_2_action_dict(query_dict, action_mapping):
    """
    Matches actions in query_dict to their high-level activities based on the provided action_mapping.

    Args:
        query_dict (dict): Dictionary of actions and their corresponding indices.
        action_mapping (dict): Dictionary where keys are high-level activities, and values are lists of low-level actions.

    Returns:
        dict: A dictionary mapping high-level activities to the indices of their corresponding low-level actions.
    """
    high_level_to_indices = []
    for query_action in query_dict:
        for high_level, low_level_list in action_mapping.items():
            # Check if the low-level action exists in query_dict
            for low_level_action in low_level_list:
                # Match low-level action ignoring placeholders like {_prep, _core, _post}
                if low_level_action in query_action:
                    high_level_to_indices.append(high_level)

    return high_level_to_indices


class BaseDataset(Dataset):
    def __init__(self, vid_list, actions_dict, features_path, gt_path, pad_idx, n_class,
                 n_query=8,  mode='train', obs_perc=0.2, args=None, query_dict=None):
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.pad_idx = pad_idx
        self.features_path = features_path
        self.gt_path = gt_path
        self.mode = mode
        self.sample_rate = args.sample_rate
        self.vid_list = list()
        self.n_query = n_query
        self.args = args
        self.NONE = self.n_class - 1

        self.query_dict = query_dict

        if self.mode == 'train' or self.mode == 'val':
            for vid in vid_list:
                self.vid_list.append([vid, .2])
                self.vid_list.append([vid, .3])
                self.vid_list.append([vid, .5])
                #self.vid_list.append([vid, .1])
        elif self.mode == 'test' :
            for vid in vid_list:
                self.vid_list.append([vid, obs_perc])

        self._make_input(vid, 0.2)


    def __getitem__(self, idx):
        vid_file, obs_perc = self.vid_list[idx]
        obs_perc = float(obs_perc)
        item = self._make_input(vid_file, obs_perc)
        return item


    def _make_input(self, vid_file, obs_perc ):
        vid_file = vid_file.split('/')[-1]
        vid_name = vid_file

        gt_file = os.path.join(self.gt_path, vid_file)
        feature_file = os.path.join(self.features_path, vid_file.split('.')[0]+'.npy')
        features = np.load(feature_file)
        features = features.transpose()

        file_ptr = open(gt_file, 'r')
        query = file_ptr.read().split('\n')[:-1]
        
        ##############################################################
        # Custom
        all_content = change_query_dict_2_action_dict(query, action_mapping)
        ##############################################################
        #all_content = file_ptr.read().split('\n')[:-1]
        
        vid_len = len(all_content)
        observed_len = int(obs_perc*vid_len)
        pred_len = int(0.5*vid_len)

        start_frame = 0

        # feature slicing
        features = features[start_frame : start_frame + observed_len] #[S, C]
        features = features[::self.sample_rate]

        query = query[start_frame : start_frame + observed_len]
        query = query[::self.sample_rate]
        query_label = self.seq2idx4query(query) # [8,]

        past_content = all_content[start_frame : start_frame + observed_len] #[S]
        past_content = past_content[::self.sample_rate]
        past_label = self.seq2idx(past_content)

        if np.shape(features)[0] != len(past_content) :
            features = features[:len(past_content),]

        if np.shape(query_label)[0] != len(past_content) :
            query_label = query_label[:len(past_content),]

        future_content = \
        all_content[start_frame + observed_len: start_frame + observed_len + pred_len] #[T]
        future_content = future_content[::self.sample_rate]
        trans_future, trans_future_dur = self.seq2transcript(future_content)
        trans_future = np.append(trans_future, self.NONE)
        trans_future_target = trans_future #target


        # add padding for future input seq
        trans_seq_len = len(trans_future_target)
        diff = self.n_query - trans_seq_len
        if diff > 0 :
            tmp = np.ones(diff)*self.pad_idx
            trans_future_target = np.concatenate((trans_future_target, tmp))
            tmp_len = np.ones(diff+1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))
        elif diff < 0 :
            trans_future_target = trans_future_target[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else :
            tmp_len = np.ones(1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))


        item = {'features':torch.Tensor(features), # features: (65, 2048)
                'past_label':torch.Tensor(past_label), # past label: (65,)
                'trans_future_dur':torch.Tensor(trans_future_dur), # trans_future_dur: (8, )
                'trans_future_target' : torch.Tensor(trans_future_target), # trans_future_target: (8,)
                'query_label':torch.Tensor(query_label),
                }

        return item


    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_dur = [item['trans_future_dur'] for item in batch]
        b_trans_future_target = [item['trans_future_target'] for item in batch]
        b_trans_query_label = [item['query_label'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self.pad_idx)
        b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
                                                        padding_value=self.pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)

        b_trans_query_label = torch.nn.utils.rnn.pad_sequence(b_trans_query_label, batch_first=True, padding_value=19)
        #print(b_trans_query_label)

        batch = [b_features, b_past_label, b_trans_future_dur, b_trans_future_target, b_trans_query_label]

        return batch


    def __len__(self):
        return len(self.vid_list)

    def seq2idx4query(self, seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = self.query_dict[seq[i]]
        return idx

    def seq2idx(self, seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = self.actions_dict[seq[i]]
        return idx

    def seq2transcript(self, seq):
        transcript_action = []
        transcript_dur = []
        action = seq[0]
        transcript_action.append(self.actions_dict[action])
        last_i = 0
        for i in range(len(seq)):
            if action != seq[i]:
                action = seq[i]
                transcript_action.append(self.actions_dict[action])
                duration = (i-last_i)/len(seq)
                last_i = i
                transcript_dur.append(duration)
        duration = (len(seq)-last_i)/len(seq)
        transcript_dur.append(duration)
        return np.array(transcript_action), np.array(transcript_dur)








