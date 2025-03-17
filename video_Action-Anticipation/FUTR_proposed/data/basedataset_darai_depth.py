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
import cv2
import torchvision.models as models

class BaseDataset(Dataset):
    def __init__(self, vid_path, actions_dict, features_path, depth_features_path, gt_path, pad_idx, n_class,
                 n_query=8,  mode='train', obs_perc=0.2, args=None, query_dict=None, query_pad_idx=48):
        self.query_pad_idx = query_pad_idx
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.pad_idx = pad_idx
        self.features_path = features_path
        self.depth_features_path = depth_features_path
        self.gt_path = gt_path
        self.mode = mode
        self.sample_rate = args.sample_rate
        self.vid_list = list()
        
        # with open(split_file, 'r') as f:
        #     self.vid_list = [[line.strip()] for line in f.readlines()]
        self.n_query = n_query
        self.args = args
        self.NONE = self.n_class - 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        self.query_dict = query_dict
        
        self.all_sequences = []

        # Read vid_list file and store each line as a video name
        with open(vid_path, 'r') as f:
            vid_list = [line.strip() for line in f.readlines()]

        # Populate all_sequences with (vid_file, seq_idx, obs_perc) for each video
        
        for vid in vid_list:
            base_name = os.path.splitext(vid)[0]
            feature_depth_file = os.path.join(self.depth_features_path, f"{base_name}_1.npy")
            if 'camera_1_fps_15' in feature_depth_file:
                feature_depth_file = feature_depth_file.replace('camera_1_fps_15', 'depth_1')
            elif 'camera_2_fps_15' in feature_depth_file:
                feature_depth_file = feature_depth_file.replace('camera_2_fps_15', 'depth_2')
            seq_idx = 1
            while True:
                # Check for existence of sequence files
                gt_file = os.path.join(self.gt_path, f"{base_name}_{seq_idx}.txt")
                feature_file = os.path.join(self.features_path, f"{base_name}_{seq_idx}.npy")
                
                #print(gt_file, feature_file, feature_depth_file)
                if os.path.exists(gt_file) and os.path.exists(feature_file) and os.path.exists(feature_depth_file):
                    #print("all exists----")
                    with open(gt_file, 'r') as file_ptr:
                        lines = file_ptr.readlines()
                        if len(lines) > self.sample_rate:
                            # Append entries with different obs_perc values depending on mode
                            if mode in ['train', 'val']:
                                #self.all_sequences.append((vid, seq_idx, 0.04))
                                #self.all_sequences.append((vid, seq_idx, 0.05))
                                #self.all_sequences.append((vid, seq_idx, 0.1))
                                #self.all_sequences.append((vid, seq_idx, 0.15))
                                self.all_sequences.append((vid, seq_idx, 0.2))
                                #self.all_sequences.append((vid, seq_idx, 0.25))
                                self.all_sequences.append((vid, seq_idx, 0.3))
                                #self.all_sequences.append((vid, seq_idx, 0.4))
                                self.all_sequences.append((vid, seq_idx, 0.5))
                                #self.all_sequences.append((vid, seq_idx, 0.8))
                                #self.all_sequences.append((vid, seq_idx, 0.9))
                            elif mode == 'test':
                                self.all_sequences.append((vid, seq_idx, obs_perc))
                            seq_idx += 1
                        else:
                            break
                else:
                    break

    def __getitem__(self, idx):
        # Retrieve (vid_file, seq_idx, obs_perc) for the given index
        vid_file, seq_idx, obs_perc = self.all_sequences[idx]
        obs_perc = float(obs_perc)
        return self._make_input(vid_file, obs_perc, seq_idx=seq_idx)


    def _make_input(self, vid_file, obs_perc, seq_idx):
        gt_file = os.path.join(self.gt_path, f"{vid_file.split('.')[0]}_{seq_idx}.txt")
        feature_file = os.path.join(self.features_path, f"{vid_file.split('.')[0]}_{seq_idx}.npy")
        feature_depth_file = os.path.join(self.depth_features_path, f"{vid_file.split('.')[0]}_1.npy")
        if 'camera_1_fps_15' in feature_depth_file:
            feature_depth_file = feature_depth_file.replace('camera_1_fps_15', 'depth_1')
        elif 'camera_2_fps_15' in feature_depth_file:
            feature_depth_file = feature_depth_file.replace('camera_2_fps_15', 'depth_2')

        #print(gt_file, feature_file, feature_depth_file)
        with open(gt_file, 'r') as file_ptr:
            lines = file_ptr.readlines()
            valid_lines = [line.strip() for line in lines if len(line.strip().split(',')) == 3]
        
        image_filenames = [line.split(',')[0] for line in valid_lines]  
        image_indices = [int(os.path.basename(img).split('_')[-1].split('.')[0]) for img in image_filenames]

        # Load precomputed features from the .npy file
        features = np.load(feature_file)#.transpose()  # Transpose as needed
        depth_features = np.load(feature_depth_file)
        start_frame = image_indices[0]
        end_frame = image_indices[-1]
        depth_features = depth_features[start_frame:end_frame + 1]

        # Parsing labels from valid_lines
        all_content = [line.split(',')[1] for line in valid_lines]  # L2 labels ################### split(',')[1]
        query = [line.split(',')[2] for line in valid_lines]  # L3 labels
        #all_content = query
        vid_len = len(all_content)
        observed_len = int(obs_perc*vid_len)
        pred_len = int(0.5*vid_len)

        start_frame = 0

        # Feature and label slicing
        features = features[start_frame : start_frame + observed_len]  # Select observed range
        features = features[::self.sample_rate]  # Sample features by sample rate

        depth_features = depth_features[start_frame : start_frame + observed_len]  # Select observed range
        depth_features = depth_features[::self.sample_rate]  # Sample features by sample rate

        # Process query (L3) labels
        query = query[start_frame : start_frame + observed_len]  # Observed range
        query = query[::self.sample_rate]  # Sample by rate
        query_label = self.seq2idx4query(query)  # Convert L3 labels to indices

        # Process past content (L2) labels
        past_content = all_content[start_frame : start_frame + observed_len]  # Observed range
        past_content = past_content[::self.sample_rate]  # Sample by rate
        past_label = self.seq2idx(past_content)  # Convert L2 labels to indices

        # Ensure feature and label lengths are consistent
        if features.shape[0] != len(past_content):
            features = features[:len(past_content)]
        if depth_features.shape[0] != len(past_content):
            depth_features = depth_features[:len(past_content)]
        if len(query_label) != len(past_content):
            query_label = query_label[:len(past_content)]

        # Process future content
        #future_content_length = len(all_content)
        #future_content = all_content[future_content_length - 8 * 15: ]
        future_content = all_content[start_frame + observed_len: start_frame + observed_len + pred_len]  # Prediction range
        future_content = future_content[::self.sample_rate]  # Sample by rate
        trans_future, trans_future_dur = self.seq2transcript(future_content)
        trans_future = np.append(trans_future, self.NONE)
        trans_future_target = trans_future  # Target future sequence

        # Add padding for future sequence
        trans_seq_len = len(trans_future_target)
        diff = self.n_query - trans_seq_len
        if diff > 0:
            # Pad with pad_idx if needed
            trans_future_target = np.concatenate((trans_future_target, np.ones(diff) * self.pad_idx))
            trans_future_dur = np.concatenate((trans_future_dur, np.ones(diff + 1) * self.pad_idx))
        elif diff < 0:
            # Trim if sequence is too long
            trans_future_target = trans_future_target[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else:
            trans_future_dur = np.concatenate((trans_future_dur, np.ones(1) * self.pad_idx))

        # Create item dictionary
        item = {
            'features': torch.tensor(features, dtype=torch.float32),  # Features as a tensor
            'depth_features': torch.tensor(depth_features, dtype=torch.float32),
            'past_label': torch.tensor(past_label, dtype=torch.long),  # Past L2 labels
            'trans_future_dur': torch.tensor(trans_future_dur, dtype=torch.float32),  # Future durations
            'trans_future_target': torch.tensor(trans_future_target, dtype=torch.long),  # Future target sequence
        }

        return item


    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_dur = [item['trans_future_dur'] for item in batch]
        b_trans_future_target = [item['trans_future_target'] for item in batch]
        b_depth_features = [item['depth_features'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self.pad_idx)
        b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
                                                        padding_value=self.pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)

        b_depth_features = torch.nn.utils.rnn.pad_sequence(b_depth_features, batch_first=True, padding_value=0)
        batch = [b_features, b_depth_features, b_past_label, b_trans_future_dur, b_trans_future_target]

        return batch


    def __len__(self):
        return len(self.all_sequences)
        #return len(self.vid_list)

    def seq2idx4query(self, seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            l3_class_name = seq[i].replace(' ', '')
            idx[i] = self.query_dict[l3_class_name]
        return idx

    def seq2idx(self, seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            l2_class_name = seq[i].replace(' ', '')
            idx[i] = self.actions_dict[l2_class_name]
        return idx

    def seq2transcript(self, seq):
        transcript_action = []
        transcript_dur = []
        
        action = seq[0].replace(' ', '')
        #transcript_action.append(self.query_dict[action])
        transcript_action.append(self.actions_dict[action])
        
        last_i = 0
        for i in range(len(seq)):
            if action != seq[i]:
                action = seq[i].replace(' ', '')
                #transcript_action.append(self.query_dict[action])
                transcript_action.append(self.actions_dict[action])
                duration = (i-last_i)/len(seq)
                last_i = i
                transcript_dur.append(duration)
        duration = (len(seq)-last_i)/len(seq)
        transcript_dur.append(duration)
        return np.array(transcript_action), np.array(transcript_dur)