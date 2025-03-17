# import torch
# import numpy as np
# from torch.utils.data import Dataset
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from numpy.random import randint
# from utils import *
# import pdb
# import random
# import cv2
# import torchvision.models as models
# from PIL import Image
# import torchvision.transforms as transforms

# class BaseDataset(Dataset):
#     def __init__(self, vid_path, actions_dict, features_path, gt_path, pad_idx, n_class,
#                  n_query=8,  mode='train', obs_perc=0.2, args=None, query_dict=None, query_pad_idx=47):
#         self.query_pad_idx = query_pad_idx
#         self.n_class = n_class
#         self.actions_dict = actions_dict
#         self.pad_idx = pad_idx
#         self.features_path = features_path
#         self.gt_path = gt_path
#         self.mode = mode
#         self.sample_rate = args.sample_rate
#         self.vid_list = list()
        
#         # with open(split_file, 'r') as f:
#         #     self.vid_list = [[line.strip()] for line in f.readlines()]
#         self.n_query = n_query
#         self.args = args
#         self.NONE = self.n_class - 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
#         self.query_dict = query_dict
        
#         self.all_sequences = []

#         # Read vid_list file and store each line as a video name
#         with open(vid_path, 'r') as f:
#             vid_list = [line.strip() for line in f.readlines()]

#         # Populate all_sequences with (vid_file, seq_idx, obs_perc) for each video
#         for vid in vid_list:
#             base_name = os.path.splitext(vid)[0]
#             seq_idx = 1
#             while True:
#                 # Check for existence of sequence files
#                 gt_file = os.path.join(self.gt_path, f"{base_name}_{seq_idx}.txt")
#                 feature_file = os.path.join(self.features_path, f"{base_name}_{seq_idx}.npy")
                

#                 if os.path.exists(gt_file) and os.path.exists(feature_file):
#                     with open(gt_file, 'r') as file_ptr:
#                         lines = file_ptr.readlines()
#                         if len(lines) > self.sample_rate:
#                             # Append entries with different obs_perc values depending on mode
#                             if mode in ['train', 'val']:
#                                 #self.all_sequences.append((vid, seq_idx, 0.04))
#                                 #self.all_sequences.append((vid, seq_idx, 0.05))
#                                 #self.all_sequences.append((vid, seq_idx, 0.1))
#                                 #self.all_sequences.append((vid, seq_idx, 0.15))
#                                 self.all_sequences.append((vid, seq_idx, 0.2))
#                                 #self.all_sequences.append((vid, seq_idx, 0.25))
#                                 self.all_sequences.append((vid, seq_idx, 0.3))
#                                 #self.all_sequences.append((vid, seq_idx, 0.4))
#                                 self.all_sequences.append((vid, seq_idx, 0.5))
#                             elif mode == 'test':
#                                 self.all_sequences.append((vid, seq_idx, obs_perc))
#                             seq_idx += 1
#                         else:
#                             break
#                 else:
#                     break

#     def __getitem__(self, idx):
#         # Retrieve (vid_file, seq_idx, obs_perc) for the given index
#         vid_file, seq_idx, obs_perc = self.all_sequences[idx]
#         obs_perc = float(obs_perc)
#         return self._make_input(vid_file, obs_perc, seq_idx=seq_idx)
    
#     # def transform(self, image_path_list):
#     #     image_tensors = []
#     #     for path in image_path_list:
#     #         image = Image.open(path).convert("RGB")  # Load and convert to RGB
#     #         transform = transforms.Compose([
#     #             transforms.Resize((224, 224)),  # Resize to match model input size
#     #             transforms.ToTensor(),         # Convert to tensor
#     #         ])
#     #         image_tensor = transform(image)
#     #         image_tensors.append(image_tensor)

#     #     return image_tensors
    
#     def list_to_txt(self, image_path_list):
#         # open file
#         with open('/home/seulgi/work/darai-anticipation/FUTR_proposed/image_path.txt', 'w+') as f:
            
#             # write elements of list
#             for items in image_path_list:
#                 data_to_write = '\n'.join(image_path_list)
#                 f.write(data_to_write)
#             f.write('\n\n')
#         # close the file
#         f.close()

#     def transform(self, image_path_list):
#         """
#         Transform a list of image paths into a single stacked tensor.
#         Args:
#             image_path_list (list): List of image paths.
#         Returns:
#             torch.Tensor: A tensor containing all processed images.
#         """
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),  # Resize to match model input size
#             transforms.ToTensor(),         # Convert to tensor
#         ])

#         # Process all images and stack them into a single tensor
#         image_tensors = torch.stack([
#             transform(Image.open(path).convert("RGB")) for path in image_path_list
#         ])

#         return image_tensors



#     def _make_input(self, vid_file, obs_perc, seq_idx):
#         gt_file = os.path.join(self.gt_path, f"{vid_file.split('.')[0]}_{seq_idx}.txt")
#         feature_file = os.path.join(self.features_path, f"{vid_file.split('.')[0]}_{seq_idx}.npy")

#         with open(gt_file, 'r') as file_ptr:
#             lines = file_ptr.readlines()
#             valid_lines = [line.strip() for line in lines if len(line.strip().split(',')) == 3]
        
#         # Load precomputed features from the .npy file
#         features = np.load(feature_file)#.transpose()  # Transpose as needed

#         # Parsing labels from valid_lines
#         #input_image_path = [line.split(',')[0] for line in valid_lines]
#         all_content = [line.split(',')[1] for line in valid_lines]  # L2 labels ################### split(',')[1]
#         query = [line.split(',')[2] for line in valid_lines]  # L3 labels

#         vid_len = len(all_content)
#         observed_len = int(obs_perc*vid_len)
#         pred_len = int(0.5*vid_len)

#         start_frame = 0

#         # Feature and label slicing
#         features = features[start_frame : start_frame + observed_len]  # Select observed range
#         features = features[::self.sample_rate]  # Sample features by sample rate

#         # Process query (L3) labels
#         query = query[start_frame : start_frame + observed_len]  # Observed range
#         query = query[::self.sample_rate]  # Sample by rate
#         query_label = self.seq2idx4query(query)  # Convert L3 labels to indices

#         # Process past content (L2) labels
#         past_content = all_content[start_frame : start_frame + observed_len]  # Observed range
#         past_content = past_content[::self.sample_rate]  # Sample by rate
#         past_label = self.seq2idx(past_content)  # Convert L2 labels to indices

#         # input_image_path = input_image_path[start_frame: start_frame + observed_len]
#         # input_image_path = input_image_path[::self.sample_rate]

#         # self.list_to_txt(input_image_path)
        
#         # input_image_path = self.transform(input_image_path)

#         # Ensure feature and label lengths are consistent
#         if features.shape[0] != len(past_content):
#             features = features[:len(past_content)]
#         if len(query_label) != len(past_content):
#             query_label = query_label[:len(past_content)]
#         # if len(input_image_path) != len(past_content):
#         #     input_image_path = input_image_path[:len(past_content)]

#         # Process future content
#         future_content = all_content[start_frame + observed_len: start_frame + observed_len + pred_len]  # Prediction range
#         future_content = future_content[::self.sample_rate]  # Sample by rate
#         trans_future, trans_future_dur = self.seq2transcript(future_content)
#         trans_future = np.append(trans_future, self.NONE)
#         trans_future_target = trans_future  # Target future sequence

#         # Add padding for future sequence
#         trans_seq_len = len(trans_future_target)
#         diff = self.n_query - trans_seq_len
#         if diff > 0:
#             # Pad with pad_idx if needed
#             trans_future_target = np.concatenate((trans_future_target, np.ones(diff) * self.pad_idx))
#             trans_future_dur = np.concatenate((trans_future_dur, np.ones(diff + 1) * self.pad_idx))
#         elif diff < 0:
#             # Trim if sequence is too long
#             trans_future_target = trans_future_target[:self.n_query]
#             trans_future_dur = trans_future_dur[:self.n_query]
#         else:
#             trans_future_dur = np.concatenate((trans_future_dur, np.ones(1) * self.pad_idx))

#         # Create item dictionary
#         item = {
#             'features': torch.tensor(features, dtype=torch.float32),  # Features as a tensor
#             'past_label': torch.tensor(past_label, dtype=torch.long),  # Past L2 labels
#             'trans_future_dur': torch.tensor(trans_future_dur, dtype=torch.float32),  # Future durations
#             'trans_future_target': torch.tensor(trans_future_target, dtype=torch.long),  # Future target sequence
#             'query_label': torch.tensor(query_label, dtype=torch.long),  # L3 labels as query
#             #'image_path':input_image_path,
#         }

#         return item


#     def my_collate(self, batch):
#         '''custom collate function, gets inputs as a batch, output : batch'''

#         b_features = [item['features'] for item in batch]
#         b_past_label = [item['past_label'] for item in batch]
#         b_trans_future_dur = [item['trans_future_dur'] for item in batch]
#         b_trans_future_target = [item['trans_future_target'] for item in batch]
#         b_trans_query_label = [item['query_label'] for item in batch]
#         #b_image_path = [item['image_path'] for item in batch]

#         batch_size = len(batch)

#         b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
#         b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
#                                                          padding_value=self.pad_idx)
#         b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
#                                                         padding_value=self.pad_idx)
#         b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)

#         b_trans_query_label = torch.nn.utils.rnn.pad_sequence(b_trans_query_label, batch_first=True, padding_value=self.query_pad_idx)
#         #b_image_path = torch.nn.utils.rnn.pad_sequence(b_image_path, batch_first=True, padding_value=0)
#         batch = [b_features, b_past_label, b_trans_future_dur, b_trans_future_target, b_trans_query_label]

#         return batch


#     def __len__(self):
#         return len(self.all_sequences)
#         #return len(self.vid_list)

#     def seq2idx4query(self, seq):
#         idx = np.zeros(len(seq))
#         for i in range(len(seq)):
#             l3_class_name = seq[i].replace(' ', '')
#             idx[i] = self.query_dict[l3_class_name]
#         return idx

#     def seq2idx(self, seq):
#         idx = np.zeros(len(seq))
#         for i in range(len(seq)):
#             l2_class_name = seq[i].replace(' ', '')
#             idx[i] = self.actions_dict[l2_class_name]
#         return idx

#     def seq2transcript(self, seq):
#         transcript_action = []
#         transcript_dur = []
        
#         action = seq[0].replace(' ', '')
#         transcript_action.append(self.actions_dict[action])
        
#         last_i = 0
#         for i in range(len(seq)):
#             if action != seq[i]:
#                 action = seq[i].replace(' ', '')
#                 transcript_action.append(self.actions_dict[action])
#                 duration = (i-last_i)/len(seq)
#                 last_i = i
#                 transcript_dur.append(duration)
#         duration = (len(seq)-last_i)/len(seq)
#         transcript_dur.append(duration)
#         return np.array(transcript_action), np.array(transcript_dur)



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
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(Dataset):
    def __init__(self, vid_path, actions_dict, features_path, gt_path, pad_idx, n_class,
                 n_query=8,  mode='train', obs_perc=0.2, args=None, query_dict=None, query_pad_idx=48):
        self.query_pad_idx = 47
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.pad_idx = pad_idx
        self.features_path = features_path
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
            seq_idx = 1
            while True:
                # Check for existence of sequence files
                gt_file = os.path.join(self.gt_path, f"{base_name}_{seq_idx}.txt")
                feature_file = os.path.join(self.features_path, f"{base_name}_{seq_idx}.npy")
                

                if os.path.exists(gt_file) and os.path.exists(feature_file):
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

    def transform(self, image_path_list):
        """
        Transform a list of image paths into a single stacked tensor.
        Args:
            image_path_list (list): List of image paths.
        Returns:
            torch.Tensor: A tensor containing all processed images.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match model input size
            transforms.ToTensor(),         # Convert to tensor
        ])

        # Process all images and stack them into a single tensor
        image_tensors = torch.stack([
            transform(Image.open(path).convert("RGB")) for path in image_path_list
        ])

        return image_tensors

    def _make_input(self, vid_file, obs_perc, seq_idx):
        gt_file = os.path.join(self.gt_path, f"{vid_file.split('.')[0]}_{seq_idx}.txt")
        feature_file = os.path.join(self.features_path, f"{vid_file.split('.')[0]}_{seq_idx}.npy")

        with open(gt_file, 'r') as file_ptr:
            lines = file_ptr.readlines()
            valid_lines = [line.strip() for line in lines if len(line.strip().split(',')) == 3]
        
        # Load precomputed features from the .npy file
        features = np.load(feature_file)#.transpose()  # Transpose as needed

        # Parsing labels from valid_lines
        input_image_path = [line.split(',')[0] for line in valid_lines]
        all_content = [line.split(',')[1] for line in valid_lines]  # L2 labels ################### split(',')[1]
        query = [line.split(',')[2] for line in valid_lines]  # L3 labels

        vid_len = len(all_content)
        observed_len = int(obs_perc*vid_len)
        pred_len = int(0.5*vid_len)

        start_frame = 0

        # Feature and label slicing
        features = features[start_frame : start_frame + observed_len]  # Select observed range
        features = features[::self.sample_rate]  # Sample features by sample rate

        # Process query (L3) labels
        query = query[start_frame : start_frame + observed_len]  # Observed range
        query = query[::self.sample_rate]  # Sample by rate
        query_label = self.seq2idx4query(query)  # Convert L3 labels to indices

        # Process past content (L2) labels
        past_content = all_content[start_frame : start_frame + observed_len]  # Observed range
        past_content = past_content[::self.sample_rate]  # Sample by rate
        past_label = self.seq2idx(past_content)  # Convert L2 labels to indices

        input_image_path = input_image_path[start_frame: start_frame + observed_len]
        input_image_path = input_image_path[::self.sample_rate]
        input_image_path = self.transform(input_image_path)

        # Ensure feature and label lengths are consistent
        if features.shape[0] != len(past_content):
            features = features[:len(past_content)]
        if len(query_label) != len(past_content):
            query_label = query_label[:len(past_content)]

        if len(input_image_path) != len(past_content):
            input_image_path = input_image_path[:len(past_content)]

        # Process future content
        future_content_length = len(all_content)
        future_content = all_content[start_frame + observed_len: start_frame + observed_len + 8*self.sample_rate]
        #future_content = all_content[start_frame + observed_len: start_frame + observed_len + pred_len]  # Prediction range
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
            'past_label': torch.tensor(past_label, dtype=torch.long),  # Past L2 labels
            'trans_future_dur': torch.tensor(trans_future_dur, dtype=torch.float32),  # Future durations
            'trans_future_target': torch.tensor(trans_future_target, dtype=torch.long),  # Future target sequence
            'query_label': torch.tensor(query_label, dtype=torch.long),  # L3 labels as query
            'image_path': input_image_path,
        }

        return item


    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_dur = [item['trans_future_dur'] for item in batch]
        b_trans_future_target = [item['trans_future_target'] for item in batch]
        b_trans_query_label = [item['query_label'] for item in batch]
        b_image_path = [item['image_path'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self.pad_idx)
        b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
                                                        padding_value=self.pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)

        b_trans_query_label = torch.nn.utils.rnn.pad_sequence(b_trans_query_label, batch_first=True, padding_value=self.query_pad_idx)
        b_image_path = torch.nn.utils.rnn.pad_sequence(b_image_path, batch_first=True, padding_value=0)

        batch = [b_features, b_past_label, b_trans_future_dur, b_trans_future_target, b_trans_query_label, b_image_path]

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
        transcript_action.append(self.actions_dict[action])
        
        last_i = 0
        for i in range(len(seq)):
            if action != seq[i]:
                action = seq[i].replace(' ', '')
                transcript_action.append(self.actions_dict[action])
                duration = (i-last_i)/len(seq)
                last_i = i
                transcript_dur.append(duration)
        duration = (len(seq)-last_i)/len(seq)
        transcript_dur.append(duration)
        return np.array(transcript_action), np.array(transcript_dur)




