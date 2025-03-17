# import torch
# import torch.nn as nn
# import numpy
# import pdb
# import os
# import copy
# from collections import defaultdict
# import numpy as np
# from utils import normalize_duration, eval_file


# def predict(model, vid_list, args, obs_p, n_class, actions_dict, device, query_dict):
#     def seq2idx4query(seq):
#         idx = np.zeros(len(seq))
#         for i in range(len(seq)):
#             idx[i] = query_dict[seq[i]]
#         return idx
    
    

#     model.eval()
#     with torch.no_grad():
#         data_path = './datasets'
#         if args.dataset == 'breakfast':
#             data_path = os.path.join(data_path, 'breakfast')
#         elif args.dataset == '50salads':
#             data_path = os.path.join(data_path, '50salads')
#         elif args.dataset == 'darai':
#             data_path = os.path.join(data_path, 'darai')
#         gt_path = os.path.join(data_path, 'groundTruth')
#         features_path = os.path.join(data_path, 'features')

#         eval_p = [0.1, 0.2, 0.3, 0.5]
#         pred_p = 0.5
#         sample_rate = args.sample_rate
#         NONE = n_class-1
#         T_actions = np.zeros((len(eval_p), len(actions_dict)))
#         F_actions = np.zeros((len(eval_p), len(actions_dict)))
#         actions_dict_with_NONE = copy.deepcopy(actions_dict)
#         actions_dict_with_NONE['NONE'] = NONE

#         for vid in vid_list:
#             file_name = vid.split('/')[-1].split('.')[0]

#             # load ground truth actions
#             gt_file = os.path.join(gt_path, file_name+'.txt')
#             gt_read = open(gt_file, 'r')
#             # gt_seq = gt_read.read().split('\n')[:-1]
#             # gt_read.close()
#             with open(gt_file, 'r') as file_ptr:
#                 lines = file_ptr.readlines()
#                 valid_lines = [line.strip() for line in lines if len(line.strip().split(',')) == 3]

#             all_content = [line.split(',')[1] for line in valid_lines]  # L2 labels
#             query = [line.split(',')[2] for line in valid_lines]  # L3 labels

            
#             file_length = len(all_content)

#             # load features
#             features_file = os.path.join(features_path, file_name+'.npy')
#             features = np.load(features_file)#.transpose()

#             vid_len = len(all_content)
#             past_len = int(obs_p*vid_len)
#             future_len = int(pred_p*vid_len)

#             past_seq = all_content[:past_len]
#             features = features[:past_len]
#             inputs = features[::sample_rate, :]
#             inputs = torch.Tensor(inputs).to(device)

#             query = query[:past_len]
#             query = query[::sample_rate]
#             query_label = seq2idx4query(query)
#             query_label = torch.Tensor(query_label).to(device)

#             outputs = model(inputs=inputs.unsqueeze(0), query=query_label.unsqueeze(0), mode='test')
#             output_action = outputs['action']
#             output_dur = outputs['duration']
#             output_label = output_action.max(-1)[1]

#             # fine the forst none class
#             none_mask = None
#             none_idx = None
#             for i in range(output_label.size(1)) :
#                 if output_label[0,i] == NONE :
#                     none_idx = i
#                     break
#                 else :
#                     none = None
#             if none_idx is not None :
#                 none_mask = torch.ones(output_label.shape).type(torch.bool)
#                 none_mask[0, none_idx:] = False
#                 # Apply normalize_duration only if none_mask is available
#                 output_dur = normalize_duration(output_dur, none_mask.to(device))
#             else:
#                 # If none_idx is None, use default normalized duration
#                 output_dur = normalize_duration(output_dur, torch.ones_like(output_dur).to(device))

#             #output_dur = normalize_duration(output_dur, none_mask.to(device))

#             pred_len = (0.5+future_len*output_dur).squeeze(-1).long()

#             pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
#             predicted = torch.ones(future_len)
#             action = output_label.squeeze()

#             for i in range(len(action)) :
#                 predicted[int(pred_len[i]) : int(pred_len[i] + pred_len[i+1])] = action[i]
#                 pred_len[i+1] = pred_len[i] + pred_len[i+1]
#                 if i == len(action) - 1 :
#                     predicted[int(pred_len[i]):] = action[i]


#             prediction = past_seq
#             for i in range(len(predicted)):
#                 prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]]))

#             #evaluation
#             for i in range(len(eval_p)):
#                 p = eval_p[i]
#                 eval_len = int((obs_p+p)*vid_len)
#                 eval_prediction = prediction[:eval_len]
#                 T_action, F_action = eval_file(all_content, eval_prediction, obs_p, actions_dict)
#                 T_actions[i] += T_action
#                 F_actions[i] += F_action

#         results = []
#         for i in range(len(eval_p)):
#             acc = 0
#             n = 0
#             for j in range(len(actions_dict)):
#                 total_actions = T_actions + F_actions
#                 if total_actions[i,j] != 0:
#                     acc += float(T_actions[i,j]/total_actions[i,j])
#                     n+=1

#             result = 'obs. %d '%int(100*obs_p) + 'pred. %d '%int(100*eval_p[i])+'--> MoC: %.4f'%(float(acc)/n)
#             results.append(result)
#             print(result)
#         print('--------------------------------')

#         return


import torch
import torch.nn as nn
import numpy as np
import os
import copy
from collections import defaultdict
from utils import normalize_duration, eval_file
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


def accuracy(pred, gold, actions_dict):
    pred = pred[0]
    total_correct = 0

    assert len(gold) == len(pred)
    length = min(len(gold), len(pred))
    for i in range(length):
        gt = actions_dict[gold[i].replace(' ', '')]
        if pred[i].item() == gt:
            total_correct += 1
    return total_correct / length

    

def weighted_accuracy(pred, gold, t_n_labels, actions_dict, image_base=None, image_target=None, gif_name=None, weight_same=1.0, weight_different=10.0):
    '''Calculate weighted accuracy based on comparison between t+n and t+m labels.'''
    pred = pred[0]#.max(1)[1]

    frames = []

    save_path = './save_dir/darai/visualization/'

    total_weighted_correct = 0
    total_weighted_labels = 0

    weight = weight_different if gold[0] != t_n_labels[0] else weight_same
    length = min(len(gold), len(pred))

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, max(len(image_base), len(image_target)), height_ratios=[2, 2, 1])
    # visualize image base
    for j, img_path in enumerate(image_base):
        img = Image.open(img_path)
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img)
        ax.axis('off')
    fig.text(0.5, 0.85, 'Predict Before', ha='center', fontsize=14, fontweight='bold')


    for i in range(length):
        gt = actions_dict[gold[i].replace(' ', '')]
        target_img = Image.open(image_target[i])
        ax_target = fig.add_subplot(gs[1, i])
        ax_target.imshow(target_img)
        ax_target.axis('off')

        if pred[i].item() == gt:
            total_weighted_correct += weight
            # the network is correct: save the images, pred, gt.
            label_color = 'blue'
        else:
            # the network is wrong: save the images, pred, gt.
            label_color = 'red'

        # 하단에 레이블 추가
        ax_label = fig.add_subplot(gs[2, i])
        ax_label.axis('off')
        ax_label.text(
            0.5, 0.5,
            f"GT: {gt} | Pred: {pred[i].item()}",
            color=label_color,
            ha='center', va='center',
            fontsize=12, fontweight='bold'
        )
        total_weighted_labels += weight

    # 이미지 저장
    plt.tight_layout()
    save_file_path = os.path.join(save_path, f"visualization_{i}.png")
    plt.savefig(save_file_path)
    plt.close()

    weighted_accuracy = total_weighted_correct / total_weighted_labels if total_weighted_labels > 0 else 0
    return weighted_accuracy

def predict(model, vid_list, args, obs_p, n_class, actions_dict, device, query_dict):
    def seq2idx4query(seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = query_dict[seq[i].replace(' ', '')]
        return idx

    acc = 0
    idx = 0
    model.eval()
    with torch.no_grad():
        data_path = './datasets'
        if args.dataset == 'breakfast':
            data_path = os.path.join(data_path, 'breakfast')
        elif args.dataset == '50salads':
            data_path = os.path.join(data_path, '50salads')
        elif args.dataset == 'darai':
            data_path = os.path.join(data_path, 'darai')
        gt_path = os.path.join(data_path, 'groundTruth_nov11')
        features_path = os.path.join(data_path, 'features_temp')

        eval_p = [0.1, 0.2, 0.3, 0.5]
        pred_p = 0.5
        sample_rate = args.sample_rate
        NONE = n_class - 1
        T_actions = np.zeros((len(eval_p), len(actions_dict)))
        F_actions = np.zeros((len(eval_p), len(actions_dict)))
        actions_dict_with_NONE = copy.deepcopy(actions_dict)
        actions_dict_with_NONE['NONE'] = NONE
        t_actions = 0
        f_actions = 0

        for vid in vid_list:
            base_name = vid.split('/')[-1].split('.')[0]
            seq_idx = 1

            while True:
                # Check if gt and feature files with the sequence index exist
                gt_file = os.path.join(gt_path, f"{base_name}_{seq_idx}.txt")
                features_file = os.path.join(features_path, f"{base_name}_{seq_idx}.npy")

                if not os.path.exists(gt_file) or not os.path.exists(features_file):
                    break  # Exit loop if no more sequence files exist for this video

                # Load ground truth actions for this sequence
                with open(gt_file, 'r') as file_ptr:
                    lines = file_ptr.readlines()
                    valid_lines = [line.strip() for line in lines if len(line.strip().split(',')) == 3]
                
                image_path = [line.split(',')[0] for line in valid_lines] # images
                all_content = [line.split(',')[1] for line in valid_lines]  # L2 labels
                query = [line.split(',')[2] for line in valid_lines]  # L3 labels
                file_length = len(all_content)

                # Load features
                features = np.load(features_file)

                vid_len = len(all_content)
                past_len = int(obs_p * vid_len)
                future_len = int(pred_p * vid_len)

                past_seq = all_content[:past_len]
                features = features[:past_len]
                inputs = features[::sample_rate, :]
                inputs = torch.Tensor(inputs).to(device)

                query = query[:past_len]
                query = query[::sample_rate]
                query_label = seq2idx4query(query)
                query_label = torch.Tensor(query_label).to(device)

                future_content = all_content[past_len: past_len + future_len]
                future_content = future_content[::sample_rate]

                ## for visualize: base images.
                image_base = image_path[:past_len]
                image_base = image_base[::sample_rate]

                ## for visualize: images that needs to be anticipated.
                image_target = image_path[past_len: past_len + future_len]
                image_target = image_target[::sample_rate]

                # Model inference
                outputs = model(inputs=inputs.unsqueeze(0), query=query_label.unsqueeze(0), mode='test')

                ## Action Anticipation
                output_action = outputs['action']
                output_dur = outputs['duration']
                output_label = output_action.max(-1)[1]

                acc += weighted_accuracy(output_label, future_content, past_seq[-1], actions_dict, image_base, image_target, f"{base_name}_{seq_idx}")
                idx += 1
                # Find the first NONE class
                none_mask = None
                none_idx = None
                for i in range(output_label.size(1)):
                    if output_label[0, i] == NONE:
                        none_idx = i
                        break
                if none_idx is not None:
                    none_mask = torch.ones(output_label.shape).type(torch.bool)
                    none_mask[0, none_idx:] = False
                    output_dur = normalize_duration(output_dur, none_mask.to(device))
                else:
                    output_dur = normalize_duration(output_dur, torch.ones_like(output_dur).to(device))

                pred_len = (0.5 + future_len * output_dur).squeeze(-1).long()
                pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
                predicted = torch.ones(future_len)
                action = output_label.squeeze()

                for i in range(len(action)):
                    predicted[int(pred_len[i]): int(pred_len[i] + pred_len[i + 1])] = action[i]
                    pred_len[i + 1] = pred_len[i] + pred_len[i + 1]
                    if i == len(action) - 1:
                        predicted[int(pred_len[i]):] = action[i]

                # Combine past sequence and predicted sequence
                prediction = past_seq
                for i in range(len(predicted)):
                    prediction = np.concatenate(
                        (prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]])
                    )

                # Evaluation
                for i in range(len(eval_p)):
                    p = eval_p[i]
                    eval_len = int((obs_p + p) * vid_len)
                    eval_prediction = prediction[:eval_len]
                    T_action, F_action = eval_file(all_content, eval_prediction, obs_p, actions_dict)
                    T_actions[i] += T_action
                    F_actions[i] += F_action
                    t_actions += T_action
                    f_actions += F_action

                seq_idx += 1  # Move to the next sequence file for the current video

        print("!!!!!!!!!!!!!!", acc/idx)
        # Calculate and print results
        results = []
        for i in range(len(eval_p)):
            acc = 0
            n = 0
            for j in range(len(actions_dict)):
                total_actions = T_actions + F_actions
                TOTAL_ACTIONS = t_actions + f_actions
                #print("----- ", t_actions, f_actions, TOTAL_ACTIONS)
                if total_actions[i, j] != 0:
                    acc += float(T_actions[i, j] / total_actions[i, j])
                    n += 1

            result = f'obs. {int(100 * obs_p)}% pred. {int(100 * eval_p[i])}% --> MoC: {float(acc) / n:.4f}'
            results.append(result)
            print(result)
        print('--------------------------------')

        return




