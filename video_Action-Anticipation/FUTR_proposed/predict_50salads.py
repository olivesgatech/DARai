import torch
import torch.nn as nn
import numpy
import pdb
import os
import copy
from collections import defaultdict
import numpy as np
from utils import normalize_duration, eval_file, cal_performance

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


def normal_accuracy_without_gif(pred, gold, actions_dict, log=None, exclude_class_idx=None, image_base=None, label_base=None, image_target=None, gif_name=None, duration=0.2, weight_same=1.0, weight_different=10.0):
    '''Calculate weighted accuracy based on comparison between t+n and t+m labels.'''
    #pred = pred[0]

    total_correct = 0
    #assert len(gold) == len(pred)
    length = min(len(gold), len(pred))
    #print("-----------------------------")
    #print("length: ", length)
    
    idx = 0
    for i in range(length):
        gt = actions_dict[gold[i].replace(' ', '')]
        # if exclude_class_idx is not None and pred[i].item() == exclude_class_idx:
        #     if log is not None:
        #         log.write(f"\tgt: {gold[i].replace(' ', '')}\t, prediction: {pred[i].item()}\n")

        #     continue
        idx += 1

        if pred[i].item() == gt:
            total_correct += 1
        if log is not None:
            log.write(f"\tgt: {gold[i].replace(' ', '')}\t, prediction: {pred[i].item()}\n")

    #accuracy = total_correct / idx
    #print("accuracy: ", accuracy)
    # if accuracy == 0.0:
    #     print("gt: ", gold[0])
    #     print("pred: ", pred[0].item())
    # print("-----------------------------")
    return total_correct, idx

def seq2idx(seq, actions_dict):
    idx = np.zeros(len(seq))
    for i in range(len(seq)):
        idx[i] = actions_dict[seq[i]]
    return idx

def predict(model, vid_list, args, obs_p, n_class, actions_dict, device, query_dict):
    def seq2idx4query(seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = query_dict[seq[i].replace(' ', '')]
        return idx
    model.eval()
    seg_acc = 0
    ant_acc = 0
    idx = 0
    with torch.no_grad():
        data_path = './datasets'
        if args.dataset == 'breakfast':
            data_path = os.path.join(data_path, 'breakfast')
        elif args.dataset == '50salads':
            data_path = os.path.join(data_path, '50salads')
        elif args.dataset == 'darai':
            data_path = os.path.join(data_path, 'darai')
        gt_path = os.path.join(data_path, 'groundTruth')
        features_path = os.path.join(data_path, 'features')

        eval_p = [0.1, 0.2, 0.3, 0.5]
        pred_p = 0.5
        sample_rate = args.sample_rate
        NONE = n_class-1
        T_actions = np.zeros((len(eval_p), len(actions_dict)))
        F_actions = np.zeros((len(eval_p), len(actions_dict)))
        actions_dict_with_NONE = copy.deepcopy(actions_dict)
        actions_dict_with_NONE['NONE'] = NONE

        log_idx = 0
        total_correct_seg = 0
        total_correct_ant = 0
        total_seg_idx = 0
        total_ant_idx = 0
        for vid in vid_list:
            with open("/home/seulgi/work/darai-anticipation/FUTR_proposed/save_dir/50salads/visualization/files_l2supervision203050/gt_pred_log_{}_{}.txt".format(log_idx, obs_p), "w") as log:
                log.write("gt file\tGround Truth (GT)\tPrediction (Pred)\n")

                file_name = vid.split('/')[-1].split('.')[0]

                # load ground truth actions
                gt_file = os.path.join(gt_path, file_name+'.txt')
                gt_read = open(gt_file, 'r')
                gt_seq = gt_read.read().split('\n')[:-1]
                gt_read.close()

                # load features
                features_file = os.path.join(features_path, file_name+'.npy')
                features = np.load(features_file).transpose()

                vid_len = len(gt_seq)
                past_len = int(obs_p*vid_len)
                future_len = int(pred_p*vid_len)

                
                features = features[:past_len]
                inputs = features[::sample_rate, :]
                inputs = torch.Tensor(inputs).to(device)

                past_seq_before_sampling = gt_seq[:past_len]
                past_seq = past_seq_before_sampling[::sample_rate]
                past_seq = change_query_dict_2_action_dict(past_seq, action_mapping)
                #past_seq = seq2idx(past_seq, actions_dict)
                #past_seq = torch.Tensor(past_seq).to(device)

                future_content = gt_seq[past_len: past_len + future_len]
                future_content = future_content[::sample_rate]
                future_content = change_query_dict_2_action_dict(future_content, action_mapping)
                
                #future_content = seq2idx(future_content, actions_dict)
                #future_content = torch.Tensor(future_content).to(device)
                
                query = gt_seq[:past_len]
                query = query[::sample_rate]
                query_label = seq2idx4query(query)
                query_label = torch.Tensor(query_label).to(device)

                query_label = query_label.unsqueeze(0)
                # query_size = query_label.shape[1]
                # for batch in range(1):
                #     prev_label = query_label[batch][0]
                #     zero_or_one = 0
                #     for idx_query_size in range(query_size):
                #         if prev_label == query_label[batch][idx_query_size]:
                #             prev_label = query_label[batch][idx_query_size].detach().clone()
                #             query_label[batch][idx_query_size] = zero_or_one
                #         else:
                #             if zero_or_one == 0:
                #                 zero_or_one = 1
                #             else:
                #                 zero_or_one = 0
                #             prev_label = query_label[batch][idx_query_size].detach().clone()
                #             query_label[batch][idx_query_size] = zero_or_one

                outputs = model(inputs=inputs.unsqueeze(0), query=query_label, mode='test', epoch=log_idx, idx=obs_p)
                #outputs = model(inputs=inputs.unsqueeze(0), mode='test')
                log.write(f"{gt_file}\n")
                log.write("------------------\n")
                log.write(f"past seq: {len(past_seq)}\n")
                log.write(f"query:  {query}\n")
                log.write("---------------------")
                log.write(f"query_num: {query_label}\n")
                
                if False:
                    output_segmentation = outputs['seg']
                    B, T, C = output_segmentation.size()
                    output_segmentation = output_segmentation.view(-1, C).to(device)
                    output_segmentation_label = output_segmentation.max(-1)[1]
                    correct_seg, seg_idx = normal_accuracy_without_gif(output_segmentation_label, past_seq, actions_dict)
                    #_, correct_seg, seg_idx, _ = cal_performance(output_segmentation_label, past_seq, len(actions_dict) + 2)

                    total_correct_seg += correct_seg
                    total_seg_idx += seg_idx

                output_action = outputs['action']
                B, T, C = output_action.size()

                output_label_size = output_action.max(-1)[1].size(1)
                output_action = output_action.view(-1, C).to(device)
                output_label = output_action.max(-1)[1]
                
                correct_ant, ant_idx = normal_accuracy_without_gif(output_label, future_content, actions_dict, exclude_class_idx=5, log=log)
                #_, correct_ant, ant_idx, _ = cal_performance(output_label, future_content[:8], len(actions_dict) + 2)

                total_correct_ant += correct_ant
                total_ant_idx += ant_idx

                output_dur = outputs['duration']
                log_idx += 1

                # fine the forst none class
                none_mask = None
                none_idx = None
                for i in range(output_label_size) :
                    if output_label[i] == NONE :
                        none_idx = i
                        break
                    else :
                        none = None
                if none_idx is not None :
                    none_mask = torch.ones(output_label.shape).type(torch.bool)
                    #none_mask[0, none_idx:] = False
                    none_mask[none_idx:] = False
                    # Apply normalize_duration only if none_mask is available
                    output_dur = normalize_duration(output_dur, none_mask.to(device))
                else:
                    # If none_idx is None, use default normalized duration
                    output_dur = normalize_duration(output_dur, torch.ones_like(output_dur).to(device))

                #output_dur = normalize_duration(output_dur, none_mask.to(device))
                #print(output_dur)
                pred_len = (0.5+future_len*output_dur).squeeze(-1).long()
                #print(pred_len)

                pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
                predicted = torch.ones(future_len)
                #print(output_label.shape)
                action = output_label.squeeze()
                #print(output_label.shape)

                for i in range(len(action)) :
                    predicted[int(pred_len[i]) : int(pred_len[i] + pred_len[i+1])] = action[i]
                    pred_len[i+1] = pred_len[i] + pred_len[i+1]
                    if i == len(action) - 1 :
                        predicted[int(pred_len[i]):] = action[i]


                prediction = past_seq_before_sampling
                #print(len(predicted))
                for i in range(len(predicted)):
                    #print(prediction)
                    prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]]))

                #evaluation
                for i in range(len(eval_p)):
                    p = eval_p[i]
                    eval_len = int((obs_p+p)*vid_len)
                    eval_prediction = prediction[:eval_len]
                    T_action, F_action = eval_file(change_query_dict_2_action_dict(gt_seq, action_mapping), eval_prediction, obs_p, actions_dict)
                    T_actions[i] += T_action
                    F_actions[i] += F_action

        results = []
        for i in range(len(eval_p)):
            acc = 0
            n = 0
            for j in range(len(actions_dict)):
                total_actions = T_actions + F_actions
                if total_actions[i,j] != 0:
                    acc += float(T_actions[i,j]/total_actions[i,j])
                    n+=1

            result = 'obs. %d '%int(100*obs_p) + 'pred. %d '%int(100*eval_p[i])+'--> MoC: %.4f'%(float(acc)/n)
            results.append(result)
            print(result)
            
        print('--------------------------------')
        print(100 * obs_p, '%')
        print("!!!!!!!!!!!!! ant Acc: ", total_correct_ant/total_ant_idx)
        #print("@!@!@!@!@!@!@ seg Acc: ", total_correct_seg/total_seg_idx)

        return






