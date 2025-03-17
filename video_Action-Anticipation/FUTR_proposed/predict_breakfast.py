import torch
import torch.nn as nn
import numpy
import pdb
import os
import copy
from collections import defaultdict
import numpy as np
from utils import normalize_duration, eval_file, cal_performance, cal_performance_focal

def get_last_non_padding_labels(past_label, pad_value):
        """
        For each sequence in `past_label`, find the last non-padding label.
        
        Args:
            past_label (torch.Tensor): Tensor of shape [batch_size, frame_count] with padding.
            pad_value (int): Value used for padding.

        Returns:
            torch.Tensor: Tensor of shape [batch_size] containing the last non-padding label for each sequence.
        """
        batch_size = past_label.size(0)
        t_n_last_labels = torch.zeros(batch_size, dtype=past_label.dtype, device=past_label.device)
        
        for i in range(batch_size):
            # Get the sequence for the batch item, reverse it, and find the first non-padding value
            non_pad_indices = (past_label[i] != pad_value).nonzero(as_tuple=True)[0]
            if non_pad_indices.numel() > 0:
                t_n_last_labels[i] = past_label[i, non_pad_indices[-1]]
            else:
                t_n_last_labels[i] = pad_value  # Fallback if entire sequence is padding

        return t_n_last_labels


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
        gt = gold[i]#actions_dict[gold[i].replace(' ', '')]
        # if exclude_class_idx is not None and pred[i].item() == exclude_class_idx:
        #     if log is not None:
        #         log.write(f"\tgt: {gold[i]}\t, prediction: {pred[i].item()}\n")

        #     continue
        
        idx += 1

        if pred[i].item() == gt:
            total_correct += 1
        if log is not None:
            log.write(f"\tgt: {gold[i]}\t, prediction: {pred[i].item()}\n")

    if length == 0 or idx == 0:
        return 0
    accuracy = total_correct / idx
    #print("accuracy: ", accuracy)
    # if accuracy == 0.0:
    #     print("gt: ", gold[0])
    #     print("pred: ", pred[0].item())
    # print("-----------------------------")
    return accuracy

def seq2idx(seq, actions_dict):
    idx = np.zeros(len(seq))
    for i in range(len(seq)):
        idx[i] = actions_dict[seq[i]]
    return idx

def weighted_accuracy(pred, gold, pad_idx, t_n_labels, weight_same=1.0, weight_different=10.0):
    '''Calculate weighted accuracy based on comparison between t+n and t+m labels.'''
    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(pad_idx)

    total_weighted_correct = 0
    total_weighted_labels = 0

    weight = weight_different if gold[0] != t_n_labels[0] else weight_same

    for i in range(len(gold)):
        if non_pad_mask[i].any():
            if pred[i] == gold[i]:
                total_weighted_correct += weight
            total_weighted_labels += weight

    weighted_accuracy = total_weighted_correct / total_weighted_labels if total_weighted_labels > 0 else 0
    return weighted_accuracy

def predict(model, test_loader, pad_idx, device):
    model.eval()
    
    val_class_correct = 0
    val_class_total = 0
    val_seg_correct = 0
    val_seg_total = 0
    total_l3 = 0
    total_l3_correct = 0
    val_weighted_accuracy_total = 0
    with torch.no_grad():
        for data in test_loader:
            features, past_label, trans_dur_future, trans_future_target, query_label = data
            features = features.to(device)
            past_label = past_label.to(device)
            query_label = query_label.to(device)
            trans_dur_future = trans_dur_future.to(device)
            trans_future_target = trans_future_target.to(device)
            
            if features.shape[1] > 2000:
                continue

            outputs = model((features, past_label), query_label)


            output_l3 = outputs['l3']
            B, T, C = output_l3.size()
            output_l3 = output_l3.view(-1, C).to(device)
            query_label = query_label.view(-1)
            _, n_l3_correct, n_l3_total, _ = cal_performance_focal(output_l3, query_label, 48, 48, reference=None, target_ref=None)
            
            total_l3 += n_l3_total
            total_l3_correct += n_l3_correct

            output_seg = outputs['seg']
            _, n_seg_correct, n_seg_total, _ = cal_performance(output_seg.view(-1, output_seg.size(-1)),
                                                                    past_label.view(-1), pad_idx, exclude_class_idx=None, reference=None, target_ref=None)
            
            val_seg_correct += n_seg_correct
            val_seg_total += n_seg_total

            output = outputs['action']
            _, n_class_correct, n_class_total, _ = cal_performance(output.view(-1, output.size(-1)),
                                                                            trans_future_target.view(-1), pad_idx, exclude_class_idx=None, reference=get_last_non_padding_labels(past_label, pad_idx), target_ref=trans_future_target[:,0])
            
            val_class_correct += n_class_correct
            val_class_total += n_class_total

            # Calculate weighted accuracy using specific t+n and t+m labels
            val_weighted_acc = weighted_accuracy(
                output.view(-1, output.size(-1)), trans_future_target.view(-1), pad_idx, get_last_non_padding_labels(past_label, pad_idx)
            )
            val_weighted_accuracy_total += val_weighted_acc


    l3_accuracy = total_l3_correct / total_l3 if total_l3 else 0
    val_accuracy = val_class_correct / val_class_total if val_class_total else 0
    val_seg_accuracy = val_seg_correct / val_seg_total if val_seg_total else 0
    val_weighted_accuracy = val_weighted_accuracy_total / len(test_loader)
    print(f"l3 accuracy: {l3_accuracy}, Class Accuracy: {val_accuracy:.3f}, Segmentation Accuracy: {val_seg_accuracy:.3f}, Weighted Accuracy: {val_weighted_accuracy:.3f}")
    return val_accuracy, val_weighted_accuracy
 

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
        for vid in vid_list:
            with open("gt_pred_log_{}_{}.txt".format(log_idx, obs_p), "w") as log:
                log.write("gt file\tGround Truth (GT)\tPrediction (Pred)\n")

                file_name = vid.split('/')[-1].split('.')[0]

                # load ground truth actions
                gt_file = os.path.join(gt_path, file_name+'.txt')
                gt_read = open(gt_file, 'r')
                query = gt_read.read().split('\n')[:-1]
                gt_read.close()

                ## custom ##
                L2_label = (os.path.basename(gt_file)).split('_')[-1].split('.')[0]
                all_content = [L2_label] * len(query)
                vid_len = len(all_content)
                ##########

                # load features
                features_file = os.path.join(features_path, file_name+'.npy')
                features = np.load(features_file).transpose()

                past_len = int(obs_p*vid_len)
                future_len = int(pred_p*vid_len)
                if(past_len/sample_rate > 2000):
                    #print(past_len/sample_rate)
                    continue
                
                features = features[:past_len]
                inputs = features[::sample_rate, :]
                inputs = torch.Tensor(inputs).to(device)

                past_seq_before_sampling = all_content[:past_len]
                past_seq = past_seq_before_sampling[::sample_rate]
                past_label = seq2idx(past_seq, actions_dict)

                future_content = all_content[past_len: past_len + future_len]
                future_content = future_content[::sample_rate]
                future_content = seq2idx(future_content, actions_dict)
                
                query = query[:past_len]
                query = query[::sample_rate]
                query_label = seq2idx4query(query)
                query_label = torch.Tensor(query_label).to(device)

                query_label = query_label.unsqueeze(0)
                query_size = query_label.shape[1]
                for batch in range(1):
                    prev_label = query_label[batch][0]
                    zero_or_one = 0
                    for idx_query_size in range(query_size):
                        if prev_label == query_label[batch][idx_query_size]:
                            prev_label = query_label[batch][idx_query_size].detach().clone()
                            query_label[batch][idx_query_size] = zero_or_one
                        else:
                            if zero_or_one == 0:
                                zero_or_one = 1
                            else:
                                zero_or_one = 0
                            prev_label = query_label[batch][idx_query_size].detach().clone()
                            query_label[batch][idx_query_size] = zero_or_one

                outputs = model(inputs=inputs.unsqueeze(0), query=query_label, mode='test')
                #outputs = model(inputs=inputs.unsqueeze(0), mode='test')
                log.write(f"{gt_file}\n")
                log.write("------------------\n")
                log.write(f"past seq: {len(past_seq)}\n")
                
                if False:
                    output_segmentation = outputs['seg']
                    B, T, C = output_segmentation.size()
                    output_segmentation = output_segmentation.view(-1, C).to(device)
                    output_segmentation_label = output_segmentation.max(-1)[1]
                    seg_acc += normal_accuracy_without_gif(output_segmentation_label, past_label, actions_dict)

                output_action = outputs['action']
                B, T, C = output_action.size()
                output_label_size = output_action.max(-1)[1].size(1)
                output_action = output_action.view(-1, C).to(device)
                output_label = output_action.max(-1)[1]
                ant_acc += normal_accuracy_without_gif(output_label, future_content, actions_dict, exclude_class_idx=10, log=log)

                idx += 1

                output_dur = outputs['duration']

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
                    none_mask[none_idx:] = False
                    # Apply normalize_duration only if none_mask is available
                    output_dur = normalize_duration(output_dur, none_mask.to(device))
                else:
                    # If none_idx is None, use default normalized duration
                    output_dur = normalize_duration(output_dur, torch.ones_like(output_dur).to(device))

                #output_dur = normalize_duration(output_dur, none_mask.to(device))

                pred_len = (0.5+future_len*output_dur).squeeze(-1).long()

                pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
                predicted = torch.ones(future_len)
                action = output_label.squeeze()

                for i in range(len(action)) :
                    predicted[int(pred_len[i]) : int(pred_len[i] + pred_len[i+1])] = action[i]
                    pred_len[i+1] = pred_len[i] + pred_len[i+1]
                    if i == len(action) - 1 :
                        predicted[int(pred_len[i]):] = action[i]


                prediction = past_seq_before_sampling#action.cpu().numpy()
                #prediction = past_seq
                for i in range(len(predicted)):
                    prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]]))

                #evaluation
                for i in range(len(eval_p)):
                    p = eval_p[i]
                    eval_len = int((obs_p+p)*vid_len)
                    eval_prediction = prediction[:eval_len]
                    T_action, F_action = eval_file(all_content, eval_prediction, obs_p, actions_dict)
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
        print("!!!!!!!!!!!!! ant Acc: ", ant_acc/idx)
        print("@!@!@!@!@!@!@ seg Acc: ", seg_acc/idx)

        return






