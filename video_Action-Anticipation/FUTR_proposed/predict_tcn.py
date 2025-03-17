import torch
import torch.nn as nn
import numpy
import pdb
import os
import copy
from collections import defaultdict
import numpy as np
from utils import normalize_duration, eval_file, cal_performance

def accuracy(pred, gold, actions_dict, image_base=None, label_base=None, image_target=None, gif_name=None, duration=0.2, weight_same=1.0, weight_different=10.0):
    '''Calculate weighted accuracy based on comparison between t+n and t+m labels.'''
    pred = pred[0]

    total_correct = 0
    
    #assert len(gold) == len(pred)
    length = min(len(gold), len(pred))
    
    for i in range(length):
        gt = actions_dict[gold[i].replace(' ', '')]
        # print("-----------------")
        # print(pred[i].item())
        # print(gt)
        # print("-----------------")
        if pred[i].item() == gt:
            total_correct += 1

    accuracy = total_correct / length
    return accuracy


def predict(model, vid_list, args, obs_p, n_class, actions_dict, device, pad_idx):
    model.eval()
    acc = 0
    idx = 0
    total_class = 0
    total_class_correct = 0
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

        for vid in vid_list:
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

            past_seq = gt_seq[:past_len]
            features = features[:past_len]
            inputs = features[::sample_rate, :]
            inputs = torch.Tensor(inputs).to(device)
            future_seq = gt_seq[past_len:]
            future_seq = future_seq[::sample_rate]
            future_seq = torch.Tensor(future_seq)

            output_action = model(x=inputs.unsqueeze(0))
            B, T, C = output_action.size()
            output_label = output_action.view(-1, C)

            loss, n_correct, n_total = cal_performance(output_label, future_seq, pad_idx)
            #output_label = output_action.max(-1)[1]
            total_class += n_total
            total_class_correct += n_correct

            acc += accuracy(output_label, future_seq, actions_dict)
            idx += 1

            predicted = torch.ones(future_len)
            action = output_label.squeeze()
            prediction = past_seq
            for i in range(len(predicted)):
                prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]]))

            #evaluation
            for i in range(len(eval_p)):
                p = eval_p[i]
                eval_len = int((obs_p+p)*vid_len)
                eval_prediction = prediction[:eval_len]
                T_action, F_action = eval_file(gt_seq, eval_prediction, obs_p, actions_dict)
                T_actions[i] += T_action
                F_actions[i] += F_action

        results = []
        print("----------- accuracy : ", acc/idx)
        print("------------- accuracy_util: ", total_class_correct/total_class)
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

        return






