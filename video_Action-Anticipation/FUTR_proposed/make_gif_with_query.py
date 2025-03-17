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
import imageio

def get_action_by_number(number):
    actions = {
        0: "Cleaning_Countertops",
        1: "Cleaning_Floor",
        2: "Having_a_meal",
        3: "Making_pancake_with_recipe",
        4: "Making_pancake_without_recipe",
        5: "Mix_ingredients",
        6: "Prep_ingredients",
        7: "Prepare_Kitchen_appliance",
        8: "Setting_a_table",
        9: "Take_out_Kitchen_and_cooking_tools",
        10: "Take_out_smartphone",
        11: "Throw_out_leftovers",
        12: "Using_Smartphone",
        13: "Using_Tablet",
        14: "Washing_and_Drying_dishes_with_hands",
        15: "UNDEFINED"
    }
    return actions.get(number, "Invalid number")

def weighted_accuracy(pred, gold, t_n_labels, actions_dict, image_base=None, label_base=None, image_target=None, gif_name=None, duration=0.2, weight_same=1.0, weight_different=10.0):
    '''Calculate weighted accuracy based on comparison between t+n and t+m labels.'''
    pred = pred[0]#.max(1)[1]

    frames = []

    save_path = './save_dir/darai/visualization/'

    total_weighted_correct = 0
    total_weighted_labels = 0

    weight = weight_different if gold[0] != t_n_labels[0] else weight_same
    length = min(len(gold), len(pred))
    idx = 0
    for img_path in image_base:
        fig, ax = plt.subplots(figsize=(6, 6))
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        fig.text(0.5, 0.1, label_base[idx].replace(' ', ''), ha='center', fontsize=14, fontweight='bold')
        idx += 1
        
        # 이미지 버퍼에 저장하고 GIF 프레임에 추가
        fig.canvas.draw()
        frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(frame)
        plt.close(fig)

    for i in range(length):
        gt = actions_dict[gold[i].replace(' ', '')]

        if pred[i].item() == gt:
            total_weighted_correct += weight
            # the network is correct: save the images, pred, gt.
            label_color = 'blue'
        else:
            # the network is wrong: save the images, pred, gt.
            label_color = 'red'

        # 시각화 작업 시작
        fig, ax = plt.subplots(figsize=(6, 6))
        target_img = Image.open(image_target[i])
        ax.imshow(target_img)
        ax.axis('off')
        
        # # GT와 Pred 레이블 추가
        fig.text(
            0.5, 0.9,
            f"GT: {get_action_by_number(gt)} | Pred: {get_action_by_number(pred[i].item())}",
            color=label_color,
            ha='center', va='top',
            fontsize=12, fontweight='bold'
        )
        
        # 이미지 버퍼에 저장하고 GIF 프레임에 추가
        fig.canvas.draw()
        frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(frame)
        plt.close(fig)

        total_weighted_labels += weight

    gif_path = os.path.join(save_path, 'uemp'+gif_name+'_'+str(duration)+'.gif')
    imageio.mimsave(gif_path, frames, duration=5, loop=0)
    print(f"GIF saved at: {gif_path}")

    weighted_accuracy = total_weighted_correct / total_weighted_labels if total_weighted_labels > 0 else 0
    return weighted_accuracy

def weighted_accuracy_without_gif(log, pred, gold, t_n_labels, actions_dict, exclude_class_idx=None, image_base=None, label_base=None, image_target=None, gif_name=None, duration=0.2, weight_same=1.0, weight_different=10.0):
    '''Calculate weighted accuracy based on comparison between t+n and t+m labels.'''
    pred = pred[0]

    #print("len of pred: ", len(pred))
    assert len(pred) == 8

    total_weighted_correct = 0
    total_weighted_labels = 0

    weight = weight_different if gold[0] != t_n_labels else weight_same
    length = min(len(gold), len(pred))

    for i in range(length):
        gt = actions_dict[gold[i].replace(' ', '')]

        # Skip if the ground truth or prediction corresponds to exclude_class_idx
        if exclude_class_idx is not None and gt == exclude_class_idx:
            continue

        if pred[i].item() == gt:
            total_weighted_correct += weight

        total_weighted_labels += weight

        log.write(f"\t{gold[i].replace(' ', '')}\t{pred[i].item()}\t{weight}\n")

    weighted_accuracy = total_weighted_correct / total_weighted_labels if total_weighted_labels > 0 else 0
    return weighted_accuracy


def normal_accuracy_without_gif(log, pred, gold, actions_dict, exclude_class_idx=None, image_base=None, label_base=None, image_target=None, gif_name=None, duration=0.2, weight_same=1.0, weight_different=10.0):
    '''Calculate weighted accuracy based on comparison between t+n and t+m labels.'''
    total_correct = 0
    assert len(gold) == len(pred)
    length = len(gold)
    
    for i in range(length):
        gt = actions_dict[gold[i].replace(' ', '')]

        if exclude_class_idx is not None and gt == exclude_class_idx:
            continue

        if pred[i].item() == gt:
            total_correct += 1

        log.write(f"\t{gold[i].replace(' ', '')}\t{pred[i].item()}\n")

    accuracy = total_correct / length
    
    return accuracy


def predict(model, vid_list, args, obs_p, n_class, actions_dict, device, query_dict):
    def seq2idx4query(seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = query_dict[seq[i].replace(' ', '')]
        return idx

    acc = 0
    seg_acc = 0
    l3_acc = 0
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

        log_idx = 0

        for vid in vid_list:
            base_name = vid.split('/')[-1].split('.')[0]
            seq_idx = 1

            while True:
                with open("gt_pred_log_{}_{}.txt".format(log_idx, obs_p), "w") as log:
                    log.write("gt file\tGround Truth (GT)\tPrediction (Pred)\n")
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
                    label_base = past_seq[::sample_rate]

                    ## for visualize: images that needs to be anticipated.
                    image_target = image_path[past_len: past_len + future_len]
                    image_target = image_target[::sample_rate]

                    # Model inference
                    outputs = model(inputs=inputs.unsqueeze(0), query=query_label.unsqueeze(0), mode='test')

                    ## Action Segmentation
                    output_segmentation = outputs['seg']
                    B, T, C = output_segmentation.size()
                    output_segmentation = output_segmentation.view(-1, C).to(device)
                    output_segmentation_label = output_segmentation.max(-1)[1]
                    #target_past_label = past_label.view(-1)
                    log.write(f"{gt_file}\n")
                    log.write(f"---- segmentation -----\n")
                    seg_acc += normal_accuracy_without_gif(log, output_segmentation_label, label_base, actions_dict, 16)

                    #print("input length: ", len(inputs), ", output length: ", len(outputs['action']))
                    output_action = outputs['action']
                    output_dur = outputs['duration']
                    output_label = output_action.max(-1)[1]
                    
                    
                    log.write(f"---- anticipation -----\n")
                    acc += weighted_accuracy_without_gif(log, output_label, future_content, past_seq[-1], actions_dict, 16, image_base, label_base, image_target, f"{base_name}_{seq_idx}", obs_p)
                    
                    output_l3 = outputs['l3']
                    B, T, C = output_l3.size()
                    output_l3 = output_l3.view(-1, C).to(device)
                    #print(output_l3[18], output_l3[19])
                    output_l3_label = output_l3.max(-1)[1]
                    log.write(f"---- L3 -----\n")
                    l3_acc += normal_accuracy_without_gif(log, output_l3_label, query, query_dict, 47)
                    
                    log_idx += 1
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

                    seq_idx += 1  # Move to the next sequence file for the current video

        print("!!!!!!!!!!!!! ant Acc: ", acc/idx)
        print("@!@!@!@!@!@!@ seg Acc: ", seg_acc/idx)
        print("///////////// l3 Acc: ", l3_acc/idx)
        # Calculate and print results
        results = []
        for i in range(len(eval_p)):
            acc = 0
            n = 0
            for j in range(len(actions_dict)):
                total_actions = T_actions + F_actions
                if total_actions[i, j] != 0:
                    acc += float(T_actions[i, j] / total_actions[i, j])
                    n += 1

            result = f'obs. {int(100 * obs_p)}% pred. {int(100 * eval_p[i])}% --> MoC: {float(acc) / n:.4f}'
            results.append(result)
            print(result)
        print('--------------------------------')

        return




