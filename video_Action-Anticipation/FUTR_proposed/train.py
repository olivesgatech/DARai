import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import numpy as np
from utils import cal_performance, normalize_duration

def validate(model, val_loader, pad_idx, device, model_save_path, epoch, best_accuracy):
    model.eval()
    total_class = 0
    total_class_correct = 0
    epoch_loss_class = 0
    epoch_loss = 0
    total_seg = 0
    total_seg_correct = 0
    epoch_loss_seg = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            features, past_label, trans_dur_future, trans_future_target, _ = data
            features = features.to(device)
            past_label = past_label.to(device)
            trans_future_target = trans_future_target.to(device)
            target = trans_future_target

            output = model((features, past_label))
            losses = 0

            ## segmentation 
            output_seg = output['seg']
            B, T, C = output_seg.size()
            output_seg = output_seg.view(-1, C).to(device)
            target_past_label = past_label.view(-1)
            loss_seg, n_seg_correct, n_seg_total, _ = cal_performance(output_seg, target_past_label, pad_idx)
            
            losses += loss_seg
            total_seg += n_seg_total
            total_seg_correct += n_seg_correct
            epoch_loss_seg += loss_seg.item()

            ## anticipation
            output = output['action']
            B, T, C = output.size()
            output = output.view(-1, C).to(device)
            target = target.contiguous().view(-1)
            out = output.max(1)[1] #oneshot
            out = out.view(B, -1)
            loss, n_correct, n_total, _ = cal_performance(output, target, pad_idx)
            
            losses += loss
            total_class += n_total
            total_class_correct += n_correct
            epoch_loss_class += loss.item()

        epoch_loss = epoch_loss / (i+1)
        accuracy = total_class_correct/total_class
        accuracy_seg = total_seg_correct / total_seg
        epoch_loss_class = epoch_loss_class / (i+1)
        epoch_loss_seg = epoch_loss_seg / (i+1)
        print('Validation Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class, 'segmentation Acc :%.3f'%accuracy_seg, 'seg loss :%.3f'%epoch_loss_seg )

        save_path = os.path.join(model_save_path)
        if accuracy > best_accuracy :
            best_accuracy = accuracy
            save_file = os.path.join(save_path, 'checkpoint'+str(epoch)+'.ckpt')
            torch.save(model.state_dict(), save_file)
            print("saved --- epoch ", epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    return best_accuracy


def train(args, model, train_loader, optimizer, scheduler, criterion,  model_save_path, pad_idx, device, val_loader):
    model.to(device)
    model.train()
    best_accuracy = 0
    print("Training Start")
    for epoch in range(args.epochs):
        epoch_acc =0
        epoch_loss = 0
        epoch_loss_class = 0
        epoch_loss_dur = 0
        epoch_loss_seg = 0
        total_class = 0
        total_class_correct = 0
        total_seg = 0
        total_seg_correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            features, past_label, trans_dur_future, trans_future_target, _ = data
            features = features.to(device) #[B, S, C]
            past_label = past_label.to(device) #[B, S]
            trans_dur_future = trans_dur_future.to(device)
            trans_future_target = trans_future_target.to(device)
            trans_dur_future_mask = (trans_dur_future != pad_idx).long().to(device)

            B = trans_dur_future.size(0)
            target_dur = trans_dur_future*trans_dur_future_mask
            target = trans_future_target
            if args.input_type == 'i3d_transcript':
                inputs = (features, past_label)
            elif args.input_type == 'gt':
                gt_features = past_label.int()
                inputs = (gt_features, past_label)

            #print(inputs[0].shape, inputs[1].shape)
            outputs = model(inputs)
            #print(outputs['seg'].shape, outputs['action'].shape) # (16, S, C), (16, 8, C) 
            #print("--------------------------------------")
            losses = 0
            if args.seg :
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                output_seg = output_seg.view(-1, C).to(device)
                target_past_label = past_label.view(-1)
                loss_seg, n_seg_correct, n_seg_total, _ = cal_performance(output_seg, target_past_label, pad_idx)
                losses += loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                epoch_loss_seg += loss_seg.item()
            if args.anticipate :
                output = outputs['action']
                B, T, C = output.size()
                output = output.view(-1, C).to(device)
                target = target.contiguous().view(-1)
                out = output.max(1)[1] #oneshot
                out = out.view(B, -1)
                loss, n_correct, n_total, _ = cal_performance(output, target, pad_idx)
                acc = n_correct / n_total
                loss_class = loss.item()
                losses += loss
                total_class += n_total
                total_class_correct += n_correct
                epoch_loss_class += loss_class

                output_dur = outputs['duration']
                output_dur = normalize_duration(output_dur, trans_dur_future_mask)
                target_dur = target_dur * trans_dur_future_mask
                loss_dur = torch.sum(criterion(output_dur, target_dur)) / \
                torch.sum(trans_dur_future_mask)

                losses += loss_dur
                epoch_loss_dur += loss_dur.item()


            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()


        epoch_loss = epoch_loss / (i+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss)
        if args.anticipate :
            accuracy = total_class_correct/total_class
            epoch_loss_class = epoch_loss_class / (i+1)
            print('Training Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )
            if args.task == 'long' :
                epoch_loss_dur = epoch_loss_dur / (i+1)
                print('dur loss: %.5f'%epoch_loss_dur)

        if args.seg :
            acc_seg = total_seg_correct / total_seg
            epoch_loss_seg = epoch_loss_seg / (i+1)
            print('seg loss :%.3f'%epoch_loss_seg, ', seg acc : %.5f'%acc_seg)

        scheduler.step()

        best_accuracy = validate(model, val_loader, pad_idx, device, model_save_path, epoch, best_accuracy)

    return model