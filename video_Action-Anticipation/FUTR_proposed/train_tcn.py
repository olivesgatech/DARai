import torch
import torch.nn.functional as F
import os
from utils import cal_performance

def validate(model, val_loader, pad_idx, device, model_save_path, epoch, best_accuracy):
    model.eval()
    total_class = 0
    total_class_correct = 0
    epoch_loss_class = 0
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            features, past_label, trans_dur_future, trans_future_target, _ = data
            features = features.to(device)
            past_label = past_label.to(device)
            trans_future_target = trans_future_target.to(device)
            target = trans_future_target

            output = model((features, past_label)[0])
            losses = 0
            B, T, C = output.size()
            output = output.view(-1, C).to(device)
            target = target.contiguous().view(-1)
            out = output.max(1)[1] #oneshot
            out = out.view(B, -1)
            loss, n_correct, n_total = cal_performance(output, target, pad_idx)
            
            loss_class = loss.item()
            losses += loss
            total_class += n_total
            total_class_correct += n_correct
            epoch_loss_class += loss_class

            epoch_loss += losses.item()


        epoch_loss = epoch_loss / (i+1)
        accuracy = total_class_correct/total_class
        epoch_loss_class = epoch_loss_class / (i+1)
        print('Validation Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )

        save_path = os.path.join(model_save_path)
        if accuracy > best_accuracy :
            best_accuracy = accuracy
            save_file = os.path.join(save_path, 'checkpoint'+str(epoch)+'.ckpt')
            torch.save(model.state_dict(), save_file)
            print("saved --- epoch ", epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    return best_accuracy


def train(args, model, train_loader, val_loader, optimizer, scheduler, criterion,  model_save_path, pad_idx, device):
    model.to(device)
    model.train()
    print("Training Start")
    best_accuracy = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_loss_class = 0
        total_class = 0
        total_class_correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            features, past_label, trans_dur_future, trans_future_target, _ = data
            features = features.to(device) #[B, S, C]
            past_label = past_label.to(device) #[B, S]
            trans_future_target = trans_future_target.to(device)

            target = trans_future_target
            inputs = (features, past_label)

            output = model(inputs[0])
            #output = output.unsqueeze(1)

            losses = 0

            B, T, C = output.size()
            output = output.view(-1, C).to(device)
            target = target.contiguous().view(-1)
            out = output.max(1)[1] #oneshot
            out = out.view(B, -1)
            loss, n_correct, n_total = cal_performance(output, target, pad_idx)
            acc = n_correct / n_total
            loss_class = loss.item()
            losses += loss
            total_class += n_total
            total_class_correct += n_correct
            epoch_loss_class += loss_class

            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()


        epoch_loss = epoch_loss / (i+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss)
        accuracy = total_class_correct/total_class
        epoch_loss_class = epoch_loss_class / (i+1)
        print('Training Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )

        scheduler.step()

        best_accuracy = validate(model, val_loader, pad_idx, device, model_save_path, epoch, best_accuracy)

    return model