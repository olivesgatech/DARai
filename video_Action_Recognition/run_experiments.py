import argparse
from train_test import train_and_evaluate
import torch
import numpy as np
from utils import save_results_to_json
import matplotlib.pyplot as plt

import os
# Set PyTorch CUDA memory configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a 3D ResNet model.')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the training and testing data')
    parser.add_argument('--base_dir', type=str, default= os.path.dirname(os.path.abspath(__file__)),required=True, help='Directory base for results and figures directories')
    parser.add_argument("--device" , type = str , required=False)
    parser.add_argument("--batch_size" , type = int , default = 16, help = "batch size for the model" )
    parser.add_argument("--sampling" , type = str , default = "multi-uniform" , help = "sequence sampling method from each input sample")
    parser.add_argument('--backbone' , type = str , default= 'r3d' , help = "backbone of the model")
    parser.add_argument("--env" , type = str , default= "both" , help = "set of classes to train and test on")
    parser.add_argument("--load_model" , type = str , help = "load a checkpoint" )
    parser.add_argument("--cam_view" , type = str , default="cam_1")


    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--weights" , type = str2bool , default= False , help="Enable default weights (True or False)")
    # parser.add_argument("--env", nargs='+', default=['a', 'b', 'c', 'd'])
    parser.add_argument("--modality" , type = str , default = "rgb")
    args = parser.parse_args()
    seeds = [ 1 , 42 , 1000 ]

    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    all_test_losses = []
    all_test_accuracies = []

    for s in seeds:
        args.seed = s
        torch.cuda.empty_cache()
        seed_result = train_and_evaluate(args)

        all_train_losses.append(seed_result['train_loss'])
        all_val_losses.append(seed_result['val_loss'])
        all_train_accuracies.append(seed_result['train_acc'])
        all_val_accuracies.append(seed_result['val_acc'])
        all_test_losses.append(seed_result['test_loss'])
        all_test_accuracies.append(seed_result['test_acc'])

    max_epochs = max(len(loss) for loss in all_train_losses)
    def pad_sequence(seq, target_length):
        # Pad the sequence with np.nan until it reaches the target_length
        return seq + [np.nan] * (target_length - len(seq))
    # Convert lists to numpy arrays for easier computation of mean and std
    # all_train_losses = np.array(all_train_losses)
    # all_val_losses = np.array(all_val_losses)
    # all_train_accuracies = np.array(all_train_accuracies)
    # all_val_accuracies = np.array(all_val_accuracies)
    # all_test_losses = np.array(all_test_losses)
    # all_test_accuracies = np.array(all_test_accuracies)
    all_train_losses = np.array([pad_sequence(loss, max_epochs) for loss in all_train_losses])
    all_val_losses = np.array([pad_sequence(loss, max_epochs) for loss in all_val_losses])
    all_train_accuracies = np.array([pad_sequence(acc, max_epochs) for acc in all_train_accuracies])
    all_val_accuracies = np.array([pad_sequence(acc, max_epochs) for acc in all_val_accuracies])
    all_test_losses = np.array(all_test_losses)
    all_test_accuracies = np.array(all_test_accuracies)

    # Compute mean and standard deviation
    # train_loss_mean = all_train_losses.mean(axis=0)
    # train_loss_std = all_train_losses.std(axis=0)
    # val_loss_mean = all_val_losses.mean(axis=0)
    # val_loss_std = all_val_losses.std(axis=0)
    # train_acc_mean = all_train_accuracies.mean(axis=0)
    # train_acc_std = all_train_accuracies.std(axis=0)
    # val_acc_mean = all_val_accuracies.mean(axis=0)
    # val_acc_std = all_val_accuracies.std(axis=0)
    train_loss_mean = np.nanmean(all_train_losses, axis=0)
    train_loss_std = np.nanstd(all_train_losses, axis=0)
    val_loss_mean = np.nanmean(all_val_losses, axis=0)
    val_loss_std = np.nanstd(all_val_losses, axis=0)
    train_acc_mean = np.nanmean(all_train_accuracies, axis=0)
    train_acc_std = np.nanstd(all_train_accuracies, axis=0)
    val_acc_mean = np.nanmean(all_val_accuracies, axis=0)
    val_acc_std = np.nanstd(all_val_accuracies, axis=0)

    # Print test results
    print(f"Test Loss: {np.mean(all_test_losses)} ± {np.std(all_test_losses)}")
    print(f"Test Accuracy: {np.mean(all_test_accuracies)} ± {np.std(all_test_accuracies)}")
    average_result = {"mean_test_accuracy": np.mean(all_test_accuracies),
                      'std_test_accuracy': np.std(all_test_accuracies)}
    save_results_to_json(average_result,
                         os.path.join(args.base_dir ,"results",
                                      f"average seed result_{args.modality}_{args.backbone}_{args.sampling}_ep {args.epochs}_B {args.batch_size}_{args.env}_{args.cam_view}_weights {args.weights}.json"))
    # Plotting
    # epochs = range(1, args.epochs + 1)
    epochs = range(1, max_epochs + 1)

    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_mean, 'b', label='Training loss')
    plt.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2, color='b')
    plt.plot(epochs, val_loss_mean, 'r', label='Validation loss')
    plt.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2, color='r')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_mean, 'b', label='Training accuracy')
    plt.fill_between(epochs, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, alpha=0.2, color='b')
    plt.plot(epochs, val_acc_mean, 'r', label='Validation accuracy')
    plt.fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, alpha=0.2, color='r')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.base_dir ,"figures",
                             f"averaged_training_val_loss_acc_{args.modality}_{args.backbone}_{args.sampling}_ep {args.epochs}_B {args.batch_size}_{args.env}_{args.cam_view}_{args.data_dir.split('/')[-1]}_weights {args.weights}.png"))
    plt.show()


# Example Usage:
# python run_experiments.py --epochs 1 --data_dir "/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/" --base_dir "/home/ghazal/Activity_Recognition_benchmarking/"

