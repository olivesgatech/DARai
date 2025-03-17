import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import random
import seaborn as sns
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
# Plotting some examples from the dataloader
def plot_sequences(dataset, fig_dir , num_sequences=1 ):
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    for i in range(num_sequences):
        frames, label , id = dataset[i]
        # frames = frames.permute(1, 0, 2, 3)  # Change to (C, T, H, W) for plotting
        frames = denormalize(frames, mean, std)
        fig, axes = plt.subplots(1, frames.size(1), figsize=(10, 6))
        for j, ax in enumerate(axes):
            frame = frames[:, j, :, :].permute(1, 2, 0).numpy()
            frame = np.clip(frame, 0, 1)  # Clip values to [0, 1]
            ax.imshow(frame)
        plt.suptitle(f'Sequence {i + 1} - Label: {label}')
        save_path = os.path.join(fig_dir, f'sequence_{i + 1}_label_{label}.png')
        plt.savefig(save_path)
        plt.show()


# Plot some examples from the training dataset

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cuda_availability():
    # Check CUDA and PyTorch availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"PyTorch version: {torch.__version__}")

    if cuda_available:
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Running on CPU")

def plot_training_validation_loss_accuracy(epochs, train_losses, val_losses, train_accuracies, val_accuracies, seed ,savepath ):
    # plt.figure(figsize=(14, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_losses, label='Training Loss')
    # plt.plot(epochs, val_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title(f'Training and Validation Loss (Seed {seed})')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_accuracies, label='Training Accuracy')
    # plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title(f'Training and Validation Accuracy (Seed {seed})')
    #
    # plt.tight_layout()
    # plt.savefig(savepath)
    # plt.show()
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title(f'Training and Validation Loss (Seed {seed})', fontsize=14, fontweight='bold')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.title(f'Training and Validation Accuracy (Seed {seed})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
def plot_confusion_matrix(confusion_matrix, classes , save_path):
    # plt.figure(figsize=(16, 12))
    # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Confusion Matrix')
    # plt.tight_layout()
    # plt.savefig(save_path)
    # plt.show()
    plt.figure(figsize=(16, 12))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
def save_results_to_json(results, path):
    import json
    with open(path, 'w') as f:
        json.dump(results, f)