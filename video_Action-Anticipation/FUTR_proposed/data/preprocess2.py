import os
import numpy as np
import torch
import cv2
from torchvision import models

# Load ResNet50 model and set up feature extractor
resnet50 = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
feature_extractor.eval()

def preprocess_image(image_path):
    """Preprocess an image and extract features using ResNet50."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image at {image_path}")
        return None
    try:
        image = cv2.resize(image, (224, 224))
    except Exception as e:
        print(f"Error resizing image at {image_path}: {e}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = feature_extractor(image)
        features = features.view(features.size(0), -1)
    return features.numpy()

def process_ground_truth_folder(groundtruth_folder, feature_folder):
    """Process all .txt files in a ground truth folder and save the first split of each row as .npy."""
    for txt_file in os.listdir(groundtruth_folder):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(groundtruth_folder, txt_file)
            base_name = os.path.splitext(txt_file)[0]
            npy_path = os.path.join(feature_folder, f"{base_name}.npy")
            
            features = []
            # Read the text file and process each line
            with open(txt_path, 'r') as file:
                lines = file.readlines()
                
                first_splits = [preprocess_image(line.strip().split(',')[0]) for line in lines if ',' in line]

            # Save the extracted values as a .npy file
            np.save(npy_path, np.vstack(first_splits))
            print(f"Processed {txt_file} and saved as {npy_path}")

# Specify the path to the ground truth folder
groundtruth_folder = "/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth_temp"  # Update with the correct path
feature_folder = "/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/features_update"
process_ground_truth_folder(groundtruth_folder, feature_folder)
