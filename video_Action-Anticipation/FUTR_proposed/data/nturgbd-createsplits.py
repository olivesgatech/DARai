import os
import random

def split_dataset(groundtruth_dir, splits_dir, train_ratio=0.6, valid_ratio=0.25, test_ratio=0.15, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all files in groundTruth directory
    files = sorted(os.listdir(groundtruth_dir))
    
    # Shuffle files to ensure randomness
    random.shuffle(files)
    
    # Compute split indices
    total_files = len(files)
    train_end = int(total_files * train_ratio)
    valid_end = train_end + int(total_files * valid_ratio)
    
    # Split files
    train_files = files[:train_end]
    valid_files = files[train_end:valid_end]
    test_files = files[valid_end:]
    
    # Ensure splits directory exists
    os.makedirs(splits_dir, exist_ok=True)
    
    # Function to write file names to a txt file
    def write_split(file_list, filename):
        with open(os.path.join(splits_dir, filename), 'w') as f:
            for file in file_list:
                f.write(file + '\n')
    
    # Write files to respective splits
    write_split(train_files, "train.txt")
    write_split(valid_files, "valid.txt")
    write_split(test_files, "test.txt")
    
    print("Splitting complete! Files saved in:", splits_dir)

# Define paths
groundtruth_dir = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/groundTruth"
splits_dir = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/splits"

# Run the function
split_dataset(groundtruth_dir, splits_dir)