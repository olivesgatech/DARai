import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import xml.etree.ElementTree as ET
import re

# actions_dict와 query_dict를 로드하는 함수
def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            idx, label = line.strip().split()
            mapping[label] = int(idx)
    return mapping

class BaseDataset(Dataset):
    def __init__(self, train_split_path, actions_dict, pad_idx, n_class, args=None):
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.pad_idx = pad_idx
        
        self.args = args

        # Load specific files listed in train_split.txt
        with open(train_split_path, 'r') as f:
            self.vid_list = [os.path.join(args.groundtruth_dir, line.strip()) for line in f]

    def preprocess_depth(self, depth_path, save_npy=False, npy_path=None):
        # Load depth map directly
        depth_path = depth_path.replace('extracted_frames', 'nturgb+d_depth_masked')
        filename = os.path.basename(depth_path)
        if filename.startswith("frame_") and filename.endswith(".png"):
            frame_number = int(filename[6:10]) + 1  # Increment by 1
            depth_filename = f"MDepth-{frame_number:08d}.png"
            depth_path = os.path.join(os.path.dirname(depth_path), depth_filename)
        # Load depth map using cv2 (grayscale mode)
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # Ensure the data shape is (512, 424)
        if depth_data.shape != (424, 512):
            raise ValueError(f"Unexpected depth map shape: {depth_data.shape}, expected (512, 424)")
        
        # Normalize depth values to range [0,1]
        depth_min = np.min(depth_data)
        depth_max = np.max(depth_data)
        if depth_max > depth_min:
            depth_data = (depth_data - depth_min) / (depth_max - depth_min)
        else:
            depth_data = np.zeros_like(depth_data, dtype=np.float32)  # Handle uniform depth case
        
        # Resize to (224, 224)
        depth_data = cv2.resize(depth_data, (224, 224))
        
        # Save as .npy file if needed
        if save_npy and npy_path:
            np.save(npy_path, depth_data)
        
        # Convert to PyTorch tensor (1 channel → 1, 224, 224)
        depth_tensor = torch.tensor(depth_data, dtype=torch.float32).unsqueeze(0)
        
        return depth_tensor

    def save_preprocessed_sequences(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 비디오 파일마다 처리
        for gt_file in self.vid_list:
            base_filename = os.path.basename(gt_file).replace('.txt', '')
            existing_files = [f for f in os.listdir(save_dir) if f.startswith(base_filename)]
            
            # 이미 처리된 파일이 있으면 건너뜀
            if existing_files:
                print(f"Skipping {gt_file} (already processed)")
                continue
            with open(gt_file, 'r') as file_ptr:
                lines = file_ptr.readlines()
                # 유효한 라인만 포함하도록 필터링
                valid_lines = [line.strip() for line in lines]

            features = []
            l2_labels = []
            l3_labels = []
            depth_paths = []

            # 유효한 모든 라인에 대해 처리
            for line in valid_lines:
                split_data = line.split(',')
                depth_path, l2_label, l3_label = split_data

                # 이미지 전처리
                depth_tensor = self.preprocess_depth(depth_path)
                if depth_tensor is not None:
                    features.append(depth_tensor.numpy())  # numpy array로 변환
                    l2_labels.append(l2_label)
                    l3_labels.append(l3_label)
                    depth_paths.append(depth_path)  # 이미지 경로 저장
                else:
                    print(f"Warning: {depth_path} cannot load the image. Saving...")
                    if features:
                        self._save_sequence(save_dir, gt_file, features, l2_labels, l3_labels, depth_paths)
                    # 새로운 시퀀스 시작
                    features = []
                    l2_labels = []
                    l3_labels = []
                    depth_paths = []

            # 루프가 종료된 후 남아있는 시퀀스를 저장
            if features:  # features가 비어있지 않은 경우에만 저장
                self._save_sequence(save_dir, gt_file, features, l2_labels, l3_labels, depth_paths)

    def _save_sequence(self, save_dir, gt_file, features, l2_labels, l3_labels, depth_paths):
        base_filename = os.path.basename(gt_file).replace('.txt', '')
        npy_file_name = f"{base_filename}.npy"
        npy_file_path = os.path.join(save_dir, npy_file_name)
        np.save(npy_file_path, np.vstack(features))  # [sequence 개수, 1, 224, 224]

        txt_file_name = f"{base_filename}.txt"
        txt_file_path = os.path.join(save_dir, txt_file_name)

        with open(txt_file_path, 'w') as f:
            for idx in range(len(l2_labels)):
                f.write(f"{depth_paths[idx]},{l2_labels[idx]},{l3_labels[idx]}\n")

        print(f"Saved: {txt_file_path} and {npy_file_path}")


def main():
    # 경로 설정
    train_split_path = '/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/splits/test_split.txt'
    l2_mapping_file = '/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/mapping_l2_changed.txt'
    
    save_dir = '/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/features_depth'  # 시퀀스를 저장할 디렉터리 경로
    os.makedirs(save_dir, exist_ok=True)

    # actions_dict와 query_dict 로드
    actions_dict = load_mapping(l2_mapping_file)

    pad_idx = 0
    n_class = len(actions_dict)
    
    class Args:
        groundtruth_dir = '/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/groundTruth'

    args = Args()

    # BaseDataset 생성 및 시퀀스 저장
    dataset = BaseDataset(train_split_path, actions_dict, pad_idx, n_class, args=args)
    dataset.save_preprocessed_sequences(save_dir)


if __name__ == "__main__":
    main()
