import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import re
import torchvision.models as models

# ResNet-50을 기반으로 feature extractor 정의
resnet50 = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
feature_extractor.eval()

# actions_dict와 query_dict를 로드하는 함수
def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            idx, label = line.strip().split()
            mapping[label] = int(idx)
    return mapping

class BaseDataset(Dataset):
    def __init__(self, train_split_path, actions_dict, query_dict, pad_idx, n_class, args=None):
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.query_dict = query_dict
        self.pad_idx = pad_idx
        #self.groundtruth_dir = groundtruth_dir
        self.sample_rate = args.sample_rate
        
        self.args = args
        self.NONE = self.n_class - 1
        self.all_sequences = []

        # Load specific files listed in train_split.txt
        with open(train_split_path, 'r') as f:
            self.vid_list = [os.path.join(args.groundtruth_dir, line.strip()) for line in f]


        # 유효한 시퀀스 수집
        for gt_file in self.vid_list:
            with open(gt_file, 'r') as file_ptr:
                lines = file_ptr.readlines()
                valid_lines = [line.strip() for line in lines if len(line.strip().split(',')) == 3]
                for i in range(len(valid_lines)):
                    self.all_sequences.append((i, gt_file))

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        try:
            image = cv2.resize(image, (224, 224))
        except:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        image = image.unsqueeze(0)  # 배치 차원 추가
        with torch.no_grad():
            features = feature_extractor(image)
            features = features.view(features.size(0), -1)
        return features

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
                valid_lines = [
                    line.strip() for line in lines 
                    if len(line.strip().split(',')) == 3 and 
                    line.strip().split(',')[1] != '' and 
                    line.strip().split(',')[2] != '' and
                    line.strip().split(',')[1] != ' ' and 
                    line.strip().split(',')[2] != ' '
                ]

            features = []
            l2_labels = []
            l3_labels = []
            image_paths = []
            seq_idx = 1  # 시퀀스 번호
            last_frame_number = None  # 마지막 프레임 번호를 저장

            # 유효한 모든 라인에 대해 처리
            for line in valid_lines:
                split_data = line.split(',')
                image_path, l2_label, l3_label = split_data

                # 이미지 경로에서 프레임 번호 추출
                frame_number = int(re.search(r'(\d+)\.jpg$', image_path).group(1))

                # 프레임 번호가 연속되지 않으면 현재 시퀀스를 저장하고 새로운 시퀀스를 시작
                if last_frame_number is not None and frame_number != last_frame_number + 1:
                    # 현재까지 수집된 시퀀스를 저장
                    if features:  # features가 비어있지 않은 경우에만 저장
                        self._save_sequence(save_dir, gt_file, seq_idx, features, l2_labels, l3_labels, image_paths)
                        seq_idx += 1  # 시퀀스 번호 증가

                    # 새로운 시퀀스 시작
                    features = []
                    l2_labels = []
                    l3_labels = []
                    image_paths = []

                # 이미지 전처리
                image_tensor = self.preprocess_image(image_path)
                if image_tensor is not None:
                    features.append(image_tensor.numpy())  # numpy array로 변환
                    l2_labels.append(l2_label)
                    l3_labels.append(l3_label)
                    image_paths.append(image_path)  # 이미지 경로 저장
                    last_frame_number = frame_number  # 마지막 프레임 번호 업데이트
                else:
                    print(f"Warning: {image_path} cannot load the image.")

            # 루프가 종료된 후 남아있는 시퀀스를 저장
            if features:  # features가 비어있지 않은 경우에만 저장
                self._save_sequence(save_dir, gt_file, seq_idx, features, l2_labels, l3_labels, image_paths)

    def _save_sequence(self, save_dir, gt_file, seq_idx, features, l2_labels, l3_labels, image_paths):
        # .npy 파일로 features 저장
        base_filename = os.path.basename(gt_file).replace('.txt', '')
        npy_file_name = f"{base_filename}_{seq_idx}.npy"
        npy_file_path = os.path.join(save_dir, npy_file_name)
        np.save(npy_file_path, np.vstack(features))  # [sequence 개수, 2048]

        # 텍스트 파일에 라벨과 이미지 경로 저장
        txt_file_name = f"{base_filename}_{seq_idx}.txt"
        txt_file_path = os.path.join(save_dir, txt_file_name)

        with open(txt_file_path, 'w') as f:
            for idx in range(len(l2_labels)):
                f.write(f"{image_paths[idx]},{l2_labels[idx]},{l3_labels[idx]}\n")

        print(f"Saved: {txt_file_path} and {npy_file_path}")



def main():
    # 경로 설정
    groundtruth_dir = 'FUTR_proposed/datasets/darai/groundTruth_temp'  # GroundTruth 파일 경로
    train_split_path = 'FUTR_proposed/datasets/darai/splits/test_split.txt'
    l2_mapping_file = 'FUTR_proposed/datasets/darai/mapping_l2.txt'
    l3_mapping_file = 'FUTR_proposed/datasets/darai/mapping_l3.txt'
    save_dir = 'FUTR_proposed/datasets/darai/features_temp'  # 시퀀스를 저장할 디렉터리 경로

    # actions_dict와 query_dict 로드
    actions_dict = load_mapping(l2_mapping_file)
    query_dict = load_mapping(l3_mapping_file)

    pad_idx = 0
    n_class = len(actions_dict)
    
    class Args:
        sample_rate = 15
        groundtruth_dir = 'FUTR_proposed/datasets/darai/groundTruth_temp'

    args = Args()

    # BaseDataset 생성 및 시퀀스 저장
    dataset = BaseDataset(train_split_path, actions_dict, query_dict, pad_idx, n_class, args=args)
    dataset.save_preprocessed_sequences(save_dir)


if __name__ == "__main__":
    main()
