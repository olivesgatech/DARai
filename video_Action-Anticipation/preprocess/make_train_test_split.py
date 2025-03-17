# import os

# # 데이터 경로 설정
# data_path = '/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth'

# # 두 개의 리스트 생성
# same_l2_list = []
# different_l2_list = []

# # 필터링할 파일 이름 목록
target_filenames = [
    '_Dining.txt', 'Cleaning the kitchen.txt', 'Making pancake.txt', 
    'Cleaning dishes.txt', 'Using handheld smart devices.txt', 
    'Making a cup of instant coffee.txt', 'Making a cup of coffee in coffee maker.txt'
]
# all_file_list = []

# # txt 파일들 iterate
# for filename in os.listdir(data_path):
#     # 파일 이름이 target_filenames 중 하나로 끝나는지 확인
#     if filename.endswith(".txt") and any(filename.endswith(target) for target in target_filenames):
#         flag = True
#         file_path = os.path.join(data_path, filename)
        
#         # 각 파일 읽기
#         with open(file_path, 'r') as f:
#             lines = f.readlines()
            
#             # 각 행에 대해 split 길이가 3인지 확인하여 유효한 데이터만 필터링
#             valid_lines = [line.strip() for line in lines if len(line.strip().split(',')) == 3]
            
#             # 15 프레임마다 5개의 이미지를 뽑을 수 있는지 확인
#             for i in range(0, len(valid_lines) - 6 * 15, 15):
#                 if flag:
#                     all_file_list.append(filename)
#                     flag = False
#                 sequence_images = []
#                 sequence_l2_labels = []  # l2 라벨을 저장하기 위한 리스트
                
#                 # 5개의 이미지와 l2 label 추가
#                 for j in range(5):
#                     split_data = valid_lines[i + j * 15].split(',')
#                     image_path, l2_label, l3_label = split_data
#                     sequence_images.append(image_path)
#                     sequence_l2_labels.append(l2_label)  # l2 라벨 저장
                
#                 # 6초 시점의 l2 label 가져오기
#                 sixth_frame_data = valid_lines[i + 5 * 15].split(',')
#                 l2_label_5sec = sequence_l2_labels[-1]  # 5초 시점의 l2 label 가져오기
#                 l2_label_6sec = sixth_frame_data[1]     # 6초 시점의 l2 label 가져오기

#                 # 5초 시점의 l2 label과 6초 시점의 l2 label 비교하여 리스트에 추가
#                 if l2_label_5sec == l2_label_6sec:
#                     same_l2_list.append((sequence_images, sequence_l2_labels, l2_label_6sec))
#                 else:
#                     different_l2_list.append((sequence_images, sequence_l2_labels, l2_label_6sec))

# # 결과 출력
# print("Same L2 list:", len(same_l2_list))
# print("Different L2 list:", len(different_l2_list))
# print(all_file_list)

import os
import random
from collections import defaultdict

# 데이터 경로 설정
data_path = '/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth'

# 분할 비율 설정
split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}


# Activity-based 분할을 위한 딕셔너리 초기화
class_files = defaultdict(list)

# 각 파일을 활동 클래스에 따라 분류
for filename in os.listdir(data_path):
    if filename.endswith(".txt") and any(filename.endswith(target) for target in target_filenames):
        # 파일 경로를 활동 클래스별로 추가
        for target in target_filenames:
            if filename.endswith(target):
                class_files[target].append(filename)

# Train, Validation, Test 분할 딕셔너리
split_data = {'train': [], 'val': [], 'test': []}

# 각 클래스별로 분할 수행
for class_name, files in class_files.items():
    random.shuffle(files)  # 무작위로 셔플하여 랜덤 분할

    # 분할 인덱스 계산
    train_idx = int(len(files) * split_ratios['train'])
    val_idx = train_idx + int(len(files) * split_ratios['val'])

    # 분할 결과 할당
    split_data['train'].extend(files[:train_idx])
    split_data['val'].extend(files[train_idx:val_idx])
    split_data['test'].extend(files[val_idx:])

# 결과 확인
print("Train files:", len(split_data['train']))
print("Validation files:", len(split_data['val']))
print("Test files:", len(split_data['test']))

# same_l2_list와 different_l2_list를 각 분할 세트로 처리
def process_files(file_list, data_category):
    same_l2_list = []
    different_l2_list = []
    
    for filename in file_list:
        file_path = os.path.join(data_path, filename)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            valid_lines = [line.strip() for line in lines if len(line.strip().split(',')) == 3]
            
            for i in range(0, len(valid_lines) - 6 * 15, 15):
                sequence_images = []
                sequence_l2_labels = []
                
                for j in range(5):
                    split_data = valid_lines[i + j * 15].split(',')
                    image_path, l2_label, l3_label = split_data
                    sequence_images.append(image_path)
                    sequence_l2_labels.append(l2_label)
                
                sixth_frame_data = valid_lines[i + 5 * 15].split(',')
                l2_label_5sec = sequence_l2_labels[-1]
                l2_label_6sec = sixth_frame_data[1]

                if l2_label_5sec == l2_label_6sec:
                    same_l2_list.append((sequence_images, sequence_l2_labels, l2_label_6sec))
                else:
                    different_l2_list.append((sequence_images, sequence_l2_labels, l2_label_6sec))

    print(f"{data_category.capitalize()} set - Same L2 list:", len(same_l2_list))
    print(f"{data_category.capitalize()} set - Different L2 list:", len(different_l2_list))
    return same_l2_list, different_l2_list

# Train, Validation, Test 세트 처리
train_same, train_different = process_files(split_data['train'], 'train') #0.0204
val_same, val_different = process_files(split_data['val'], 'val') # 0.0234
test_same, test_different = process_files(split_data['test'], 'test') # 0.025

for split_name, file_list in split_data.items():
    with open(os.path.join(data_path.replace('groundTruth', 'split'), f"{split_name}_split.txt"), 'w') as f:
        for filename in file_list:
            f.write(filename + '\n')
