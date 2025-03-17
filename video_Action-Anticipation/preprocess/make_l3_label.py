import os

# 데이터 경로 설정
data_path = '/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth_nov11'

# 필터링할 파일 이름 목록
target_filenames = [
    '_Dining.txt', 'Cleaning the kitchen.txt', 'Making pancake.txt', 
    'Cleaning dishes.txt', 'Using handheld smart devices.txt', 
    'Making a cup of instant coffee.txt', 'Making a cup of coffee in coffee maker.txt'
]

# L2 레이블을 저장할 세트
unique_l2_labels = set()

# 각 파일을 iterate하여 L2 레이블 수집
for filename in os.listdir(data_path):
    if filename.endswith(".txt") and any(filename.endswith(target) for target in target_filenames):
        file_path = os.path.join(data_path, filename)
        
        with open(file_path, 'r') as file_content:
            lines = file_content.readlines()
            for line in lines:
                split_data = line.strip().split(',')
                if len(split_data) == 3:
                    l2_label = split_data[2]  # L3 레이블 추출
                    if l2_label == ' ' or l2_label == '':
                        continue
                    unique_l2_labels.add(l2_label)

# L2 레이블에 번호 매기기
l2_label_mapping = {label: idx for idx, label in enumerate(sorted(unique_l2_labels))}

# 매핑 파일 저장
output_dir = '/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai'
os.makedirs(output_dir, exist_ok=True)
mapping_file_path = os.path.join(output_dir, 'mapping_l3_changed.txt')

with open(mapping_file_path, 'w') as mapping_file:
    for label, idx in l2_label_mapping.items():
        mapping_file.write(f"{idx} {label}\n")

print(f"L3 label mapping saved in {mapping_file_path}")
