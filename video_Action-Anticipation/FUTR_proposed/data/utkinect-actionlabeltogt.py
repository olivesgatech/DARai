import os
import re

dataset_path = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect/RGB"
action_labels_path = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect/actionLabel.txt"
groundTruth_path = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect/groundTruth"

# 정규 표현식 패턴 (숫자 추출)
num_pattern = re.compile(r'(\d+)')

# **1. actionlabels.txt 파일 읽기 및 파싱**
session_labels = {}
current_session = None

with open(action_labels_path, "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue

        # 세션 이름 (예: s01_e01)
        if line.startswith("s") and ('_e' in line):
            current_session = line
            session_labels[current_session] = {}
        else:
            # 동작 레이블 및 범위 파싱 (예: walk: 252 390)
            parts = line.split(":")
            action = parts[0].strip()
            start, end = map(int, parts[1].strip().split())

            # 레이블 저장
            session_labels[current_session][action] = (start, end)

# **2. 데이터셋 폴더에서 세션별로 처리**
for session, labels in session_labels.items():
    session_path = os.path.join(dataset_path, session)

    if not os.path.isdir(session_path):
        print(f" {session} 폴더가 존재하지 않습니다. 건너뜁니다.")
        continue

    # 해당 세션 폴더 내 이미지 파일 가져오기
    image_files = [f for f in os.listdir(session_path) if f.endswith(".jpg")]
    
    # 숫자 기준으로 **오름차순 정렬**
    image_files.sort(key=lambda x: int(num_pattern.search(x).group()))

    # txt 파일 생성
    txt_file_path = os.path.join(groundTruth_path, f"{session}.txt")
    with open(txt_file_path, "w") as txt_file:
        for img_file in image_files:
            # 파일명에서 숫자 추출
            img_num = int(num_pattern.search(img_file).group())
            
            # 해당 숫자가 포함된 레이블 찾기
            label = "UNDEFINED"
            for action, (start, end) in labels.items():
                if start <= img_num <= end:
                    label = action
                    break
            
            # 파일 경로, 레이블 저장
            txt_file.write(f"{os.path.join(session_path, img_file)}, {label}, UNDEFINED\n")
    
    print(f" {session}.txt 파일 생성 완료! (오름차순 정렬)")

