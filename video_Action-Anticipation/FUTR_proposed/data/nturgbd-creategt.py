import os
from glob import glob

# 경로 설정
extracted_frames_dir = "/home/hice1/skim3513/AIFirst_F24_data/NTURGBD/extracted_frames/"
mapping_file = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/mapping_l2_changed.txt"
output_gt_dir = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/groundTruth/"

# 매핑 파일 로드
mapping_dict = {}
with open(mapping_file, "r") as f:
    for line in f:
        parts = line.strip().split(" ", 1)  # 숫자와 클래스 이름 분리
        if len(parts) == 2:
            index, class_name = parts
            mapping_dict[int(index)] = class_name.strip()

# extracted_frames 안의 모든 폴더 가져오기
folders = [f for f in os.listdir(extracted_frames_dir) if os.path.isdir(os.path.join(extracted_frames_dir, f))]

for folder in folders:
    folder_path = os.path.join(extracted_frames_dir, folder)
    output_txt_path = os.path.join(output_gt_dir, f"{folder}.txt")

    # 폴더명이 A### 형식인지 확인 (예: A010)
    if not folder.startswith("S"):
        print(f"⚠️ 폴더명 형식이 맞지 않음: {folder}, 건너뜀")
        continue

    # 매핑된 클래스 찾기
    class_index = int(folder.split("A")[-1])
    #class_index = int(folder[1:]) - 1  # A### -> 숫자로 변환 후 -1
    class_name = mapping_dict.get(class_index, "UNKNOWN")
    if class_name == "UNKNOWN":
        print(folder_path)

    # 파일 리스트 가져오기
    image_files = sorted(glob(os.path.join(folder_path, "*.png")))

    # .txt 파일 작성
    with open(output_txt_path, "w") as txt_file:
        for img_path in image_files:
            line = ",".join([img_path, class_name, "UNDEFINED"])
            txt_file.write(line + "\n")

    print(f"✅ {folder} -> {output_txt_path} 생성 완료 ({len(image_files)}개 파일 기록됨)")

print("🎉 모든 폴더 처리 완료!")
