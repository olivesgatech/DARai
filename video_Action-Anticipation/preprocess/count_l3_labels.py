import os
from collections import defaultdict

# 경로 정의
dst_path = "/data/sophia/AVT/DATA/frames"

# 전체 레이블 빈도수 저장을 위한 딕셔너리
global_label_count = defaultdict(int)

# 하위 폴더 확인
subfolders = [f for f in os.listdir(dst_path) if os.path.isdir(os.path.join(dst_path, f))]

# 각 폴더에 대해 작업 수행
for folder in subfolders:
    # l3.txt 파일이 해당 폴더에 있는지 확인
    l3_txt_path = os.path.join(dst_path, folder, "l3.txt")
    if not os.path.exists(l3_txt_path):
        print(f"No l3.txt file found in {folder}. Skipping...")
        continue

    # l3.txt 파일 읽기
    with open(l3_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 각 줄에서 레이블 추출 (파일명과 레이블로 구분되어 있음)
            parts = line.strip().split(" ")
            if len(parts) == 2:
                label = parts[1]
                global_label_count[label] += 1

# 전체 결과를 global_label_count.txt 파일로 저장
global_label_count_path = os.path.join(dst_path, "global_label_count.txt")
with open(global_label_count_path, 'w') as out_file:
    for label, count in sorted(global_label_count.items()):
        out_file.write(f"{label}: {count}\n")

print(f"Global label count saved to {global_label_count_path}")
