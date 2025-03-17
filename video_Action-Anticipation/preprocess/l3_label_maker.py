import os
import pandas as pd

# 경로 정의
hierarchy_labels_path = "/mnt/data-tmp/ghazal/Hierarchy_labels"
dst_path = "/data/sophia/AVT/DATA/frames"

# dst_path의 하위 폴더 확인
subfolders = [f for f in os.listdir(dst_path) if os.path.isdir(os.path.join(dst_path, f))]

# 각 하위 폴더에 대해 작업 수행
for folder in subfolders:
    # 폴더명 분석하여 subject ID, session ID, 그리고 Subject 추출
    parts = folder.split("_")
    if len(parts) < 5:
        continue

    # dst_path의 폴더명 분석하여 [subject ID], [session ID], [cam ID], [Subject] 구분
    subject_id = parts[0]  # 예: '12'
    session_id = parts[1]  # 예: '02'
    cam_id = parts[3]      # 예: '01'
    subject = " ".join(parts[4:]).replace("_", " ")  # 예: 'Writing'

    # Hierarchy_labels 폴더에서 해당 Subject가 있는지 확인
    subject_path = os.path.join(hierarchy_labels_path, subject)
    if not os.path.exists(subject_path):
        print(f"Warning: {subject} not found in Hierarchy Labels.")
        continue

    # 해당 Subject 폴더에서 'Level_3'이 포함된 CSV 파일만 가져오기
    level_3_files = [f for f in os.listdir(subject_path) if "Level_3" in f and f.endswith(".csv")]

    # 각 파일을 읽어서 처리
    for csv_file in level_3_files:
        # CSV 파일에서 subject ID와 session ID 추출하기
        file_parts = csv_file.split("_")
        if len(file_parts) < 4:
            continue
        file_subject_id = file_parts[1][1:]  # 예: 'S12' -> '12'
        file_session_id = file_parts[2][7:]  # 예: 'session02' -> '02'

        # 현재 폴더의 subject ID와 session ID가 CSV 파일의 ID와 일치하는지 확인
        if int(subject_id) == int(file_subject_id) and int(session_id) == int(file_session_id):
            csv_path = os.path.join(subject_path, csv_file)
            try:
                # CSV 파일 읽기
                df = pd.read_csv(csv_path)

                # l3.txt 파일 경로 설정
                l3_txt_path = os.path.join(dst_path, folder, "l3.txt")

                # 프레임 번호별 레이블 생성
                frame_labels = {}

                # CSV 데이터프레임의 각 행에 대해 start_frame과 end_frame 처리
                for _, row in df.iterrows():
                    activity = row['Activity'].replace(" ", "_")  # 띄어쓰기를 _로 변환
                    start_frame = int(row['start_frame'])
                    end_frame = int(row['end_frame'])

                    # start_frame과 end_frame 사이의 프레임 번호를 모두 해당 activity로 매핑
                    for frame_num in range(start_frame, end_frame + 1):
                        frame_labels[frame_num] = activity

                # dst_path의 해당 폴더 안의 프레임 파일 이름 확인
                frame_files = [f for f in os.listdir(os.path.join(dst_path, folder)) if f.endswith(".png")]

                # 프레임 파일 번호를 기준으로 정렬
                sorted_frame_files = sorted(frame_files, key=lambda x: int(x.split("_")[0]))

                # 프레임 파일 번호에 대한 매핑
                with open(l3_txt_path, 'w') as f_out:
                    for frame_file in sorted_frame_files:
                        # 파일명에서 프레임 번호 추출
                        frame_number = int(frame_file.split("_")[0])  # 예: '00027_Prepare_kitchen_appliances.png'에서 27 추출
                        label = frame_labels.get(frame_number, "None")
                        f_out.write(f"{frame_file} {label}\n")

                print(f"Successfully created: {l3_txt_path}")

            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
