import os
import shutil

def copy_folders_for_consecutive_and_repeated_frames(src_folder, dst_folder):
    # os.walk로 폴더 구조 탐색
    for root, dirs, files in os.walk(src_folder):
        # PNG 파일만 필터링
        png_files = sorted([file for file in files if file.endswith(".png")])
        if not png_files:
            continue

        previous_frame = None
        current_folder_files = []
        
        # root 폴더의 이름만 가져와 dst_folder에 붙여 사용
        base_folder_name = os.path.basename(root)

        for file in png_files:
            # 파일명에서 frame number 추출
            frame_number = int(file.split("_")[0])

            if previous_frame is None or frame_number == previous_frame + 1 or frame_number == previous_frame:
                # 연속되거나 반복된 경우, 현재 폴더에 추가
                current_folder_files.append(file)
            else:
                # 연속되지 않는 경우, 이전 파일들을 새로운 폴더로 복사
                if current_folder_files:
                    first_frame = current_folder_files[0].split("_")[0]
                    last_frame = current_folder_files[-1].split("_")[0]
                    folder_name = f"{base_folder_name}_{first_frame}-{last_frame}"
                    new_folder_path = os.path.join(dst_folder, folder_name)
                    os.makedirs(new_folder_path, exist_ok=True)

                    # 파일들을 새로운 폴더로 복사
                    for f in current_folder_files:
                        src_file_path = os.path.join(root, f)
                        dst_file_path = os.path.join(new_folder_path, f)
                        shutil.copy(src_file_path, dst_file_path)

                # 새로운 폴더 시작
                current_folder_files = [file]

            # 다음 파일과 비교를 위해 frame number 저장
            previous_frame = frame_number

        # 마지막 그룹 파일 처리
        if current_folder_files:
            first_frame = current_folder_files[0].split("_")[0]
            last_frame = current_folder_files[-1].split("_")[0]
            folder_name = f"{base_folder_name}_{first_frame}-{last_frame}"
            new_folder_path = os.path.join(dst_folder, folder_name)
            os.makedirs(new_folder_path, exist_ok=True)

            for f in current_folder_files:
                src_file_path = os.path.join(root, f)
                dst_file_path = os.path.join(new_folder_path, f)
                shutil.copy(src_file_path, dst_file_path)

# 사용 예시
src_folder = '/data/sophia/AVT/DATA/frames'  # frames 폴더 경로
dst_folder = '/data/sophia/AVT/DATA/frames_copied'  # 새로운 복사 폴더 경로
copy_folders_for_consecutive_and_repeated_frames(src_folder, dst_folder)
