import os
import shutil

def replace_spaces_with_underscore(path):
    # 경로의 폴더와 파일 이름의 공백을 모두 '_'로 바꿔주는 함수
    return path.replace(' ', '_')

def copy_files_with_new_folder_structure(src_folder, dst_folder):
    # os.walk를 사용하여 src_folder를 재귀적으로 탐색
    for root, dirs, files in os.walk(src_folder):
        # root에서 폴더 경로를 분할하여 새로운 폴더 이름 생성
        relative_path = os.path.relpath(root, src_folder)
        path_parts = relative_path.split(os.sep)  # 경로를 분할
        
        # 새로운 폴더 이름 만들기 (예: 02_1_cam_1_Carrying Object)
        if len(path_parts) >= 3:  # 폴더 깊이가 4 이상이어야 함
            new_folder_name = f"{path_parts[0]}_{path_parts[1]}_{path_parts[2]}"

            # 새로운 목적지 경로 생성
            new_dst_folder = os.path.join(dst_folder, new_folder_name)
            new_dst_folder = replace_spaces_with_underscore(new_dst_folder)
            
            # 목적지 폴더가 없으면 생성
            os.makedirs(new_dst_folder, exist_ok=True)
            
            # 해당 폴더 안의 파일을 모두 복사
            for file in files:
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(new_dst_folder, replace_spaces_with_underscore(file))
                
                shutil.copy(src_file_path, dst_file_path)
                print(f"Copied {src_file_path} to {dst_file_path}")

# 사용 예시
src_folder = '/mnt/data-tmp/ghazal/DARai_DATA/timestamp_order'
dst_folder = '/data/sophia/AVT/DATA/frames'
copy_files_with_new_folder_structure(src_folder, dst_folder)
