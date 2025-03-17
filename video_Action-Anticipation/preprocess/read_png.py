import os
import shutil

def replace_spaces_with_underscore(path):
    # 경로의 폴더와 파일 이름의 공백을 모두 '_'로 바꿔주는 함수
    return path.replace(' ', '_')

def process_and_copy_files(src_folder):
    for root, dirs, files in os.walk(src_folder):
        # 폴더 경로 공백 변환
        new_root = replace_spaces_with_underscore(root)
        
        # 새로운 폴더가 없다면 생성
        if not os.path.exists(new_root):
            os.makedirs(new_root)

        for file in files:
            # 파일 경로 공백 변환
            frame_path = os.path.join(root, file)
            new_frame_path = os.path.join(new_root, replace_spaces_with_underscore(file))
            
            # 파일을 새로운 경로로 복사
            try:
                shutil.copy2(frame_path, new_frame_path)
            except:
                print(frame_path, new_frame_path)
                continue
            print(f"Copied {frame_path} to {new_frame_path}")

# 사용 예시
src_folder = '/data/sophia/AVT/DATA/frames'
process_and_copy_files(src_folder)
