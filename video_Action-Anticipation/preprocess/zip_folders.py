import os
import zipfile


def zip_subfolders(parent_folder, output_folder):
    # output_folder가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # parent_folder 경로의 하위 폴더들을 모두 탐색
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        # 하위 폴더인 경우만 ZIP 파일 생성
        if os.path.isdir(folder_path):
            # 압축 파일을 output_folder 경로에 저장
            zip_file_path = os.path.join(output_folder, f"{folder_name}.zip")
            try:
                with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # 모든 파일을 ZIP 파일에 추가
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, start=folder_path)
                            zipf.write(file_path, arcname)
                print(f"Successfully compressed {folder_name} to {zip_file_path}.")
            except OSError as e:
                print(f"Failed to compress {folder_name}: {e}")

output_folder = "/home/seulgi/work/img_to_video"
parent_folder_path = '/data/sophia/AVT/DATA/frames/frames_0'
zip_subfolders(parent_folder_path, output_folder)
parent_folder_path = '/data/sophia/AVT/DATA/frames/frames_1'
zip_subfolders(parent_folder_path, output_folder)
parent_folder_path = '/data/sophia/AVT/DATA/frames/frames_2'
zip_subfolders(parent_folder_path, output_folder)