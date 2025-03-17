import cv2
import os
from glob import glob

# 경로 설정
rgb_video_dir = "/home/hice1/skim3513/AIFirst_F24_data/NTURGBD/nturgb+d_rgb/"
depth_mask_dir = "/home/hice1/skim3513/AIFirst_F24_data/NTURGBD/nturgb+d_depth_masked/"
output_base_dir = "/home/hice1/skim3513/AIFirst_F24_data/NTURGBD/extracted_frames/"

# RGB AVI 파일 목록 가져오기
avi_files = glob(os.path.join(rgb_video_dir, "*.avi"))

for avi_path in avi_files:
    video_name = os.path.basename(avi_path)  # e.g., S001C001P001R001A001_rgb.avi
    base_name = video_name.replace("_rgb.avi", "")  # S001C001P001R001A001
    depth_folder_path = os.path.join(depth_mask_dir, base_name)
    output_folder = os.path.join(output_base_dir, base_name)
    if os.path.exists(output_folder):
        print(f"{output_folder} exists...")
        continue

    if not os.path.exists(depth_folder_path):
        print(f"no{depth_folder_path} path exists... skip")
        continue

    # Depth mask 개수 확인 (파일 개수 카운트)
    num_depth_masks = len(glob(os.path.join(depth_folder_path, "*")))

    if num_depth_masks == 0:
        print(f"no {base_name} depth mask exists... skip")
        continue

    # AVI 파일 열기
    cap = cv2.VideoCapture(avi_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 개수

    # FPS 계산: 비디오 길이를 depth mask 개수에 맞춤
    fps_ratio = total_frames / num_depth_masks
    frame_interval = int(fps_ratio) if fps_ratio >= 1 else 1  # 최소 1로 설정

    print(f"processing...: {video_name}, total frames: {total_frames}, "
          f"number of Depth Mask: {num_depth_masks}, frame interval: {frame_interval}")

    # 출력 폴더 생성
    
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 비디오 끝

        # 특정 프레임마다 저장 (depth mask 개수에 맞춤)
        if frame_count % frame_interval == 0 and saved_count < num_depth_masks:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ {base_name} -> {saved_count}/{num_depth_masks} frame saved!")

print("All processed complete!")
