import lmdb
import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
frames_dir = "/mnt/data-tmp/ghazal/DARai_DATA/l2_rgb_dataset"  # Path to your extracted frames
lmdb_path = "/mnt/data-tmp/ghazal/DARai_DATA/l2_rgb_dataset/lmdb"  # Path to save LMDB

# Define LMDB map size (adjust this as needed)
map_size = 1e12  # 1 TB

# Create the LMDB environment
env = lmdb.open(lmdb_path, map_size=int(map_size))

# Define a function to write data to LMDB
def write_lmdb(env, key, data):
    with env.begin(write=True) as txn:
        txn.put(key.encode(), data)

# Process each video sequence
with env.begin(write=True) as txn:
    for video_name in tqdm(os.listdir(frames_dir)):
        video_path = os.path.join(frames_dir, video_name)
        for frame_file in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame_file)
            
            # Read frame as image (or use features instead of raw frames)
            frame = cv2.imread(frame_path)  # Replace with feature extraction if needed

            # Optionally resize or normalize the frame
            # frame = cv2.resize(frame, (224, 224))  # Example resize if needed

            # Convert frame to bytes (or directly use if they are features)
            frame_bytes = frame.tobytes()  # For raw frames

            # Generate unique key (e.g., video name + frame number)
            key = f"{video_name}_{frame_file}"

            # Write frame to LMDB
            write_lmdb(env, key, frame_bytes)

print("LMDB creation completed.")
