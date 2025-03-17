import os
import pandas as pd
import shutil

def process_csv_and_generate_paths(file_path, root_dir):
    # Extract Subject ID, Session ID, and Big Title from filename
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    big_title = parts[0]
    subject_id = parts[1][1:]  # Remove 'S' from Subject ID
    session_id = int(parts[2].replace('session', ''))  # Remove 'session' prefix

    # Load CSV
    try:
        df = pd.read_csv(file_path)
    except:
        print("!!!! Do not have content in : ", file_path)
        return

    for _, row in df.iterrows():
        activity = row['Activity']#.replace(' ', '_')  # replace spaces in activity with underscores
        print(file_path, row['start_frame'], row['end_frame'])
        for frame_number in range(int(row['start_frame']), int(row['end_frame']) + 1):
            original_file_path = os.path.join(root_dir, f"{subject_id}_{session_id}/cam_2/{str(frame_number).zfill(5)}_{activity}.png")
            new_dir = os.path.join(root_dir, f"{subject_id}_{session_id}/cam_2/{big_title}")
            new_file_path = os.path.join(new_dir, f"{str(frame_number).zfill(5)}_{activity}.png")
            
            os.makedirs(new_dir, exist_ok=True)

            if os.path.exists(original_file_path):
                shutil.copy(original_file_path, new_file_path)
                #print(f"copied {original_file_path} to {new_file_path}")
            else:
                print(f"file not found: {original_file_path}")

def find_level_2_annotation_files(root_dir):
    all_annotation_files = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            annotation_files = [f for f in os.listdir(folder_path) if f.endswith('_Level_2_Annotations.csv')]

            for file in annotation_files:
                all_annotation_files.append(os.path.join(folder_path, file))

    return all_annotation_files

annotation_dir = "/mnt/data-tmp/ghazal/Hierarchy_labels"
root_dir = "/mnt/data-tmp/ghazal/DARai_DATA/timestamp_order"
annotation_files = find_level_2_annotation_files(annotation_dir)
for csv_file in annotation_files:
    process_csv_and_generate_paths(csv_file, root_dir)
