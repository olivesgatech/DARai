import os
import pandas as pd

def process_csv_and_generate_paths(level2_path, level3_path, root_dir, output_dir):
    # Extract metadata from filename for output file naming
    filename = os.path.basename(level2_path)
    parts = filename.split('_')
    l1_class_name = parts[0]  # L1 Class name
    subject_id = parts[1][1:]  # Remove 'S' from Subject ID
    session_id = int(parts[2].replace('session', ''))  # Remove 'session' prefix

    # Load Level 2 CSV
    try:
        df_level2 = pd.read_csv(level2_path)
    except:
        print("!!!! Do not have content in : ", level2_path)
        return

    # Load Level 3 CSV
    try:
        df_level3 = pd.read_csv(level3_path)
    except:
        print("!!!! Do not have content in : ", level3_path)
        return

    # Collect activities from Level 2 by frame
    level2_activities = {}
    for _, row in df_level2.iterrows():
        try:
            activity = row['Activity'].replace(" ", "_")
        except:
            continue
        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])
        for frame_number in range(start_frame, end_frame + 1):
            level2_activities[frame_number] = activity

    # Collect activities from Level 3 by frame
    level3_activities = {}
    for _, row in df_level3.iterrows():
        try:
            l3_activity = row['Activity'].replace(" ", "_")
        except:
            continue
        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])
        for frame_number in range(start_frame, end_frame + 1):
            level3_activities[frame_number] = l3_activity

    # Loop over both camera directories
    for camera in ["camera_1_fps_15", "camera_2_fps_15"]:
        # Define the output .txt file path based on metadata
        txt_file_path = os.path.join(output_dir, f"{subject_id}_{session_id}_{camera}_{l1_class_name}.txt")
        
        # Determine the full range of frames
        max_frame = max(max(level2_activities.keys(), default=0), max(level3_activities.keys(), default=0))

        # Write paths to output file, aligning Level 2 and Level 3 activities
        with open(txt_file_path, 'w') as txt_file:
            for frame_number in range(max_frame + 1):  # Include all frames up to the max
                frame_file_name = f"{str(frame_number).zfill(5)}.jpg"
                frame_path = os.path.join(root_dir, f"{l1_class_name}/{camera}/{subject_id}_{session_id}_{frame_file_name}")

                # Get activity and l3_activity, with empty values if undefined
                activity = level2_activities.get(frame_number, "UNDEFINED")
                l3_activity = level3_activities.get(frame_number, "UNDEFINED")

                # Write line with both activities
                txt_file.write(f"{frame_path}, {activity}, {l3_activity}\n")

        print(f"Written all frames with combined activities to {txt_file_path} for {camera}")

def find_annotation_file_pairs(root_dir):
    level2_files = {}
    level3_files = {}
    
    # Find all Level 2 and Level 3 annotation files
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith("_Level_2_Annotations.csv"):
                    base_name = file_name.replace("_Level_2_Annotations.csv", "")
                    level2_files[base_name] = file_path
                if file_name.endswith("_Level_3_Annotations.csv"):
                    base_name = file_name.replace("_Level_3_Annotations.csv", "")
                    level3_files[base_name] = file_path

    # Match Level 2 and Level 3 files by base name
    file_pairs = []
    for base_name, level2_path in level2_files.items():
        level3_path = level3_files.get(base_name)
        if level3_path:
            file_pairs.append((level2_path, level3_path))

    return file_pairs

annotation_dir = "/home/seulgi/work/hierarchy_labels_Nov11"
root_dir = "/home/seulgi/work/RGB_sd"
output_dir = "/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth_nov11"
os.makedirs(output_dir, exist_ok=True)

# Process each pair of Level 2 and Level 3 annotation files
for level2_path, level3_path in find_annotation_file_pairs(annotation_dir):
    process_csv_and_generate_paths(level2_path, level3_path, root_dir, output_dir)
