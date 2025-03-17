## Step 1. list 만들기
'''
1. 숫자가 반복되는게 있으면
2. 숫자가 건너뛰어지는게 있으면
'''

## Step 2. 

'''
01_3/cam_1/00306_Take out Kitchen and cooking tools.png, 01_3/cam_1/00317_Mix ingredients.png
'''
import os
from collections import defaultdict

def process_directory_structure(root_dir):
    frame_sequences = []#defaultdict(list)
    label_conflicts = []#defaultdict(list)

    prev_frame = None
    first_frame = None
    prev_rel_path = None
    prev_label = None
    prev_camera_id = None

    for subdir, _, files in os.walk(root_dir):
        sorted_files = sorted(files)

        for file in sorted_files:
            if file.endswith(".png"):
                # Extract the frame number and label
                path_parts = os.path.normpath(subdir).split(os.sep)
                camera_id = path_parts[-1]
                file_parts = file.split('_')
                frame_number = file_parts[0]
                label = '_'.join(file_parts[1:]).split('.')[0]

                # Generate the full path
                full_path = os.path.join(subdir, file)
                rel_path = os.path.relpath(full_path, root_dir)

                if prev_frame is None:
                    if first_frame is not None and prev_rel_path is not None:
                        frame_sequences.append((first_frame, prev_rel_path))
                    first_frame = rel_path
                elif int(frame_number) == int(prev_frame) and camera_id == prev_camera_id:
                    prev_rel_path = rel_path
                    label_conflicts.append(rel_path)
                elif int(frame_number) != int(prev_frame) + 1 or camera_id != prev_camera_id:
                    if first_frame is not None and prev_rel_path is not None:
                        frame_sequences.append((first_frame, prev_rel_path))
                    first_frame = rel_path

                prev_frame = frame_number
                prev_rel_path = rel_path
                prev_label = label
                prev_camera_id = camera_id

    # Handle the last sequence
    if prev_frame and first_frame and prev_rel_path:
        frame_sequences.append((first_frame, prev_rel_path))

    # Format the output lists
    continuous_sequence_list = []
    conflict_list = []

    for sequence in frame_sequences:
        continuous_sequence_list.append(f"{sequence[0]}, {sequence[1]}")

    # 01_2/cam_2/00008_Playing on TV.png
    prev_frame = 0#None
    conflict_list_Temp = []
    for sequence in label_conflicts:
        current_frame = sequence.split("/")[2].split("_")[0]
        if(int(prev_frame) + 1 == int(current_frame) or int(prev_frame) == int(current_frame)):
            pass
        else:
            #print(int(prev_frame), int(current_frame))
            conflict_list.append(sequence)
        
        prev_frame = current_frame

    return continuous_sequence_list, conflict_list

def save_lists_to_files(root_dir, cont_seq_list, conflict_list):
    cont_seq_file = os.path.join(root_dir, "continuous_sequence_list.txt")
    conflict_file = os.path.join(root_dir, "conflict_list.txt")

    with open(cont_seq_file, 'w') as f:
        for seq in cont_seq_list:
            f.write(seq + '\n')

    with open(conflict_file, 'w') as f:
        for conflict in conflict_list:
            f.write(conflict + '\n')

root_dir = "/mnt/data-tmp/ghazal/DARai_DATA/timestamp_order_temp"
save_dir = "/home/seulgi/work/img_to_video"

continuous_sequence_list, conflict_list = process_directory_structure(root_dir)
save_lists_to_files(save_dir, continuous_sequence_list, conflict_list)
