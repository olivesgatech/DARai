import os
from pathlib import Path

def restructure_dataset(root_dir, output_dir):
    """
    Restructure dataset to generate .txt files with paths to images for each [subject]_[session]_cam_[camera id]_[L1 class name].
    Ensures that the paths in each .txt file are ordered by frame number.
    
    Args:
        root_dir (str): Path to the root directory containing the original structure.
        output_dir (str): Path to the directory where the new structure will be saved.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_folder in root_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            
            for camera_folder in class_folder.iterdir():
                if camera_folder.is_dir():
                    camera_id = camera_folder.name.split('_')[1]
                    
                    # Dictionary to store image paths for each subject-session combination
                    path_dict = {}
                    
                    for image_file in camera_folder.glob("*.jpg"):
                        filename = image_file.stem  # without extension
                        subject, session, frame_number = filename.split('_')
                        
                        # Define key for txt file
                        txt_key = f"{subject}_{session}_cam_{camera_id}_{class_name}"
                        if txt_key not in path_dict:
                            path_dict[txt_key] = []
                        
                        # Store the relative path of the image file
                        path_dict[txt_key].append((int(frame_number), str(image_file)))
                    
                    # Write the paths to .txt files in sorted order
                    for txt_key, image_paths in path_dict.items():
                        txt_file_path = output_path / f"{txt_key}.txt"
                        with open(txt_file_path, 'w') as f:
                            # Sort by frame number and write paths
                            for _, img_path in sorted(image_paths, key=lambda x: x[0]):
                                f.write(f"{img_path}\n")
                        print(f"Created file {txt_file_path} with {len(image_paths)} image paths in sequence order.")

if __name__ == "__main__":
    root_dir = "/home/seulgi/work/RGB_sd"
    output_dir = "/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth_nov11"
    
    restructure_dataset(root_dir, output_dir)
