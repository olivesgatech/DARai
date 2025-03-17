import os
from pathlib import Path
import shutil

def restructure_dataset(root_dir, output_dir):
    """
    Restructure dataset from [camera id]/[class name]/[subject]_[session]_[frame number].png
    to [subject]_[session]/[camera id]/[frame number]_[class name].png.
    
    Args:
        root_dir (str): Path to the root directory containing the original structure.
        output_dir (str): Path to the directory where the new structure will be saved.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for camera_folder in root_path.iterdir():
        if camera_folder.is_dir():
            camera_id = camera_folder.name
            
            for class_folder in camera_folder.iterdir():
                if class_folder.is_dir():
                    class_name = class_folder.name
                    
                    for image_file in class_folder.glob("*.png"):
                        filename = image_file.stem  # without extension
                        subject, session, frame_number = filename.split('_')
                        
                        # New structure
                        new_folder = output_path / f"{subject}_{session}" / camera_id
                        new_folder.mkdir(parents=True, exist_ok=True)
                        
                        new_filename = frame_number+'_'+class_name+'.png'
                        new_file_path = os.path.join(new_folder, new_filename)
                        print(frame_number, class_name, new_filename, new_file_path)
                        
                        shutil.copy(image_file, new_file_path)

if __name__ == "__main__":
    root_dir = "/mnt/data-tmp/ghazal/DARai_DATA/l2_rgb_dataset/train"
    output_dir = "/mnt/data-tmp/ghazal/DARai_DATA/timestamp_order"
    
    restructure_dataset(root_dir, output_dir)
