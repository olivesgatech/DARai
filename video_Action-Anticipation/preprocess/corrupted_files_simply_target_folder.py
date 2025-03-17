import os
import shutil
from PIL import Image

def replace_spaces_with_underscore(path):
    """Replace all spaces in the path with underscores."""
    return path.replace(' ', '_')

def is_image_corrupted(file_path):
    """Check if the image file is corrupted by trying to open it using PIL."""
    try:
        _ = Image.open(file_path).convert('RGB')
        return False  # If no exception, image is not corrupted
    except Exception:
        return True  # If exception, image is corrupted

def copy_corrupted_files_from_selected_folders(src_folder, dst_folder, target_folders):
    """
    Traverse only the selected target folders in dst_folder,
    and copy only corrupted files from src_folder to dst_folder.
    """
    for target in target_folders:
        # Construct the corresponding source folder path
        relative_path = os.path.relpath(target, dst_folder)
        
        # Split the target folder into path components
        path_parts = relative_path.split("_")
        
        # Group the path parts to construct the source folder path
        if len(path_parts) >= 5:  # Ensure we have enough parts for grouping
            # Combine parts as described: '13_3', 'cam_2', 'Making a cup of coffee in coffee maker'
            src_relative_path = os.path.join(
                f"{path_parts[0]}_{path_parts[1]}",  # e.g., '13_3'
                f"{path_parts[2]}_{path_parts[3]}",  # e.g., 'cam_2'
                " ".join(path_parts[4:]).replace('_', ' ')  # Combine remaining parts and replace underscores with spaces
            )
            
            src_path = os.path.join(src_folder, src_relative_path)

            # Create the destination folder if it doesn't exist
            if not os.path.exists(target):
                os.makedirs(target, exist_ok=True)

            # Check and copy only corrupted files
            for file in os.listdir(target):
                dst_file_path = os.path.join(target, file)
                
                if file == 'labels.txt':
                    continue

                # Only process corrupted files in the target folder
                if is_image_corrupted(dst_file_path):
                    srcfilesplit = file.split("_")
                    srcfile = file.replace(srcfilesplit[0]+'_', '').replace('_', ' ')
                    srcfile = srcfilesplit[0]+'_'+srcfile
                    src_file_path = os.path.join(src_path, srcfile)
                    
                    if os.path.exists(src_file_path):
                        shutil.copy(src_file_path, dst_file_path)
                        print(f"Replaced corrupted file {dst_file_path} with {src_file_path}")
                    else:
                        print(f"Source file {src_file_path} not found for corrupted file {dst_file_path}. Skipping.")
                else:
                    print("not corrupted")

# Define your source and destination folders
src_folder = '/mnt/data-tmp/ghazal/DARai_DATA/timestamp_order'
dst_folder = '/data/sophia/AVT/DATA/frames'

# Specify the folders to process (as provided in your query)
target_folders = [
    '/data/sophia/AVT/DATA/frames/03_4_cam_1_Stocking_up_pantry',##############################
]

# Copy only corrupted files to the destination folder
copy_corrupted_files_from_selected_folders(src_folder, dst_folder, target_folders)
