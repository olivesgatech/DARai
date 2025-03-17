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

def create_and_copy_corrupted_files(src_folder, dst_folder):
    """
    Traverse the src_folder to create a new folder structure in dst_folder
    and copy only corrupted files from the src_folder to the dst_folder.
    """
    for root, dirs, files in os.walk(src_folder):
        # Get the relative path from the source folder
        relative_path = os.path.relpath(root, src_folder)
        path_parts = relative_path.split(os.sep)  # Split the path into parts

        # If the path has at least 3 parts, create a new folder name
        if len(path_parts) >= 3:
            # Construct the new folder name (e.g., 13_3_cam_2)
            new_folder_name = f"{path_parts[0]}_{path_parts[1]}_{path_parts[2]}"
            
            # Create the corresponding path in the destination folder
            new_dst_folder = os.path.join(dst_folder, new_folder_name)
            
            # Ensure all folders have underscores instead of spaces
            new_dst_folder = replace_spaces_with_underscore(new_dst_folder)

            # Create the destination folder if it doesn't exist
            os.makedirs(new_dst_folder, exist_ok=True)

            # Check each file in the source folder
            for file in files:
                src_file_path = os.path.join(root, file)
                
                # Construct the corresponding destination file path
                dst_file_path = os.path.join(new_dst_folder, replace_spaces_with_underscore(file))
                
                # Check if the destination file is corrupted
                if os.path.exists(dst_file_path):
                    if is_image_corrupted(dst_file_path):
                        # Replace the corrupted destination file with the source file
                        shutil.copy(src_file_path, dst_file_path)
                        print(f"Replaced corrupted file {dst_file_path} with {src_file_path}")
                    else:
                        print("not corrupted")
                else:
                    print(f"File {dst_file_path} does not exist, skipping.")

# Define your source and destination folders
src_folder = '/mnt/data-tmp/ghazal/DARai_DATA/timestamp_order'
dst_folder = '/data/sophia/AVT/DATA/frames'

# Copy only corrupted files to the destination folder
create_and_copy_corrupted_files(src_folder, dst_folder)
