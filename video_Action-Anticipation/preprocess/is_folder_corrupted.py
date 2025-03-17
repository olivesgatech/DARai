import os
from PIL import Image

def is_image_corrupted(file_path):
    """Check if the image file is corrupted by trying to open it using PIL."""
    try:
        _ = Image.open(file_path).convert('RGB')
        return False  # If no exception, image is not corrupted
    except Exception:
        return True  # If exception, image is corrupted

def find_folders_without_corrupted_images(dst_folder):
    """
    Traverse dst_folder and identify folders without corrupted images.
    
    Args:
    - dst_folder: Destination folder to traverse.
    
    Returns:
    - List of folder paths that do not contain any corrupted images.
    """
    folders_without_corruption = []  # List to store folders without corrupted images

    for root, _, files in os.walk(dst_folder):
        all_files_valid = True  # Flag to indicate if all images in the folder are valid
        
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                if is_image_corrupted(file_path):  # Check if the image is corrupted
                    all_files_valid = False  # If one image is corrupted, set flag to False
                    break

        # If all images are valid in the folder, add to the result list
        if all_files_valid:
            folders_without_corruption.append(root)
            print(f"Valid folder: {root}")
        else:
            print(f"Folder {root} contains corrupted images, skipping.")

    return folders_without_corruption

# Define your destination folder
dst_folder = '/data/sophia/AVT/DATA/frames'

# Get the list of folders without corrupted images
valid_folders = find_folders_without_corrupted_images(dst_folder)

# Print the results
print("Folders without corrupted images:")
for folder in valid_folders:
    print(folder)
