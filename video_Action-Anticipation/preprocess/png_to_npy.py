import os
import numpy as np
import cv2

# Function to convert a folder of PNG images to NPY format
def convert_images_to_npy(src_folder, dst_folder):
    """
    Traverse the src_folder, convert each PNG file to a Numpy array, and save it as a .npy file in dst_folder.
    The folder structure in src_folder is preserved in dst_folder.
    """
    for root, _, files in os.walk(src_folder):
        # Get the relative path from the source folder
        relative_path = os.path.basename(root)
        
        # Create a corresponding destination folder
        dst_subfolder = os.path.join(dst_folder, relative_path)
        os.makedirs(dst_subfolder, exist_ok=True)  # Create if it doesn't exist

        # Iterate through each file in the current folder
        for file in files:
            if file.endswith('.png'):
                src_file_path = os.path.join(root, file)
                dst_file_name = os.path.splitext(file)[0] + ".npy"  # Change extension to .npy
                dst_file_path = os.path.join(dst_subfolder, dst_file_name)

                # if(os.path.exists(dst_file_path)):
                #     print("continue...", dst_file_path)                    
                #     continue

                # Read the image and convert it to a Numpy array
                img = cv2.imread(src_file_path, cv2.IMREAD_COLOR)  # Read the image using OpenCV
                if img is None:
                    print(f"Skipping corrupted file: {src_file_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB

                # Save the Numpy array as a .npy file
                print(img.shape)
                np.save(dst_file_path, img)
                print(f"Saved: {dst_file_path}")

# Define source folder and destination folder

dst_folder = '/data/sophia/AVT/DATA/npy_frames'
src_folder_list = [
    #'/data/sophia/AVT/DATA/frames/14_3_cam_2_Stocking_up_pantry',
    #'/data/sophia/AVT/DATA/frames/01_4_cam_2_Stocking_up_pantry',
    #'/data/sophia/AVT/DATA/frames/13_2_cam_1_Using_handheld_smart_devices',
    #'/data/sophia/AVT/DATA/frames/13_3_cam_1_Stocking_up_pantry',
    #'/data/sophia/AVT/DATA/frames/13_3_cam_2_Stocking_up_pantry',
    #'/data/sophia/AVT/DATA/frames/15_1_cam_1_Using_handheld_smart_devices',
    #'/data/sophia/AVT/DATA/frames/18_1_cam_1_Using_handheld_smart_devices',
    #'/data/sophia/AVT/DATA/frames/03_4_cam_1_Making_a_cup_of_coffee_in_coffee_maker',
    #'/data/sophia/AVT/DATA/frames/15_2_cam_1_Using_handheld_smart_devices',
    #'/data/sophia/AVT/DATA/frames/13_4_cam_1_Making_a_cup_of_coffee_in_coffee_maker',
    #'/data/sophia/AVT/DATA/frames/08_3_cam_2_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/05_3_cam_2_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/01_4_cam_1_Making_a_cup_of_coffee_in_coffee_maker',
    #'/data/sophia/AVT/DATA/frames/01_3_cam_2_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/03_3_cam_1_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/07_3_cam_2_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/08_3_cam_1_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/10_3_cam_1_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/10_3_cam_2_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/20_1_cam_2_Using_handheld_smart_devices',
    #'/data/sophia/AVT/DATA/frames/01_4_cam_1_Stocking_up_pantry',
    #'/data/sophia/AVT/DATA/frames/06_3_cam_1_Stocking_up_pantry',
    #'/data/sophia/AVT/DATA/frames/06_3_cam_2_Stocking_up_pantry',
    #'/data/sophia/AVT/DATA/frames/03_4_cam_2_Stocking_up_pantry',
    #'/data/sophia/AVT/DATA/frames/18_1_cam_2_Using_handheld_smart_devices',
    #'/data/sophia/AVT/DATA/frames/20_2_cam_1_Using_handheld_smart_devices',
    #'/data/sophia/AVT/DATA/frames/14_3_cam_1_Stocking_up_pantry',
    #'/data/sophia/AVT/DATA/frames/13_3_cam_1_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/05_3_cam_1_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/13_3_cam_2_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/13_4_cam_2_Making_a_cup_of_coffee_in_coffee_maker',
    #'/data/sophia/AVT/DATA/frames/03_3_cam_2_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/03_4_cam_2_Making_a_cup_of_coffee_in_coffee_maker',
    #'/data/sophia/AVT/DATA/frames/07_3_cam_1_Making_a_cup_of_instant_coffee',
    #'/data/sophia/AVT/DATA/frames/15_2_cam_2_Using_handheld_smart_devices',
    #'/data/sophia/AVT/DATA/frames/13_2_cam_2_Using_handheld_smart_devices',
    #'/data/sophia/AVT/DATA/frames/15_1_cam_2_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/20_2_cam_2_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/20_1_cam_1_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/01_4_cam_2_Making_a_cup_of_coffee_in_coffee_maker',
]
# Convert images to Numpy array format
for src_folder in src_folder_list:
    convert_images_to_npy(src_folder, dst_folder)
