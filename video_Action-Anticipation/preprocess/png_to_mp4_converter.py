import cv2
import os

image_folder = "/mnt/data-tmp/ghazal/DARai_DATA/timestamp_order"
video_name = 'video.mp4'

for folder_name in os.listdir(image_folder):
    folder_path = os.path.join(image_folder, folder_name)
    if os.path.isdir(folder_path):
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m','p','4','v'), 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()