import os
from glob import glob
import random
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2



class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length=16):
        """
        Args:
            root_dir (str): Root directory where the dataset is stored.
            sequence_length (int): Number of frames to select uniformly.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.video_samples = self._get_all_video_samples()

    def _get_all_video_samples(self):
        """
        Returns:
            list of tuples: Each tuple contains (activity, camera, subject_id, session_id)
        """
        video_samples = []
        for activity in os.listdir(self.root_dir):  # Iterate over activities
            activity_path = os.path.join(self.root_dir, activity)
            if not os.path.isdir(activity_path):
                continue  # Skip non-folder items

            for camera in os.listdir(activity_path):  # Iterate over cameras
                camera_path = os.path.join(activity_path, camera)
                if not os.path.isdir(camera_path):
                    continue

                # Find unique subject-session pairs in this camera folder
                all_frames = glob(os.path.join(camera_path, "*.jpg"))
                subject_sessions = set()

                for frame_path in all_frames:
                    filename = os.path.basename(frame_path)
                    parts = filename.split("_")

                    if len(parts) >= 3:
                        subject_id, session_id = parts[0], parts[1]
                        subject_sessions.add((subject_id, session_id))

                # Add all (Activity, Camera, Subject, Session) combinations
                for subject_id, session_id in subject_sessions:
                    video_samples.append((activity, camera, subject_id, session_id))

        return video_samples

    def _get_frames_from_video_sample(self, activity, camera, subject_id, session_id):
        """
        Returns:
            list: Sorted list of frame file paths.
        """
        video_sample_path = os.path.join(self.root_dir, activity, camera)
        all_frames = sorted(glob(os.path.join(video_sample_path, f"{subject_id}_{session_id}_*.jpg")))
        return all_frames

    def _select_uniform_frames(self, frames):
        """
        Returns:
            list: Selected frame file paths.
        """
        if len(frames) < self.sequence_length:
            # Padding last frame
            frames += [frames[-1]] * (self.sequence_length - len(frames))
        # if len(frames) < self.sequence_length:
        #     #cyclic padding
        #     frames = (frames * (self.sequence_length // len(frames) + 1))[:self.sequence_length]
        # if len(frames) < self.sequence_length:
        #     #Linear Interpolation
        #     indices = np.linspace(0, len(frames) - 1, self.sequence_length).astype(int)
        #     frames = [frames[i] for i in indices]
        else:
            step = max(len(frames) // self.sequence_length, 1)
            offset = random.randint(0, step - 1) if step > 1 else 0
            frames = sorted(frames[i] for i in range(offset, len(frames), step)[:self.sequence_length])

        return frames

    def __len__(self):
        """
        Returns the total number of video samples in the dataset.
        """
        return len(self.video_samples)

    def __getitem__(self, idx):
        """
        Returns:
            np.ndarray: Stacked array of selected frames.
            str: Corresponding activity label.
            tuple: (subject_id, session_id) for reference.
        """
        activity, camera, subject_id, session_id = self.video_samples[idx]
        frames = self._get_frames_from_video_sample(activity, camera, subject_id, session_id)

        if not frames:
            raise ValueError(f"No frames found for {activity}/{camera}/{subject_id}_{session_id}")

        selected_frames = self._select_uniform_frames(frames)

        # Load and stack frames as NumPy arrays
        frame_arrays = [np.array(Image.open(frame).convert('RGB')) for frame in selected_frames]
        return np.stack(frame_arrays), activity ,camera, (subject_id, session_id) , selected_frames

#Example Usage:
#dataset = VideoFrameDataset(root_dir="/mnt/Data1/RGB_sd", sequence_length=16)
#print(len(dataset))
# gif = create_gif_from_frames(frames, f"{activity}_{camera}_{subject_id}_{session_id}")

class Video_Dataset(Dataset):
    def __init__(self, root_dir, sequence_length=16, output_video_dir="./video_outputs", fps=15):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.output_video_dir = output_video_dir
        self.fps = fps
        os.makedirs(self.output_video_dir, exist_ok=True)
        self.video_samples = self.get_all_video_samples()

    def get_all_video_samples(self):
        video_samples = []
        for activity in os.listdir(self.root_dir):
            activity_path = os.path.join(self.root_dir, activity)
            if not os.path.isdir(activity_path):
                continue
            for camera in os.listdir(activity_path):
                camera_path = os.path.join(activity_path, camera)
                if not os.path.isdir(camera_path):
                    continue
                all_frames = glob(os.path.join(camera_path, "*.jpg"))
                subject_sessions = set()
                for frame_path in all_frames:
                    parts = os.path.basename(frame_path).split("_")
                    if len(parts) >= 3:
                        subject_id, session_id = parts[0], parts[1]
                        subject_sessions.add((subject_id, session_id))
                for subject_id, session_id in subject_sessions:
                    video_samples.append((activity, camera, subject_id, session_id))
        return video_samples

    def get_frames_from_video_sample(self, activity, camera, subject_id, session_id):
        video_sample_path = os.path.join(self.root_dir, activity, camera)
        return sorted(glob(os.path.join(video_sample_path, f"{subject_id}_{session_id}_*.jpg")))

    def select_uniform_frames(self, frames):
        if len(frames) < self.sequence_length:
            frames += [frames[-1]] * (self.sequence_length - len(frames))
        else:
            step = max(len(frames) // self.sequence_length, 1)
            offset = 0  # deterministic, or random if needed
            frames = sorted(frames[i] for i in range(offset, len(frames), step)[:self.sequence_length])
        return frames

    def create_video_from_frames(self, frames, save_path):
        if os.path.exists(save_path):
            return save_path  # Skip re-creation if already exists
        frame0 = cv2.imread(frames[0])
        height, width, _ = frame0.shape
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
        for frame_path in frames:
            frame = cv2.imread(frame_path)
            out.write(frame)
        out.release()
        return save_path

    def __len__(self):
        return len(self.video_samples)

    def __getitem__(self, idx):
        activity, camera, subject_id, session_id = self.video_samples[idx]
        frames = self.get_frames_from_video_sample(activity, camera, subject_id, session_id)

        if not frames:
            raise ValueError(f"No frames found for {activity}/{camera}/{subject_id}_{session_id}")

        selected_frames = self.select_uniform_frames(frames)

        video_filename = f"{activity}_{subject_id}_{session_id}-{camera}.mp4"
        video_path = os.path.join(self.output_video_dir, video_filename)

        video_path = self.create_video_from_frames(selected_frames, video_path)

        return video_path, activity, camera, (subject_id, session_id)

#Example Usage
# dataset = Video_Dataset( root_dir="path to dataset", sequence_length=32, output_video_dir="path to output", fps=1)
# print("Video saved at:", video_path)
