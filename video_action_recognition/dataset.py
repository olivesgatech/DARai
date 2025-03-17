import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import random
from utils import plot_sequences




class Custom3DDataset(Dataset):
    def __init__(self, root_dir, include_classes, sequence_length=16, sampling="multi-uniform", transform=None , max_seq=3 ):

        self.root_dir = root_dir
        if "depth" in self.root_dir:
            self.channel = 1
        else:
            self.channel = 3
        if len(include_classes) > 0:
            self.classes = [cls for cls in os.listdir(root_dir) if cls in include_classes]
        else:
            self.classes = os.listdir(root_dir)

        # self.classes = os.listdir(root_dir)
        self.class_names = sorted(self.classes)  # Store class names
        self.sequence_length = sequence_length
        self.transform = transform
        self.sampling = sampling
        self.number_of_seq = max_seq
        self.sequences = self._create_sequences()


    def _create_sequences(self):
        sequences = []
        unique_id = -1
        for activity in self.classes:
            activity_dir = os.path.join(self.root_dir, activity)
            frames = sorted(glob(os.path.join(activity_dir, '*.png')))
            grouped_frames = self._group_frames_by_subject_and_session(frames)

            for subject_session in grouped_frames.keys():
                frames = grouped_frames.get(subject_session, [])
                unique_id += 1
                if len(frames) < self.sequence_length:
                    # Pad the sequence
                    sequence = self._pad_sequence(frames)
                    sequences.append((sequence, activity , unique_id))
                else:
                    if self.sampling == "multiple-consecutive":
                        seq_counter = 0
                        for i in range(0, len(frames) - self.sequence_length + 1):
                            while seq_counter < self.number_of_seq:
                                sequence = frames[i:i + self.sequence_length]
                                sequences.append((sequence, activity , unique_id))
                                seq_counter += 1
                    # elif self.sampling == "multiple-random":
                    #     seq_counter = 0
                    #     for _ in range(0, len(frames) - self.sequence_length + 1):
                    #         while seq_counter < self.number_of_seq :
                    #             start_idx = random.randint(0, len(frames) - self.sequence_length)
                    #             sequence = frames[start_idx:start_idx + self.sequence_length]
                    #             sequences.append((sequence, activity , unique_id))
                    #             seq_counter += 1
                    elif self.sampling == "single-random":
                        seq_counter = 0
                        while seq_counter < self.number_of_seq:
                            sequence = sorted(random.sample(frames, self.sequence_length))
                            sequences.append((sequence, activity , unique_id))

                            seq_counter += 1
                    elif self.sampling == "multi-uniform":
                        seq_counter = 0
                        while seq_counter < self.number_of_seq:
                            step = max(len(frames) // self.sequence_length, 1)
                            offset = random.randint(0, step - 1) if step > 1 else 0
                            sequence = sorted(frames[i] for i in range(offset, len(frames), step)[:self.sequence_length])
                            sequences.append((sequence, activity, unique_id))

                            seq_counter += 1
        return sequences

    def _pad_sequence(self, frames):
        if len(frames) == 0:
            return frames  # Avoid division by zero if frames is empty
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])  # Repeat the last frame
        return frames

    def _group_frames_by_subject_and_session(self, frames):
        grouped_frames = {}
        for frame in frames:
            filename = os.path.basename(frame)
            subject_session = '_'.join(filename.split('_')[:2])
            if subject_session not in grouped_frames:
                grouped_frames[subject_session] = []
            grouped_frames[subject_session].append(frame)
        return grouped_frames

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, activity , sample_id = self.sequences[idx]
        frames = []
        for frame_path in sequence:
            if self.channel == 1:
                image = Image.open(frame_path).convert('L')  # Convert to grayscale
            else:
                image = Image.open(frame_path).convert('RGB')  # Convert to RGB
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        frames = torch.stack(frames)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # Change to (C, T, H, W)
        # print(f"frames shape (C T H W) {frames.shape}")
        label = self._get_label(activity)
        return frames, label , sample_id

    def _get_label(self, activity):
        # Assuming class names are the activity names
        # class_names = sorted(os.listdir(self.root_dir))
        class_names = self.class_names
        label = class_names.index(activity)
        return label


# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize((128, 171), interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 171), interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
])

depth_transforms = transforms.Compose([
    transforms.Resize((128, 171), interpolation=transforms.InterpolationMode.NEAREST),  # Resize to match RGB
    transforms.CenterCrop((112, 112)),  # Crop to the desired size
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.25])  # Normalize around the observed depth range
])
depth_tt_transforms = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.25])  # Normalize around the observed depth range
])
mvit_transform = transforms.Compose([
transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Converts to [0.0, 1.0]
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
])

swin_transform = transforms.Compose([
transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Converts to [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Create datasets and dataloaders
# train_dataset = Custom3DDataset(root_dir='/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/train',
#                                 transform=train_transforms)
# test_dataset = Custom3DDataset(root_dir='/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/test', transform=test_transforms)


class Custom4DDataset(Dataset):
    def __init__(self, rgb_root_dir, depth_root_dir, include_classes, cam_view, sequence_length=16, sampling="multi-uniform", transform=None ,depth_transform=depth_transforms, max_seq=3):
        if cam_view.split("_")[-1] == "1":
            self.rgb_root_dir = os.path.join(rgb_root_dir , "cam_1")
            self.depth_root_dir = os.path.join(depth_root_dir , "depth_1")
            # print(f'depth path: {self.depth_root_dir} \n rgb path: {self.rgb_root_dir}')
        if cam_view.split("_")[-1] == "2":
            self.rgb_root_dir = os.path.join(rgb_root_dir, "cam_2")
            self.depth_root_dir = os.path.join(depth_root_dir, "depth_2")
            # print(f'depth path: {self.depth_root_dir} \n rgb path: {self.rgb_root_dir}')
        self.sequence_length = sequence_length
        self.transform = transform
        self.depth_transform = depth_transform
        self.sampling = sampling
        self.number_of_seq = max_seq
        self.include_classes = include_classes
        all_classes = sorted(os.listdir(self.rgb_root_dir))
        try:
            all_classes.remove("nan")
        except Exception as e:
            pass

        if len(include_classes) > 0:
            self.classes = [cls for cls in all_classes if cls in include_classes]
        else:
            self.classes = os.listdir(all_classes)
        # print(f'number of all available classes {len(all_classes)} , used classes {len(self.classes)}')

        self.class_names = sorted(self.classes)
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        unique_id = -1
        for activity in self.classes:
            activity_dir = os.path.join(self.rgb_root_dir, activity)
            frames = sorted(glob(os.path.join(activity_dir, '*.png')))
            if len(frames) == 0:
                frames = sorted(glob(os.path.join(activity_dir, '*.jpg')))
            grouped_frames = self._group_frames_by_subject_and_session(frames)

            for subject_session in grouped_frames:
                unique_id += 1
                frames = grouped_frames[subject_session]
                sequences += self._sample_sequences(frames, activity, unique_id)
        # print(f"number of sequences {len(sequences)} , {len(sequences[-1][0])}")
        return sequences

    def _sample_sequences(self, frames, activity, unique_id):
        sequences = []
        if len(frames) < self.sequence_length:
            sequence = self._pad_sequence(frames)
            sequences.append((sequence, activity, unique_id))
        else:
            seq_counter = 0
            if self.sampling == "multiple-consecutive":
                for i in range(0, len(frames) - self.sequence_length + 1):
                    while seq_counter < self.number_of_seq:
                        sequence = frames[i:i + self.sequence_length]
                        sequences.append((sequence, activity, unique_id))
                        seq_counter += 1
            elif self.sampling == "multiple-random":
                for _ in range(0, len(frames) - self.sequence_length + 1):
                    while seq_counter < self.number_of_seq:
                        start_idx = random.randint(0, len(frames) - self.sequence_length)
                        sequence = frames[start_idx:start_idx + self.sequence_length]
                        sequences.append((sequence, activity, unique_id))
                        seq_counter += 1
            elif self.sampling == "single-random":
                while seq_counter < self.number_of_seq:
                    sequence = sorted(random.sample(frames, self.sequence_length))
                    sequences.append((sequence, activity, unique_id))
                    seq_counter += 1
            elif self.sampling == "multi-uniform":
                seq_counter = 0
                while seq_counter < self.number_of_seq:
                    step = max(len(frames) // self.sequence_length, 1)
                    offset = random.randint(0, step - 1) if step > 1 else 0
                    sequence = sorted(frames[i] for i in range(offset, len(frames), step)[:self.sequence_length])
                    sequences.append((sequence, activity, unique_id))
                    # print(unique_id, subject_session, activity)
                    # print(step , offset , "\n",sequence , unique_id)
                    seq_counter += 1
        # print(f"number of sequences created in 4D dataset {len(sequences)} \n one sequence length{len(sequence)}")
        return sequences

    def _pad_sequence(self, frames):
        if len(frames) == 0:
            return frames  # Avoid division by zero if frames is empty
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])  # Repeat the last frame
        return frames
    def _group_frames_by_subject_and_session(self, frames):
        grouped_frames = {}
        for frame in frames:
            filename = os.path.basename(frame)
            subject_session = '_'.join(filename.split('_')[:2])
            if subject_session not in grouped_frames:
                grouped_frames[subject_session] = []
            grouped_frames[subject_session].append(frame)
        return grouped_frames

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frames, activity, sample_id = self.sequences[idx]
        # print(f"len frames {len(frames)} , len sequences {len(self.sequences)}")
        rgbd = []
        for frame_path in frames:
            # Load RGB and depth frames
            image_rgb = Image.open(frame_path).convert('RGB')
            # print(f"image_rgb.size {image_rgb.size}")
            # print("RGB frame path",frame_path)
            frame_depth_path = frame_path.replace(self.rgb_root_dir, self.depth_root_dir)
            # print("depth frame path",frame_depth_path)
            image_depth = Image.open(frame_depth_path).convert('L')  # Convert to grayscale
            # print(f"image_depth.size {image_depth.size} , type {type(image_depth.size)}")

            if self.transform:
                image_rgb = self.transform(image_rgb)
                image_depth = self.depth_transform(image_depth)

            # Stack RGB and depth along channel dimension
            # print(f'shape rgb {image_rgb.shape} , depth {image_depth}')
            image_combined = torch.cat((image_rgb, image_depth), dim=0)
            # print(f'shape rgbd {image_combined.shape}')
            rgbd.append(image_combined)

        frames = torch.stack(rgbd)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # Change to (C, T, H, W)
        # print(f'frames after stack {frames.shape}')
        label = self.class_names.index(activity)
        return frames, label, sample_id


    def _get_label(self, activity):
        # Assuming class names are the activity names
        # class_names = sorted(os.listdir(self.root_dir))
        class_names = self.class_names
        label = class_names.index(activity)
        return label
