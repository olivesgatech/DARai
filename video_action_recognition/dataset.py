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
    def __init__(self, root_dir, include_classes, sequence_length=16, sampling="single-random", transform=None , max_seq=3 ):

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
                    elif self.sampling == "multiple-random":
                        seq_counter = 0
                        for _ in range(0, len(frames) - self.sequence_length + 1):
                            while seq_counter < self.number_of_seq :
                                start_idx = random.randint(0, len(frames) - self.sequence_length)
                                sequence = frames[start_idx:start_idx + self.sequence_length]
                                sequences.append((sequence, activity , unique_id))
                                seq_counter += 1
                    elif self.sampling == "single-random":
                        seq_counter = 0
                        while seq_counter < self.number_of_seq:
                            sequence = sorted(random.sample(frames, self.sequence_length))
                            sequences.append((sequence, activity , unique_id))
                            # print(unique_id, subject_session, activity)
                            seq_counter += 1
                # print(f'Grouped sequence {subject_session} {activity}: {sequence}')  # Print the grouped sequence
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
        # print(f"frames shape {frames.shape}")
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

