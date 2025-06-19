import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np


def default_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])


class VideoClipDataset(Dataset):
    """
    Base dataset class for FAU/Emotion datasets using folders of frame sequences.
    """

    def __init__(self, csv_path, root_dir, clip_length=16, transform=None, label_type="fau"):
        """
        Args:
            csv_path (str): Path to CSV file with columns: [video_id, start_frame, label_1, ..., label_n]
            root_dir (str): Root folder containing video frame folders.
            clip_length (int): Number of frames per clip.
            transform (callable): Transformations applied to each frame.
            label_type (str): "fau" or "emotion"
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform or default_transform()
        self.label_type = label_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row["video_id"]
        start_frame = int(row["start_frame"])
        label = row[2:].astype(np.float32).values  # vector of FAU or emotion labels

        # Load frames as a clip
        frames = []
        frame_folder = os.path.join(self.root_dir, video_id)
        for i in range(start_frame, start_frame + self.clip_length):
            frame_path = os.path.join(frame_folder, f"{i:05d}.jpg")
            img = Image.open(frame_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)    # (C, T, H, W) if needed, or keep as (T, C, H, W)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return clip, label_tensor


# ======================
# Dataset Wrappers
# ======================

def get_dataset(name, split_csv, root_dir, clip_length=16, label_type="fau"):
    """
    Factory to load the appropriate dataset.
    Args:
        name (str): One of 'DISFA', 'BP4D', 'BP4D+', 'Aff-Wild2', 'RAF-DB'
        split_csv (str): Path to CSV with video_id, start_frame, labels
        root_dir (str): Path to video frame folders
        clip_length (int): Number of frames per clip
        label_type (str): 'fau' or 'emotion'
    Returns:
        torch Dataset
    """
    name = name.lower()
    if name in ["disfa", "bp4d", "bp4d+", "aff-wild2", "raf-db"]:
        return VideoClipDataset(csv_path=split_csv,
                                root_dir=root_dir,
                                clip_length=clip_length,
                                label_type=label_type)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
