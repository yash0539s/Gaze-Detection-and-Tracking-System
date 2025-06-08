import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch
from feature_extraction import preprocess_eye_image  # import

class MPIIGazeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform

        gaze_path = os.path.join(root_dir, "AnnotationSubset", "gaze_angle.txt")
        self.gaze_angles = pd.read_csv(gaze_path, sep=' ', header=None).values

        sample_list_path = os.path.join(root_dir, "sample_list_for_eye_image", "sample_list.txt")
        with open(sample_list_path, 'r') as f:
            lines = f.readlines()

        total_samples = len(lines)
        split_idx = int(total_samples * 0.8)

        if split == 'train':
            selected_lines = lines[:split_idx]
            self.gaze_angles = self.gaze_angles[:split_idx]
        else:
            selected_lines = lines[split_idx:]
            self.gaze_angles = self.gaze_angles[split_idx:]

        self.img_paths = [os.path.join(root_dir, "Data", "Normalized", line.strip()) for line in selected_lines]

        valid_data = [(p, g) for p, g in zip(self.img_paths, self.gaze_angles) if os.path.isfile(p)]
        self.img_paths, self.gaze_angles = zip(*valid_data)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Use feature_extraction preprocessing for consistency
        img = preprocess_eye_image(img)

        if self.transform:
            img = self.transform(img)

        img = torch.tensor(img).unsqueeze(0)  # channel dimension
        label = torch.tensor(self.gaze_angles[idx], dtype=torch.float32)
        return img, label
