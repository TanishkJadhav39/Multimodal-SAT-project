import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class S1GFloodDataset(Dataset):

    def __init__(self, root_dir, files):

        self.root_dir = root_dir

        self.a_dir = os.path.join(root_dir, "A")
        self.b_dir = os.path.join(root_dir, "B")
        self.label_dir = os.path.join(root_dir, "Label")

        # files already contains filenames such as image_1.png
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        # File paths
        a_path = os.path.join(
            self.a_dir,
            filename
        )

        b_path = os.path.join(
            self.b_dir,
            filename
        )

        label_path = os.path.join(
            self.label_dir,
            filename
        )

        # Load before-flood SAR
        before = np.array(
            Image.open(a_path).convert("L"),
            dtype=np.float32
        )

        # Load after-flood SAR
        after = np.array(
            Image.open(b_path).convert("L"),
            dtype=np.float32
        )

        # Load ground-truth flood mask
        label = np.array(
            Image.open(label_path).convert("L"),
            dtype=np.float32
        )

        # Normalize SAR images
        before = before / 255.0
        after = after / 255.0

        # Convert mask to binary
        label = (label > 0).astype(np.float32)

        # Convert NumPy → PyTorch
        before = torch.from_numpy(before).unsqueeze(0)
        after = torch.from_numpy(after).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)

        return before, after, label