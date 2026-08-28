import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """Native PyTorch Dataset for image classification with Albumentations."""

    def __init__(self, image_paths, targets=None, resize=None, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        # Convert (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        data = {
            "image": torch.tensor(image, dtype=torch.float),
        }

        if self.targets is not None:
            data["targets"] = torch.tensor(self.targets[item], dtype=torch.float)

        return data
