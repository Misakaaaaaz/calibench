import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class IWildCamDataset(Dataset):
    """
    iWildCam dataset loader that handles metadata.csv and remaps category IDs.

    Args:
        root: Path to dataset root directory (should contain metadata.csv and train/ folder)
        split: One of 'train', 'val', 'test', 'id_val', 'id_test'
        transform: Torchvision transforms to apply
    """
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Load metadata
        metadata_path = os.path.join(root, 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        # Read metadata and filter by split
        metadata = pd.read_csv(metadata_path)
        self.data = metadata[metadata['split'] == split].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(f"No data found for split '{split}'")

        # Create category ID mapping from non-contiguous to contiguous (0-205)
        unique_categories = sorted(metadata['category_id'].unique())
        self.category_mapping = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
        self.num_classes = len(unique_categories)

        # Image directory
        self.image_dir = os.path.join(root, 'train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        # Get label and remap to contiguous range
        original_label = row['category_id']
        label = self.category_mapping[original_label]

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_data_loader(root,
                    split='train',
                    batch_size=128,
                    shuffle=True,
                    num_workers=16,
                    pin_memory=True,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    image_size=224):
    """
    Returns a DataLoader for iWildCam dataset.

    Args:
        root: Path to dataset root directory
        split: One of 'train', 'val', 'test', 'id_val', 'id_test'
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for CUDA
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)
        image_size: Target image size (224, 336, or 448)

    Returns:
        DataLoader instance
    """
    # Calculate resize dimensions (following ImageNet pattern)
    resize_size = int(image_size * 256 / 224)

    # Define transforms
    normalize = transforms.Normalize(mean=mean, std=std)

    if split == 'train':
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])

    # Create dataset
    dataset = IWildCamDataset(root=root, split=split, transform=transform)

    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return data_loader


def get_train_loader(root,
                     batch_size=128,
                     shuffle=True,
                     num_workers=16,
                     pin_memory=True,
                     mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225],
                     image_size=224,
                     **kwargs):
    """Returns training data loader."""
    return get_data_loader(
        root=root,
        split='train',
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        mean=mean,
        std=std,
        image_size=image_size
    )


def get_val_loader(root,
                   batch_size=128,
                   shuffle=False,
                   num_workers=16,
                   pin_memory=True,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   image_size=224,
                   **kwargs):
    """Returns validation data loader."""
    return get_data_loader(
        root=root,
        split='val',
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        mean=mean,
        std=std,
        image_size=image_size
    )


def get_test_loader(root,
                    batch_size=128,
                    shuffle=False,
                    num_workers=16,
                    pin_memory=True,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    image_size=224,
                    **kwargs):
    """Returns test data loader."""
    return get_data_loader(
        root=root,
        split='test',
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        mean=mean,
        std=std,
        image_size=image_size
    )
