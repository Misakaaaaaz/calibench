import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

# Mapping of corruption types to their categories
corruption_categories = {
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'digital': ['contrast', 'elastic_transform', 'jpeg_compression', 'pixelate'],
    'extra': ['gaussian_blur', 'saturate', 'spatter', 'speckle_noise'],
    'noise': ['gaussian_noise', 'impulse_noise', 'shot_noise'],
    'weather': ['brightness', 'fog', 'frost', 'snow'],
}

# Create a mapping from corruption types to categories
corruption_type_to_category = {ctype: category for category, types in corruption_categories.items() for ctype in types}

def get_imagenet_c_data_loader(root, batch_size, corruption_type='defocus_blur', severity=1, valid_size=0.2, num_workers=16, pin_memory=False, random_seed=None, mean=None, std=None, image_size=224):
    """
    Utility function for loading and returning train and validation data loaders over the ImageNet-C dataset.
    Each severity level is split into validation and test sets.

    Args:
        root (str): Root directory of the ImageNet-C dataset.
        batch_size (int): How many samples per batch to load.
        corruption_type (str): Corruption type to include.
        severity (int): Severity level to include (1 to 5).
        valid_size (float): Proportion of the dataset to use as validation set (default: 0.2).
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
        random_seed (int, optional): Random seed for reproducible data splits.
        mean (list, optional): Custom normalization mean (default: [0.485, 0.456, 0.406])
        std (list, optional): Custom normalization std (default: [0.229, 0.224, 0.225])
        image_size (int): Target image size for center crop (default: 224)

    Returns:
        tuple: (validation_loader, test_loader)
    """
    # Use custom normalization if provided, otherwise use standard ImageNet
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)

    # Resize to a bit larger than image_size, then crop to exact size
    resize_size = int(image_size * 256 / 224)

    # Define transformation
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Get the corruption category
    if corruption_type not in corruption_type_to_category:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    corruption_category = corruption_type_to_category[corruption_type]

    data_dir = os.path.join(root, corruption_category, corruption_type, str(severity))
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # Create dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Get the number of samples
    num_samples = len(dataset)
    indices = list(range(num_samples))

    # Calculate the split
    split = int(np.floor(valid_size * num_samples))

    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle the indices
    np.random.shuffle(indices)

    # Split the indices
    val_indices, test_indices = indices[:split], indices[split:]

    # Create samplers
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create data loaders
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return val_loader, test_loader