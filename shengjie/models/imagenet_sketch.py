"""
Create data loaders for ImageNet-Sketch dataset.
Number of classes: 1000
The dataset is split into validation and test sets.
"""

import os
import torch
import urllib.request
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_imagenet_sketch_data_loader(root,
                                    batch_size,
                                    num_workers=16,
                                    pin_memory=False,
                                    valid_size=0.2):
    """
    Utility function for loading and returning data loaders over the ImageNet-Sketch dataset.
    Splits the dataset into validation and test sets.

    Args:
        root (str): Root directory of the ImageNet-Sketch dataset.
        batch_size (int): How many samples per batch to load.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
        valid_size (float): Proportion of the dataset to include in the validation split.

    Returns:
        tuple: (val_loader, test_loader) - Data loaders for the validation and test sets.
    """

    # Define normalization (same as standard ImageNet)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset_root = os.path.join(root, 'ImageNet-Sketch/sketch')
    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(dataset_root, transform=transform)

    # Download and load imagenet_class_index.json
    url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    imagenet_class_index_file = os.path.join(dataset_root, 'imagenet_class_index.json')

    if not os.path.exists(imagenet_class_index_file):
        urllib.request.urlretrieve(url, imagenet_class_index_file)

    with open(imagenet_class_index_file, 'r') as f:
        class_idx = json.load(f)

    # Create new class_to_idx mapping
    imagenet_class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}

    # Get old class_to_idx mapping from the dataset
    old_class_to_idx = dataset.class_to_idx

    # Create a mapping from old indices to new indices
    old_to_new_idx = {}
    for class_name, old_idx in old_class_to_idx.items():
        if class_name in imagenet_class_to_idx:
            new_idx = imagenet_class_to_idx[class_name]
            old_to_new_idx[old_idx] = new_idx

    # Update dataset's class_to_idx to match ImageNet
    dataset.class_to_idx = imagenet_class_to_idx

    # Regenerate samples and targets
    new_samples = []
    missing_classes = set()

    for (path, target) in dataset.samples:
        if target in old_to_new_idx:
            new_target = old_to_new_idx[target]
            new_samples.append((path, new_target))
        else:
            missing_classes.add(dataset.classes[target])
            continue  # Skip sample if class not found

    if missing_classes:
        print(f"Warning: {len(missing_classes)} classes not found in ImageNet class index.")
        print(f"Missing classes: {missing_classes}")

    dataset.samples = new_samples
    dataset.targets = [s[1] for s in dataset.samples]

    # Split the dataset into validation and test sets
    val_size = int(valid_size * len(dataset))
    test_size = len(dataset) - val_size
    val_dataset, test_dataset = random_split(dataset, [val_size, test_size])

    # Create data loaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=pin_memory)

    return val_loader, test_loader
