"""
Create data loader for ImageNet-LT dataset.
Number of classes: 1000 (with long-tailed distribution)
Use the test set as the evaluation set.
"""

import os
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split

def get_imagenet_lt_data_loader(root,
                                batch_size=128,
                                num_workers=4,
                                pin_memory=False,
                                valid_size=0.2,
                                random_seed=None,
                                mean=None,
                                std=None,
                                image_size=224):
    """
    Utility function for loading and returning data loaders over the ImageNet-LT dataset.
    Uses 20% of the test set as validation and 80% as test set.

    Args:
        root (str): Root directory of ImageNet-LT.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        pin_memory (bool): Whether to use pinned memory.
        valid_size (float): Proportion of the dataset to use as validation set (default: 0.2).
        random_seed (int, optional): Random seed for reproducible data splits.
        mean (list, optional): Custom normalization mean (default: [0.485, 0.456, 0.406])
        std (list, optional): Custom normalization std (default: [0.229, 0.224, 0.225])
        image_size (int): Target image size for center crop (default: 224)

    Returns:
        tuple: (val_loader, test_loader) - Data loaders for the validation and test sets.
    """

    # Use custom normalization if provided, otherwise use standard ImageNet
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)

    # Resize to a bit larger than image_size, then crop to exact size
    resize_size = int(image_size * 256 / 224)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])

    # Paths
    data_dir = os.path.join(root, 'torch_image_folder', 'mnt', 'volume_sfo3_01', 'imagenet-lt', 'ImageDataset', 'test')
    txt_file = os.path.join(root, 'ImageNet_LT_test.txt')

    # Build label to WNID mapping from txt file
    label_to_wnid = {}
    with open(txt_file, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            label = int(label)
            wnid = path.split('/')[1]  # Extract WNID from path (e.g., 'val/n01440764/...')
            if label not in label_to_wnid:
                label_to_wnid[label] = wnid

    # Load ImageNet class index mapping
    import urllib.request
    import json
    url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    imagenet_class_index_file = os.path.join(root, 'imagenet_class_index.json')
    if not os.path.exists(imagenet_class_index_file):
        urllib.request.urlretrieve(url, imagenet_class_index_file)
    with open(imagenet_class_index_file, 'r') as f:
        class_idx = json.load(f)
    wnid_to_class_idx = {value[0]: int(key) for key, value in class_idx.items()}

    # Create mapping from label (0-999) to standard ImageNet class index
    label_to_class_idx = {}
    for label, wnid in label_to_wnid.items():
        if wnid in wnid_to_class_idx:
            label_to_class_idx[label] = wnid_to_class_idx[wnid]
        else:
            print(f"WNID {wnid} not found in ImageNet class index.")

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Get old class_to_idx mapping from the dataset (should be labels 0-999)
    old_class_to_idx = dataset.class_to_idx  # Mapping from directory name to label assigned by ImageFolder

    # Create mapping from old labels to new labels (standard ImageNet class indices)
    old_idx_to_class_idx = {}
    for class_name, old_label in old_class_to_idx.items():
        label = int(class_name)  # Directory names are '0', '1', ..., '999'
        if label in label_to_class_idx:
            new_label = label_to_class_idx[label]
            old_idx_to_class_idx[old_label] = new_label
        else:
            print(f"Label {label} not found in label_to_class_idx mapping.")

    # Update labels in dataset
    new_samples = []
    for path, target in dataset.samples:
        if target in old_idx_to_class_idx:
            new_target = old_idx_to_class_idx[target]
            new_samples.append((path, new_target))
        else:
            continue  # Skip if label mapping not found

    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]

    # Split the dataset into validation and test sets
    val_size = int(valid_size * len(dataset))
    test_size = len(dataset) - val_size
    
    # Set seed for reproducibility if provided
    if random_seed is not None:
        generator = torch.Generator().manual_seed(random_seed)
        val_dataset, test_dataset = random_split(dataset, [val_size, test_size], generator=generator)
    else:
        val_dataset, test_dataset = random_split(dataset, [val_size, test_size])

    # Create data loaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return val_loader, test_loader
