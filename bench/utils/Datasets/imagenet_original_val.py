"""
Create train, val, test iterators for ImageNet.
Train set size: 1,281,167
Val set size: 50,000
After splitting:
- Val set size: 10,000
- Test set size: 40,000
Number of classes: 1000
"""

import os
import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import json

EXTENSION = 'JPEG'
NUM_CLASSES = 1000



def get_data_loader(root='/share/datasets/',
                    batch_size=128,
                    split='train',
                    shuffle=False,
                    num_workers=16,
                    pin_memory=True,
                    valid_size=0.2,
                    random_seed=42,
                    mean=None,
                    std=None,
                    image_size=224):
    """
    Utility function for loading and returning train, val, and test
    iterators over the ImageNet dataset.

    Args:
        mean (list, optional): Custom normalization mean (default: [0.485, 0.456, 0.406])
        std (list, optional): Custom normalization std (default: [0.229, 0.224, 0.225])
        image_size (int): Target image size for center crop (default: 224)
    """

    root = os.path.join(root, 'imagenet')

    # Use custom normalization if provided, otherwise use standard ImageNet
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)

    # Resize to a bit larger than image_size, then crop to exact size
    resize_size = int(image_size * 256 / 224)

    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        data_dir = os.path.join(root, 'train')
    else:
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])
        data_dir = os.path.join(root, 'val')

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # 下载并加载 imagenet_class_index.json
    url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    imagenet_class_index_file = os.path.join(root, 'imagenet_class_index.json')

    if not os.path.exists(imagenet_class_index_file):
        urllib.request.urlretrieve(url, imagenet_class_index_file)

    with open(imagenet_class_index_file, 'r') as f:
        class_idx = json.load(f)

    # 创建新的 class_to_idx 映射
    class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}

    # 获取旧的索引到类别名称的映射
    idx_to_classname = {idx: class_name for class_name, idx in dataset.class_to_idx.items()}

    # 更新数据集的 class_to_idx
    dataset.class_to_idx = class_to_idx

    # 重新生成 samples 和 targets
    new_samples = []
    missing_classes = set()
    for (p, t) in dataset.samples:
        class_name = idx_to_classname[t]  # 根据旧的索引获取类别名称
        if class_name in dataset.class_to_idx:
            new_t = dataset.class_to_idx[class_name]  # 根据类别名称获取新的索引
            new_samples.append((p, new_t))
        else:
            missing_classes.add(class_name)
            continue  # 如果类别名称不在新的映射中，跳过该样本

    if missing_classes:
        print(f"Warning: {len(missing_classes)} classes not found in new class_to_idx mapping.")
        print(f"Missing classes: {missing_classes}")

    dataset.samples = new_samples
    dataset.targets = [s[1] for s in dataset.samples]

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)

    return data_loader
