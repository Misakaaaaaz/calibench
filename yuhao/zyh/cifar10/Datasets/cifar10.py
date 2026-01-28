import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False,
                           get_val_temp=0,
                           root='./data',
                           resize=None):  # 新增 resize 参数
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # --- 1. 定义 Resize 变换 ---
    # 如果指定了 resize，就创建一个 Resize 对象，否则为空列表
    resize_transform = [transforms.Resize((resize, resize))] if resize else []

    # --- 2. 定义 Validation Transforms ---
    # 逻辑：Resize (可选) -> ToTensor -> Normalize
    valid_transform = transforms.Compose(
        resize_transform + [
            transforms.ToTensor(),
            normalize,
        ]
    )

    # --- 3. 定义 Train Transforms ---
    if augment:
        # 逻辑：完全保留你原来的增强 (RandomCrop 32) -> Resize (可选) -> ToTensor -> Normalize
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + resize_transform + [
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # 逻辑：Resize (可选) -> ToTensor -> Normalize
        train_transform = transforms.Compose(
            resize_transform + [
                transforms.ToTensor(),
                normalize,
            ]
        )

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=root, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=root, train=True,
        download=False, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    # 处理额外的 temp validation set
    valid_temp_loader = None
    if get_val_temp > 0:
        valid_temp_dataset = datasets.CIFAR10(
            root=root, train=True,
            download=False, transform=valid_transform,
        )
        split = int(np.floor(get_val_temp * split))
        valid_idx, valid_temp_idx = valid_idx[split:], valid_idx[:split]
        valid_temp_sampler = SubsetRandomSampler(valid_temp_idx)
        valid_temp_loader = torch.utils.data.DataLoader(
            valid_temp_dataset, batch_size=batch_size, sampler=valid_temp_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    if get_val_temp > 0:
        return (train_loader, valid_loader, valid_temp_loader)
    else:
        return (train_loader, valid_loader)


def get_test_loader(batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    root='./data',
                    resize=None):  # 新增 resize 参数
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # --- 定义 Resize 变换 ---
    resize_transform = [transforms.Resize((resize, resize))] if resize else []

    # define transform
    transform = transforms.Compose(
        resize_transform + [
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.CIFAR10(
        root=root, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader