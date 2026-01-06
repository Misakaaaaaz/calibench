"""
Create data loader for ImageNet-LT dataset.
Number of classes: 1000 (with long-tailed distribution)
Use the test set as the evaluation set.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
from typing import Dict, Optional, Union
from Component.metrics import *
import logging

import json

from Component.model.bbq import BBQCalibrator
from Component.model.cts import CTSCalibrator
from Component.model.ets import ETSCalibrator
from Component.model.feature_clipping import FeatureClippingCalibrator
from Component.model.group_calibration import GroupCalibrationCalibrator
from Component.model.histogram_binning import HistogramBinningCalibrator
from Component.model.procal import ProCalDensityRatioCalibrator
from Component.model.pts import PTSCalibrator
from Component.model.temperature_scaling import TemperatureScalingCalibrator
from Component.model.vector_scaling import VectorScalingCalibrator
from zyh.smart_calibrator import SMART
from models.resnet_imagenet import ResNet_ImageNet as own_resnet50, ResNet101_ImageNet as own_resnet101, \
    ResNet152_ImageNet as own_resnet152
from models.densenet_imagenet import DenseNet121_ImageNet as own_densenet121, DenseNet169_ImageNet as own_densenet169
from models.wide_resnet_imagenet import WideResNet_ImageNet as own_wide_resnet
from models.vit_imagenet import ViT_B_16_ImageNet as own_vit_b_16, ViT_B_32_ImageNet as own_vit_b_32, \
    ViT_L_16_ImageNet as own_vit_l_16, ViT_L_32_ImageNet as own_vit_l_32
from models.swin_imagenet import Swin_B_ImageNet as own_swin_b

from models.beit_imagenet import BEiT_Base_ImageNet as own_beit_base, BEiT_Large_ImageNet as own_beit_large, \
    BEiTv2_Base_ImageNet as own_beitv2_base
from models.convnext_imagenet import ConvNext_Tiny_ImageNet as own_convnext_tiny, \
    ConvNext_Base_ImageNet as own_convnext_base, ConvNext_Large_ImageNet as own_convnext_large
# EVA02 with ImageNet-1K fine-tuned classification heads (uses 448x448 input)
from models.eva_imagenet import EVA02_Base_ImageNet as own_eva02_base, EVA02_Large_ImageNet as own_eva02_large, \
    EVA02_Small_ImageNet as own_eva02_small
from models.mobilenet_v2_imagenet import MobileNet_V2_ImageNet as own_mobilenet_v2
from models.mlp_mixer_b16_imagenet import MLPMixer_B16_ImageNet as own_mlp_mixer_b16

dataset_name = 'imagenet-lt'
smart_loss = 'smooth_soft_ece'
corruption_type = None
severity = None
train_loss = None
RUN_METHODS = ["uncalibrated", "TS", "PTS", "CTS", "ETS", "SMART", "HB", "BBQ", "VS", "GC", "ProCal_DR", "FC", "Spline"]
METRICS = ['ece', 'accuracy', 'adaece', 'cece', 'nll', 'ece_debiased', 'ece_sweep', 'Brier', 'rbs']
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
pretrained_model_list = {
    'resnet50': lambda: own_resnet50(pretrained=True),
    'resnet101': lambda: own_resnet101(pretrained=True),
    'resnet152': lambda: own_resnet152(pretrained=True),
    'densenet121': lambda: own_densenet121(pretrained=True),
    'densenet169': lambda: own_densenet169(pretrained=True),
    'wide_resnet': lambda: own_wide_resnet(pretrained=True),
    'mobilenet_v2': lambda: own_mobilenet_v2(pretrained=True),
    'vit_l_16': lambda: own_vit_l_16(pretrained=True),
    'vit_b_16': lambda: own_vit_b_16(pretrained=True),
    'vit_b_32': lambda: own_vit_b_32(pretrained=True),
    'vit_l_32': lambda: own_vit_l_32(pretrained=True),
    'swin_b': lambda: own_swin_b(pretrained=True),
    'beit_base': lambda: own_beit_base(pretrained=True),
    'beit_large': lambda: own_beit_large(pretrained=True),
    'beitv2_base': lambda: own_beitv2_base(pretrained=True),
    'convnext_tiny': lambda: own_convnext_tiny(pretrained=True),
    'convnext_base': lambda: own_convnext_base(pretrained=True),
    'convnext_large': lambda:own_convnext_large(pretrained=True),
    'eva02_small': lambda: own_eva02_small(pretrained=True),
    'eva02_base': lambda: own_eva02_base(pretrained=True),
    'eva02_large': lambda: own_eva02_large(pretrained=True),
    'mlp_mixer_b16': lambda: own_mlp_mixer_b16(pretrained=True)
}


def get_pretrained_model(name):
    return pretrained_model_list[name]().to(device)


class ResNet_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet_ImageNet, self).__init__()
        self.model = torchvision.models.resnet50(**kwargs)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.fc(x)


def compute_all_metrics(
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        n_bins: int = 15
) -> Dict[str, float]:
    """
    Compute all available metrics for the given logits/probs and labels.

    Args:
        labels (torch.Tensor): Target labels
        logits (torch.Tensor, optional): Input logits before softmax
        probs (torch.Tensor, optional): Probability distributions (softmax outputs)
        n_bins (int, optional): Number of bins for ECE calculation. Defaults to 15.

    Returns:
        Dict[str, float]: Dictionary containing all metric values
    """
    if logits is None and probs is None:
        raise ValueError("Either logits or probs must be provided")

    device = labels.device

    # If probs not provided, compute from logits
    if probs is None:
        probs = F.softmax(logits, dim=1)

    # Initialize metrics
    metrics = {
        'ece': ECE(n_bins=n_bins),
        'accuracy': Accuracy(),
        'adaptive_ece': AdaptiveECE(n_bins=n_bins),
        'classwise_ece': ClasswiseECE(n_bins=n_bins),
        'nll': NLL(),
        'ece_debiased': ECEDebiased(n_bins=n_bins),
        'ece_sweep': ECESweep(),
        'Brier': BrierLoss(),
        'rbs': RBS()
    }

    # Set up basic logging
    logger = logging.getLogger(__name__)

    results = {}
    for name, metric in metrics.items():
        metric = metric.to(device)
        try:
            if name in ['nll', 'rbs', 'Brier']:
                if probs is not None:
                    value = metric(softmaxes=probs, labels=labels)
                elif logits is not None:
                    value = metric(logits=logits, labels=labels)
            elif name in ['ece', 'adaptive_ece', 'classwise_ece', 'ece_debiased', 'ece_sweep', 'accuracy']:
                value = metric(softmaxes=probs, labels=labels)
            else:
                logger.warning(f"Unknown metric type: {name}")
                continue

            # Convert to float if it's a tensor
            if torch.is_tensor(value):
                value = value.item()
            results[name] = value
        except Exception as e:
            logger.warning(f"Failed to compute {name}: {str(e)}")
            results[name] = None
            continue

    return results


def get_all_metrics(
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        n_bins: int = 15
) -> Dict[str, float]:
    """
    Get all metrics in a dictionary format compatible with the standard results structure.

    Args:
        labels (torch.Tensor): Target labels
        logits (torch.Tensor, optional): Input logits before softmax
        probs (torch.Tensor, optional): Probability distributions (softmax outputs)
        n_bins (int, optional): Number of bins for ECE calculation. Defaults to 15.

    Returns:
        Dict[str, float]: Dictionary containing the 8 standard metrics:
        {
            'ece': float,
            'accuracy': float,
            'adaece': float,
            'cece': float,
            'nll': float,
            'ece_debiased': float,
            'ece_sweep': float,
            'rbs': float
        }
    """
    # Move tensors to device
    logits = logits.to(device)
    labels = labels.to(device)

    # Determine if logits are actually probabilities
    is_probs = (logits.dim() == 2 and
                torch.allclose(logits.sum(dim=1), torch.ones(logits.size(0), device=device), atol=1e-3))
    metrics = compute_all_metrics(labels=labels, logits=logits if not is_probs else None,
                                  probs=logits if is_probs else None, n_bins=n_bins)
    return {
        'ece': metrics.get('ece', None),
        'accuracy': metrics.get('accuracy', None),
        'adaece': metrics.get('adaptive_ece', None),
        'cece': metrics.get('classwise_ece', None),
        'nll': metrics.get('nll', None),
        'ece_debiased': metrics.get('ece_debiased', None),
        'ece_sweep': metrics.get('ece_sweep', None),
        'Brier': metrics.get('Brier', None),
        'rbs': metrics.get('rbs', None)
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bootstrap_val_by_global_valid_size(
    features, logits, labels,
    valid_size, pool_valid_size=0.2
):
    if not (0 < valid_size <= 1.0):
        raise ValueError(f"valid_size must be in (0, 1], got {valid_size}")
    if valid_size > pool_valid_size:
        raise ValueError(f"valid_size {valid_size} > pool_valid_size {pool_valid_size}")

    if valid_size == pool_valid_size:
        return features, logits, labels

    N = features.shape[0]
    if logits.shape[0] != N or labels.shape[0] != N:
        raise ValueError("features / logits / labels size mismatch")
    k = int(round(N * (valid_size / pool_valid_size)))
    k = max(k, 1)

    idx = torch.randint(0, N, size=(k,), device=features.device)
    return features[idx], logits[idx], labels[idx]


def get_model_normalization(model_name):
    """
    Get the correct normalization parameters for each model.

    Different pretrained models are trained with different normalization:
    - Standard ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - BEiT-style: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    - CLIP-style (EVA02): mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]

    Returns:
        tuple: (mean, std) for normalization
    """
    # BEiT-style normalization (mean=0.5, std=0.5)
    if model_name in ['beit_base', 'beit_large', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
        return ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # CLIP-style normalization (EVA02)
    elif model_name in ['eva02_base', 'eva02_large', 'eva02_small']:
        return ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

    # Standard ImageNet normalization (default)
    # ResNet, BEiTv2, Swin, ConvNext, DenseNet, MobileNet, WideResNet, etc.
    else:
        return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_logit_paths(dataset_name, model_name, seed_value, valid_size=0.2, loss_fn='CE', corruption_type=None,
                    severity=None, train_loss=None):
    """
    Generate file paths for saving/loading logits with specific parameters

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        seed_value: Random seed used
        valid_size: Validation set size
        loss_fn: Loss function used for calibration (only affects smart model path)
        corruption_type: Type of corruption (for ImageNet-C)
        severity: Severity level (for ImageNet-C)
        train_loss: Training loss type (for CIFAR models)

    Returns:
        Dictionary of file paths for val/test logits and labels
    """
    # Create configuration-specific directory
    if dataset_name.startswith('imagenet'):
        config_dir = f"{dataset_name}_{model_name}_seed{seed_value}_vs{valid_size}"
    elif dataset_name.startswith('cifar'):
        # Always include train_loss for CIFAR datasets
        # Default to cross_entropy if not provided
        train_loss = train_loss or 'cross_entropy'
        config_dir = f"{dataset_name}_{model_name}_{train_loss}_seed{seed_value}"
    else:
        config_dir = f"{dataset_name}_{model_name}_seed{seed_value}"

    # For ImageNet-C, add corruption type and severity to the directory name
    if dataset_name == 'imagenet_c' and corruption_type is not None and severity is not None:
        config_dir = f"{dataset_name}_{corruption_type}_s{severity}_{model_name}_seed{seed_value}_vs{valid_size}"

    # Try multiple cache directories (search order: LogitsGap/cache, then cache)
    # Get the project root directory (parent of utils directory)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)

    base_dirs = [
        os.path.join(project_root, "LogitsGap", "cache"),
        os.path.join(project_root, "cache"),
        "LogitsGap/cache",
        "cache"
    ]
    cache_dir = None

    for base_dir in base_dirs:
        potential_cache_dir = os.path.join(base_dir, config_dir)
        # Check if this directory exists and has the required files
        if os.path.exists(potential_cache_dir):
            cache_dir = potential_cache_dir
            break

    # If no existing cache found, use the default cache directory
    if cache_dir is None:
        cache_dir = os.path.join("cache", config_dir)
        os.makedirs(cache_dir, exist_ok=True)

    paths = {
        'val_logits': os.path.join(cache_dir, "val_logits.npy"),
        'val_labels': os.path.join(cache_dir, "val_labels.npy"),
        'val_features': os.path.join(cache_dir, "val_features.npy"),
        'test_logits': os.path.join(cache_dir, "test_logits.npy"),
        'test_labels': os.path.join(cache_dir, "test_labels.npy"),
        'test_features': os.path.join(cache_dir, "test_features.npy"),
        'logitsgap_values': os.path.join(cache_dir, "logitsgap_values.json"),
        'test_logitsgap_values': os.path.join(cache_dir, "test_logitsgap_values.json"),
        'smart_model': os.path.join(cache_dir, f"smart_model_{loss_fn}.pth")
    }

    return paths


def logits_exist(dataset_name, model_name, seed_value, valid_size=0.2, loss_fn='CE', corruption_type=None,
                 severity=None, train_loss=None):
    """
    Check if logits already exist for the given parameters

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        seed_value: Random seed used
        valid_size: Validation set size
        loss_fn: Loss function used for calibration (not relevant for logits)
        corruption_type: Type of corruption (for ImageNet-C)
        severity: Severity level (for ImageNet-C)
        train_loss: Training loss type (for CIFAR models)

    Returns:
        Boolean indicating whether all required files exist
    """
    # Get paths
    paths = get_logit_paths(dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity,
                            train_loss)

    # Check if files exist
    files_exist = (os.path.exists(paths['val_logits']) and
                   os.path.exists(paths['val_labels']) and
                   os.path.exists(paths['val_features']) and
                   os.path.exists(paths['test_logits']) and
                   os.path.exists(paths['test_labels']) and
                   os.path.exists(paths['test_features']))

    return files_exist


def load_logits(dataset_name, model_name, seed_value, valid_size=0.2, loss_fn='CE', corruption_type=None, severity=None,
                train_loss=None):
    """
    Load logits, labels, and features for the given parameters

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        seed_value: Random seed used
        valid_size: Validation set size
        loss_fn: Loss function used for calibration (not relevant for logits)
        corruption_type: Type of corruption (for ImageNet-C)
        severity: Severity level (for ImageNet-C)
        train_loss: Training loss type (for CIFAR models)

    Returns:
        Tuple of (val_logits, val_labels, test_logits, test_labels, val_features, test_features)
    """
    # Get paths
    paths = get_logit_paths(dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity,
                            train_loss)

    # Check if files exist
    files_exist = (os.path.exists(paths['val_logits']) and
                   os.path.exists(paths['val_labels']) and
                   os.path.exists(paths['val_features']) and
                   os.path.exists(paths['test_logits']) and
                   os.path.exists(paths['test_labels']) and
                   os.path.exists(paths['test_features']))

    if not files_exist:
        dataset_info = f"{dataset_name}"
        if dataset_name == 'imagenet_c' and corruption_type is not None and severity is not None:
            dataset_info = f"{dataset_name} (corruption: {corruption_type}, severity: {severity})"
        raise FileNotFoundError(
            f"Logits not found for {dataset_info}, {model_name}, seed {seed_value}, valid_size {valid_size}")

    # Load the logits, labels, and features
    val_logits = np.load(paths['val_logits'])
    val_labels = np.load(paths['val_labels'])
    val_features = np.load(paths['val_features'])
    test_logits = np.load(paths['test_logits'])
    test_labels = np.load(paths['test_labels'])
    test_features = np.load(paths['test_features'])

    dataset_info = f"{dataset_name}"
    if dataset_name == 'imagenet_c' and corruption_type is not None and severity is not None:
        dataset_info = f"{dataset_name} (corruption: {corruption_type}, severity: {severity})"
    print(f"Loaded logits and features for {dataset_info}, {model_name}, seed {seed_value}, valid_size {valid_size}")

    return val_logits, val_labels, test_logits, test_labels, val_features, test_features


def save_logits(val_logits, val_labels, test_logits, test_labels, val_features, test_features,
                dataset_name, model_name, seed_value, valid_size=0.2, loss_fn='CE', corruption_type=None, severity=None,
                train_loss=None):
    """
    Save logits, labels, and features with parameter-specific filenames

    Args:
        val_logits: Validation set logits
        val_labels: Validation set labels
        test_logits: Test set logits
        test_labels: Test set labels
        val_features: Validation set features
        test_features: Test set features
        dataset_name: Name of the dataset
        model_name: Name of the model
        seed_value: Random seed used
        valid_size: Validation set size
        loss_fn: Loss function used for calibration
        corruption_type: Type of corruption (for ImageNet-C)
        severity: Severity level (for ImageNet-C)
        train_loss: Training loss type (for CIFAR models)
    """
    paths = get_logit_paths(dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity,
                            train_loss)

    # Save the logits, labels, and features
    np.save(paths['val_logits'], val_logits)
    np.save(paths['val_labels'], val_labels)
    np.save(paths['val_features'], val_features)
    np.save(paths['test_logits'], test_logits)
    np.save(paths['test_labels'], test_labels)
    np.save(paths['test_features'], test_features)

    dataset_info = f"{dataset_name}"
    if dataset_name == 'imagenet_c' and corruption_type is not None and severity is not None:
        dataset_info = f"{dataset_name} (corruption: {corruption_type}, severity: {severity})"
    print(f"Saved logits and features for {dataset_info}, {model_name}, seed {seed_value}, valid_size {valid_size}")


def get_model_input_size(model_name):
    """
    Get the correct input image size for each model.

    Most models use 224x224, but some require different sizes:
    - EVA02-Small: 336x336
    - EVA02-Base: 448x448
    - EVA02-Large: 448x448

    Returns:
        int: Input image size (height/width, assumes square images)
    """
    if model_name in ['eva02_small']:
        return 336
    elif model_name in ['eva02_base', 'eva02_large']:
        return 448
    else:
        return 224  # Default for most models


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


def store_method_results(overall_results, method_key, all_metrics, bins_list=None, loss_fn=None,
                         additional_params=None):
    """
    Store method results in the overall_results dictionary

    Args:
        overall_results: dict - the overall results dictionary
        method_key: str - key for storing results (e.g., 'TS_CE', 'SMART_soft_ece')
        all_metrics: dict - computed metrics
        bins_list: list - list of bin sizes
        loss_fn: str - loss function used
        additional_params: dict - additional parameters to store
    """
    if bins_list is None:
        bins_list = [15]

    method_results = {
        'acc': float(all_metrics['accuracy']),
        'nll': float(all_metrics['nll']),
        'loss_fn': loss_fn or 'none'
    }

    # Add additional parameters
    if additional_params:
        method_results.update(additional_params)

    # Add non-bin specific metrics
    if 'ece_sweep' in all_metrics:
        method_results['ece_sweep'] = float(all_metrics['ece_sweep'])
    if 'kde_ece' in all_metrics:
        method_results['kde_ece'] = float(all_metrics['kde_ece'])
    if 'rbs' in all_metrics:
        method_results['rbs'] = float(all_metrics['rbs'])
    if 'Brier' in all_metrics:
        method_results['Brier'] = float(all_metrics['Brier'])

    # Keep backward compatibility metrics (using 15 bins)
    method_results['ece'] = float(all_metrics.get('ece', 0))
    method_results['adaece'] = float(all_metrics.get('adaece', 0))
    method_results['cece'] = float(all_metrics.get('cece', 0))
    method_results['ece_debiased'] = float(all_metrics.get('ece_debiased', 0))

    overall_results['overall'][method_key] = method_results


def run(model_name, seed, valid_size):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(seed)
    cache_path = os.path.join("pt", f"imagenet_lt_{model_name}_cache_seed{seed}.pt")
    if logits_exist(dataset_name, model_name, seed, 0.2, 'CE', corruption_type, severity, train_loss):
        print("Loading cached logits/features...")

        '''
        cache = torch.load(cache_path, map_location="cpu")
        val_logits = cache["val_logits_np"]  # numpy
        val_labels = cache["val_labels_np"]
        val_features = cache["val_features_np"]
        test_logits = cache["test_logits_np"]
        test_labels = cache["test_labels_np"]
        test_features = cache["test_features_np"]
        '''

        val_logits, val_labels, test_logits, test_labels, val_features, test_features = load_logits(dataset_name,
                                                                                                    model_name, seed,
                                                                                                    0.2, 'CE',
                                                                                                    corruption_type,
                                                                                                    severity,
                                                                                                    train_loss)

    else:
        print("Cache not found → computing logits/features...")
        model = get_pretrained_model(model_name)
        model.eval()
        norm_mean, norm_std = get_model_normalization(model_name)
        input_size = get_model_input_size(model_name)
        val_loader, test_loader = get_imagenet_lt_data_loader("/share/datasets/ImageNet-LT", pin_memory=True,
                                                              random_seed=seed, mean=norm_mean, std=norm_std,
                                                              image_size=input_size, valid_size=0.2)
        # Compute logits and features
        val_logits = []
        val_labels = []
        val_features = []
        test_logits = []
        test_labels = []
        test_features = []

        print("Computing logits and features for calibration set...")
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                # Get features from model (all models should support return_features=True now)
                outputs, features = model(inputs, return_features=True)
                val_features.append(features.cpu().numpy())
                val_logits.append(outputs.cpu().numpy())
                val_labels.append(labels.numpy())

        print("Computing logits and features for test set...")
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs = inputs.to(device)
                # Get features from model (all models should support return_features=True now)
                outputs, features = model(inputs, return_features=True)
                test_features.append(features.cpu().numpy())
                test_logits.append(outputs.cpu().numpy())
                test_labels.append(labels.numpy())

        val_logits = np.vstack(val_logits)
        val_labels = np.hstack(val_labels)
        val_features = np.vstack(val_features)
        test_logits = np.vstack(test_logits)
        test_labels = np.hstack(test_labels)
        test_features = np.vstack(test_features)

        save_logits(val_logits, val_labels, test_logits, test_labels, val_features, test_features,
                    dataset_name, model_name, seed, 0.2, 'CE', corruption_type, severity, train_loss)
        '''
        torch.save({
            "val_logits_np": val_logits,
            "val_labels_np": val_labels,
            "val_features_np": val_features,
            "test_logits_np": test_logits,
            "test_labels_np": test_labels,
            "test_features_np": test_features,
        }, cache_path)

        print(f"Saved cache to {cache_path}")
        '''
    val_features,val_logits,val_labels=bootstrap_val_by_global_valid_size(val_features, val_logits,val_labels,valid_size,0.2)

    val_logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32)

    test_logits_tensor = torch.tensor(test_logits, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

    logitsgap_values = []  # Store logitsgap values
    temperatures = []  # Store temperatures generated by SMART
    optimal_temp = None  # Store optimal temperature from TS
    result_dir = os.path.join("results", f"{dataset_name}_{model_name}_seed{seed}_vs{valid_size}")
    os.makedirs(result_dir, exist_ok=True)
    overall_results_file = os.path.join(result_dir, f"calibration_results.json")
    if os.path.exists(overall_results_file):
        # Load existing results if available
        with open(overall_results_file, "r") as f:
            overall_results = json.load(f)
        print(f"Loaded existing results from {overall_results_file}")
    else:
        # Create new results dictionary if not available
        overall_results = {
            'dataset': dataset_name,
            'model': model_name,
            'seed': seed,
            'valid_size': valid_size,
            'overall': {}
        }

    # Prepare to load or compute logitsgap values
    if "SMART" in RUN_METHODS:
        # Get paths based on parameters
        paths = get_logit_paths(dataset_name, model_name, seed,
                                valid_size, smart_loss, corruption_type, severity, train_loss)
        logitsgap_file = paths['test_logitsgap_values']

        # Check if logitsgap values file exists
        if os.path.exists(logitsgap_file):
            print(f"Loading cached test logitsgap values from {logitsgap_file}")
            with open(logitsgap_file, "r") as f:
                logitsgap_dict = json.load(f)
            logitsgap_values = logitsgap_dict["logitsgap"] if "logitsgap" in logitsgap_dict else logitsgap_dict[
                "hardness"]
        else:
            # If not exists, will be computed in SMART section
            print("logitsgap values will be computed during SMART calibration")

    # Save logits with parameter-specific filenames
    # save_logits(val_logits, val_labels, test_logits, test_labels, val_features, test_features,dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity, train_loss)

    if "uncalibrated" in RUN_METHODS:
        uncal_probs = F.softmax(test_logits_tensor, dim=1)
        # Compute and print metrics
        all_metrics = get_all_metrics(logits=test_logits_tensor, labels=test_labels_tensor, n_bins=15)
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='uncalibrated',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn='none'
        )

    if "TS" in RUN_METHODS:
        print("\nTraining Temperature Scaling...")

        # Set seed before creating and training TS calibrator
        set_seed(seed)

        # Initialize and train the calibrator
        ts_calibrator = TemperatureScalingCalibrator(
            loss_type='CE',
        )

        val_logits_device = val_logits_tensor.to(device)
        val_labels_device = val_labels_tensor.to(device)
        ts_calibrator.fit(val_logits_device, val_labels_device)

        # Calibrate test logits
        test_logits_device = test_logits_tensor.to(device)

        calibrated_logits = ts_calibrator.calibrate(test_logits_device, return_logits=True)
        ts_probs = F.softmax(calibrated_logits, dim=1).detach().cpu().numpy()

        # Get optimal temperature parameter
        optimal_temp = ts_calibrator.temperature.item()

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=calibrated_logits,
            labels=test_labels_tensor, )

        print(f"Optimal temperature: {optimal_temp:.4f}")

        # Save optimal temperature value for visualization
        optimal_temp = float(optimal_temp)

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key=f'TS_CE',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn='CE',
            additional_params={'temp': float(optimal_temp)}
        )

    if "PTS" in RUN_METHODS:
        print("\nTraining Parametric Temperature Scaling...")

        # Update args for PTS calibration

        # Initialize PTSCalibrator with overwrite flag and fixed seed
        pts_calibrator = PTSCalibrator(
            steps=10000,
            lr=0.00005,
            nlayers=2,
            n_nodes=5,
            loss_fn="MSE",
            top_k_logits=10,
            seed=seed  # Use the same seed for consistency
        ).to(device)

        val_logits_device = val_logits_tensor.to(device)
        val_labels_device = val_labels_tensor.to(device)
        pts_calibrator.fit(val_logits_device, val_labels_device)

        # Calibrate test logits
        test_logits_device = test_logits_tensor.to(device)
        pts_probs = pts_calibrator.calibrate(test_logits_device).cpu().numpy()

        # Get calibrated logits for metric calculation
        calibrated_logits = pts_calibrator.calibrate(test_logits_device, return_logits=True)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=calibrated_logits,
            labels=test_labels_tensor,
        )

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key=f'PTS_MSE',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn='MSE'
        )

    if "CTS" in RUN_METHODS:
        print("\nTraining Class-based Temperature Scaling...")

        # Update args for CTS calibration

        cts_loss = "CE"

        # Ensure labels are the correct data type for the loss function
        if cts_loss == 'soft_ece':
            # Convert labels to int64 for soft_ece loss
            val_labels_for_ts = val_labels_tensor.long()
        else:
            val_labels_for_ts = val_labels_tensor

        cts_calibrator = CTSCalibrator(
            n_class=1000,  # Number of classes
            n_bins=15,
            n_iter=5,  # Number of bins for ECE computation
        ).to(device)

        # Set seed before fitting to ensure reproducibility
        set_seed(seed)

        val_logits_device = val_logits_tensor.to(device)
        val_labels_device = val_labels_for_ts.to(device)

        # Fit the calibrator on validation data
        cts_calibrator.fit(val_logits_device, val_labels_device, ts_loss=cts_loss)

        # Calibrate test logits
        test_logits_device = test_logits_tensor.to(device)

        cts_probs = cts_calibrator.calibrate(test_logits_device).cpu().detach().numpy()

        # Get calibrated logits for metric calculation
        calibrated_logits = cts_calibrator.calibrate(test_logits_device, return_logits=True)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=calibrated_logits,
            labels=test_labels_tensor,
        )

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key=f'CTS_{cts_loss}',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn=cts_loss
        )

    if "ETS" in RUN_METHODS:
        print("\nTraining Ensemble Temperature Scaling...")

        # Get number of classes from dataset
        n_classes = 1000

        # Set seed before creating and training ETS calibrator
        set_seed(seed)
        ets_loss = 'mse'
        # Initialize ETS calibrator
        ets_calibrator = ETSCalibrator(loss_type=ets_loss, n_classes=n_classes)

        # Fit the calibrator on validation data
        ets_calibrator.fit(val_logits, val_labels)

        # Calibrate test logits
        ets_probs = ets_calibrator.calibrate(test_logits)
        ets_probs_tensor = torch.tensor(ets_probs, dtype=torch.float32)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=ets_probs_tensor,  # ETS returns probabilities
            labels=test_labels_tensor,
        )
        print(f"Optimal temperature: {ets_calibrator.get_temperature():.4f}")
        print(f"Optimal weights: {ets_calibrator.get_weights()}")

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key=f'ETS_{ets_loss}',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn=ets_loss,
            additional_params={
                'temp': float(ets_calibrator.get_temperature()),
                'weights': ets_calibrator.get_weights()
            }
        )

    if "SMART" in RUN_METHODS:
        print("\nTraining Sample logitsgap Aware Temperature Scaling...")
        # Use SMART loss function from argparse

        print("Training SMART with loss function:", smart_loss)
        smart = SMART(epochs=2000, dataset_name=dataset_name,
                      model_name=model_name, seed_value=seed,
                      valid_size=valid_size, loss_fn=smart_loss,
                      patience=200, min_delta=0.0001,
                      corruption_type=corruption_type, severity=severity,
                      train_loss=train_loss)

        # Try to load existing SMART model
        if not smart.load_model():
            # Train new model if loading failed
            smart.fit(val_logits, val_labels)

        smart_probs = smart.calibrate(test_logits)
        smart_probs_tensor = torch.tensor(smart_probs, dtype=torch.float32)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=smart_probs_tensor,  # SMART returns probabilities
            labels=test_labels_tensor,
        )

        # Get logitsgap and temperature values for visualization
        paths = get_logit_paths(dataset_name, model_name, seed,
                                valid_size, smart_loss, corruption_type, severity, train_loss)
        logitsgap_file = paths['test_logitsgap_values']

        # Load logitsgap and temperature values for visualization
        if os.path.exists(logitsgap_file) and len(logitsgap_values) == 0:
            # If logitsgap values were not loaded before, load them now
            print(f"Loading cached test logitsgap values from {logitsgap_file}")
            with open(logitsgap_file, "r") as f:
                logitsgap_dict = json.load(f)
            logitsgap_values = logitsgap_dict["logitsgap"] if "logitsgap" in logitsgap_dict else logitsgap_dict[
                "hardness"]

        # Calculate SMART-generated temperature values
        if len(logitsgap_values) > 0:
            # Use SMART model to predict temperatures
            logitsgap_tensor = torch.tensor(logitsgap_values, dtype=torch.float32)
            normalized_logitsgap = (logitsgap_tensor - smart.logitsgap_mean) / (smart.logitsgap_std + 1e-8)

            with torch.no_grad():
                temperatures = smart.temp_model(normalized_logitsgap).detach().cpu().numpy().flatten().tolist()

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key=f'SMART_{smart_loss}',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn=smart_loss
        )

    if "HB" in RUN_METHODS:
        print("\nTraining Histogram Binning...")
        set_seed(seed)

        # Initialize Histogram Binning calibrator
        hb_calibrator = HistogramBinningCalibrator(n_bins=15, strategy='uniform')

        # Fit the calibrator on validation data
        hb_calibrator.fit(val_logits_tensor, val_labels_tensor)

        # Apply calibration to test set
        hb_logits = hb_calibrator.calibrate(test_logits_tensor, return_logits=True)
        hb_probs = F.softmax(hb_logits, dim=1)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=hb_logits,
            labels=test_labels_tensor,
        )

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='HB',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn='uniform_binning'
        )

    if "BBQ" in RUN_METHODS:
        print("\nTraining BBQ...")
        set_seed(seed)

        # Initialize BBQ calibrator
        bbq_calibrator = BBQCalibrator(score_type='max_prob', n_bins_max=20)

        # Fit the calibrator on validation data
        bbq_calibrator.fit(val_logits_tensor, val_labels_tensor)

        # Apply calibration to test set
        bbq_logits = bbq_calibrator.calibrate(test_logits_tensor, return_logits=True)
        bbq_probs = F.softmax(bbq_logits, dim=1)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=bbq_logits,
            labels=test_labels_tensor,
        )

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='BBQ',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn='bayesian_binning'
        )

    if "VS" in RUN_METHODS:
        print("\nTraining Vector Scaling...")
        set_seed(seed)

        # Initialize Vector Scaling calibrator
        vs_calibrator = VectorScalingCalibrator(loss_type='nll', bias=True)

        # Fit the calibrator on validation data
        vs_calibrator.fit(val_logits_tensor, val_labels_tensor)

        # Apply calibration to test set
        vs_logits = vs_calibrator.calibrate(test_logits_tensor, return_logits=True)
        vs_probs = F.softmax(vs_logits, dim=1)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=vs_logits,
            labels=test_labels_tensor,
        )

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='VS',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn='vector_scaling'
        )

    if "GC" in RUN_METHODS:
        print("\nTraining Group Calibration...")
        set_seed(seed)

        # Initialize Group Calibration calibrator (matching original paper: K=2, U=20, λ=0.1)
        gc_calibrator = GroupCalibrationCalibrator(
            num_groups=2,
            num_partitions=20,
            weight_decay=0.1
        )

        # Fit the calibrator on validation data
        gc_calibrator.fit(val_logits_tensor, val_labels_tensor)

        # Apply calibration to test set
        gc_logits = gc_calibrator.calibrate(test_logits_tensor, return_logits=True)
        gc_probs = F.softmax(gc_logits, dim=1)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=gc_logits,
            labels=test_labels_tensor,
        )

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='GC',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn='group_calibration'
        )


    if "ProCal_DR" in RUN_METHODS:
        print("\nTraining ProCal Density-Ratio Calibration...")
        set_seed(seed)

        # Initialize ProCal Density-Ratio calibrator
        procal_dr_calibrator = ProCalDensityRatioCalibrator(
            k_neighbors=10,
            bandwidth='normal_reference',
            kernel='KDEMultivariate',
            distance_measure='L2',
            normalize_features=True
        )

        # Convert features to tensors
        val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)

        # Fit the calibrator on validation data with features
        procal_dr_calibrator.fit(val_logits_tensor, val_labels_tensor, val_features_tensor)

        # Apply calibration to test set with features
        procal_dr_probs = procal_dr_calibrator.calibrate(test_logits_tensor, test_features_tensor)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=procal_dr_probs,
            labels=test_labels_tensor,
        )

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='ProCal_DR',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn='density_ratio'
        )


    if "FC" in RUN_METHODS:
        print("\nTraining Feature Clipping Calibration...")
        set_seed(seed)

        # Create the same model that was used to extract features
        model = get_pretrained_model(model_name)
        model.eval()

        # Get the classifier function - all our models have a classifier method
        classifier_fn = model.classifier

        # Initialize Feature Clipping calibrator
        fc_calibrator = FeatureClippingCalibrator(cross_validate='ece')

        # Convert features to tensors
        val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)

        # Set optimal clipping parameter using validation data
        optimal_clip = fc_calibrator.set_feature_clip(
            val_features_tensor, val_logits_tensor, val_labels_tensor,  classifier_fn
        )

        print(f"Optimal clipping parameter: {optimal_clip:.4f}")

        # Apply feature clipping to test features and get calibrated logits
        clipped_test_features = fc_calibrator.feature_clipping(test_features_tensor, optimal_clip)
        fc_logits =  classifier_fn(clipped_test_features)
        fc_probs = F.softmax(fc_logits, dim=1)

        # Compute and print metrics
        all_metrics = get_all_metrics(
            logits=fc_logits,
            labels=test_labels_tensor,
        )

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='FC',
            all_metrics=all_metrics,
            bins_list=[15],
            loss_fn='feature_clipping'
        )

    with open(overall_results_file, "w") as f:
        json.dump(overall_results, f, indent=4)

    print(f"Saved updated results to {overall_results_file}")


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    '''
    model_list=['resnet50','resnet152','densenet121','wide_resnet','vit_b_16','vit_b_32','vit_l_16','swin_b','beit_base', 'beit_large',  'convnext_tiny', 'convnext_base', 'convnext_large',
                  'eva02_small', 'eva02_base',  'mobilenet_v2', 'mlp_mixer_b16','eva02_large','beitv2_base']
    '''
    model_list = ['beitv2_base','convnext_base', 'convnext_large','convnext_tiny', 'eva02_small']
    val_size_list=[#0.0002,
                   #0.001,
                   #0.01,
                   #0.05,
                   #0.1
                   0.2
   ]
    for mode_name in model_list:
        for size in val_size_list:
            run(mode_name, 1,size)
            run(mode_name, 2,size)
            run(mode_name, 3,size)
            run(mode_name, 4,size)
            run(mode_name, 5,size)





