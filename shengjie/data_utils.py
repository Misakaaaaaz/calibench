"""
Data utility functions for managing logits, labels, features, and data loaders
"""
import os
import numpy as np
import torch
from tqdm import tqdm


def get_logit_paths(dataset_name, model_name, seed_value, valid_size=0.2, loss_fn='CE', corruption_type=None, severity=None, train_loss=None):
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


def logits_exist(dataset_name, model_name, seed_value, valid_size=0.2, loss_fn='CE', corruption_type=None, severity=None, train_loss=None):
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
    paths = get_logit_paths(dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity, train_loss)

    # Check if files exist
    files_exist = (os.path.exists(paths['val_logits']) and
                  os.path.exists(paths['val_labels']) and
                  os.path.exists(paths['val_features']) and
                  os.path.exists(paths['test_logits']) and
                  os.path.exists(paths['test_labels']) and
                  os.path.exists(paths['test_features']))

    return files_exist


def load_logits(dataset_name, model_name, seed_value, valid_size=0.2, loss_fn='CE', corruption_type=None, severity=None, train_loss=None):
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
    paths = get_logit_paths(dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity, train_loss)

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
        raise FileNotFoundError(f"Logits not found for {dataset_info}, {model_name}, seed {seed_value}, valid_size {valid_size}")

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
               dataset_name, model_name, seed_value, valid_size=0.2, loss_fn='CE', corruption_type=None, severity=None, train_loss=None):
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
    paths = get_logit_paths(dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity, train_loss)

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


def create_train_loader_for_dac(args, dataset_name):
    """
    Create a training dataloader for DAC feature extraction.

    According to the DAC paper, they use 1% of the training set for ImageNet.
    Since ImageNet training set is often not available (150GB+), we use:
    - For ImageNet-C/Sketch/LT: Original uncorrupted ImageNet validation set
    - For CIFAR: Actual training set

    This provides in-distribution reference data for KNN density estimation.

    Args:
        args: Command line arguments
        dataset_name: Name of the dataset

    Returns:
        DataLoader for training/reference data
    """
    from utils import dataset_loader

    if dataset_name == 'imagenet':
        # For ImageNet, try training set first, fall back to validation set
        try:
            train_loader = dataset_loader['imagenet'].get_data_loader(
                root=args.dataset_root,
                split='train',
                batch_size=args.test_batch_size,
                shuffle=True,
                num_workers=16,
                pin_memory=True
            )
        except FileNotFoundError:
            print("ImageNet training set not found, using validation set as reference for KNN")
            train_loader = dataset_loader['imagenet'].get_data_loader(
                root=args.dataset_root,
                split='val',
                batch_size=args.test_batch_size,
                shuffle=True,
                num_workers=16,
                pin_memory=True
            )
    elif dataset_name in ['cifar10', 'cifar100']:
        train_loader, _ = dataset_loader[dataset_name].get_train_valid_loader(
            root=args.dataset_root,
            batch_size=args.train_batch_size,
            shuffle=True,
            random_seed=args.random_seed,
            augment=False  # No augmentation for feature extraction
        )
    elif dataset_name in ['imagenet_c', 'imagenet_sketch', 'imagenet_lt', 'imagenet_original_val']:
        # For ImageNet variants, use the ORIGINAL UNCORRUPTED ImageNet validation set
        # as reference in-distribution data for KNN density estimation
        # Note: The corrupted validation set is split into calibration (20%) and test (80%)
        print(f"Using original ImageNet validation set as reference data for {dataset_name} KNN")
        train_loader = dataset_loader['imagenet'].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for DAC")

    return train_loader


def extract_train_features_for_dac(model, train_loader, device, max_samples=10000):
    """
    Extract features from training set for DAC KNN density estimation.

    According to the DAC paper, they use 1% of the training set for ImageNet (~12.8k samples).
    This function extracts features from up to max_samples training samples.

    Args:
        model: The trained model with return_features support
        train_loader: DataLoader for training data
        device: Device to run inference on
        max_samples: Maximum number of samples to extract (default 10000)

    Returns:
        torch.Tensor: Training features of shape (N, feature_dim)
    """
    model.eval()
    train_features = []
    total_samples = 0

    with torch.no_grad():
        for inputs, _ in tqdm(train_loader, desc="Extracting train features for DAC"):
            if total_samples >= max_samples:
                break

            inputs = inputs.to(device)

            # Get features from model
            _, features = model(inputs, return_features=True)
            train_features.append(features.cpu())

            total_samples += features.shape[0]

    train_features = torch.cat(train_features, dim=0)[:max_samples]
    print(f"Extracted {train_features.shape[0]} training samples for DAC")

    return train_features
