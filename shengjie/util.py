import sys
import os
import json
import time
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy import stats
import torchvision.models as models
import torchvision
# Add project paths  
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

# Calibrator imports
from calibrator.Component.model.pts import PTSCalibrator
from calibrator.Component.model.cts import CTSCalibrator
from calibrator.Component.model.temperature_scaling import TemperatureScalingCalibrator
from calibrator.Component.model.ets import ETSCalibrator
from calibrator.Component.model.histogram_binning import HistogramBinningCalibrator
from calibrator.Component.model.bbq import BBQCalibrator
from calibrator.Component.model.vector_scaling import VectorScalingCalibrator
from calibrator.Component.model.group_calibration import GroupCalibrationCalibrator
from calibrator.Component.model.procal import ProCalDensityRatioCalibrator, ProCalBinMeanShiftCalibrator
from calibrator.Component.model.feature_clipping import FeatureClippingCalibrator
from calibrator.Component.model.logit_clipping import LogitClippingCalibrator
from calibrator.Component.model.density_aware_calibration import DensityAwareCalibrator

from calibrator.Component.metrics.WeightedSoftECE import WeightedSoftECE
from calibrator.Component.metrics.SmoothSoftECE import SmoothSoftECE
from calibrator.Component.metrics.GapIndexedSoftECE import GapIndexedSoftECE

# Metrics imports
from calibrator.Component.metrics import (
    BrierLoss, FocalLoss, LabelSmoothingLoss, 
    CrossEntropyLoss, MSELoss, SoftECE, ECE, AdaptiveECE, ClasswiseECE, 
    Accuracy, NLL, KDEECE, ECEDebiased, ECESweep
)
from calibrator.Component.utils.utils import get_all_metrics, get_all_metrics_multi_bins

from Datasets.imagenet import get_data_loader

# Import models dictionary from utils
from utils import models_dict, dataset_loader, dataset_num_classes

# Import visualization functions
from plotting.visualization import (
    plot_enhanced_calibration_curve, plot_logitsgap_analysis,
    plot_temperature_distribution, plot_logitsgap_temperature_relationship,
    plot_confidence_distribution, plot_confidence_change,
    plot_logitsgap_by_correctness, plot_performance_by_logitsgap
)

# Import SMART calibration functionality
from utils.smart_calibrator import SMART, compute_logitsgap, set_seed

# Import from split utility modules
from utils.model_utils import get_model_normalization, get_model_input_size, create_model
from utils.data_utils import (
    get_logit_paths, logits_exist, load_logits, save_logits,
    create_train_loader_for_dac, extract_train_features_for_dac
)

# Constants
BINS_LIST = [5, 10, 15, 20, 25, 30]
AVAILABLE_METRICS = ['ece', 'adaece', 'cece', 'ece_debiased', 'ece_sweep', 'nll', 'accuracy', 'kde_ece', 'rbs']
DEFAULT_METRICS = ['ece', 'adaece', 'cece', 'nll', 'accuracy', 'rbs']


def get_model_and_logits(args, model_name='resnet50', batch_size=32, num_workers=4, use_cuda=True,
                         dataset_name='imagenet', seed_value=1, valid_size=0.2, loss_fn='CE'):
    """
    Get a pretrained model and compute logits for calibration and test sets
    
    Args:
        args: Command line arguments
        model_name: Name of the pretrained model to use
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        use_cuda: Whether to use CUDA (GPU) for computation
        dataset_name: Name of the dataset
        seed_value: Random seed used
        valid_size: Validation set size
        loss_fn: Loss function used for calibration
        
    Returns:
        Logits and labels for calibration and test sets
    """
    # Extract corruption parameters for ImageNet-C
    corruption_type = getattr(args, 'corruption_type', None) if dataset_name == 'imagenet_c' else None
    severity = getattr(args, 'severity', None) if dataset_name == 'imagenet_c' else None
    
    # Get train_loss for CIFAR datasets
    train_loss = args.train_loss if dataset_name.startswith('cifar') else None
    
    # Check if logits already exist
    if logits_exist(dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity, train_loss):
        dataset_info = f"{dataset_name}"
        if dataset_name == 'imagenet_c' and corruption_type is not None and severity is not None:
            dataset_info = f"{dataset_name} (corruption: {corruption_type}, severity: {severity})"
        print(f"Logits already exist for {dataset_info}, {model_name}, seed {seed_value}, valid_size {valid_size}")
        return load_logits(dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity, train_loss)
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define required arguments if not provided in args
    if not hasattr(args, 'valid_size'):
        args.valid_size = valid_size
    if not hasattr(args, 'random_seed'):
        args.random_seed = seed_value
    if not hasattr(args, 'test_batch_size'):
        args.test_batch_size = batch_size
    if not hasattr(args, 'train_batch_size'):
        args.train_batch_size = batch_size
    if not hasattr(args, 'dataset_root'):
        args.dataset_root = '/hdd/haolan/datasets/'
    if not hasattr(args, 'corruption_type') and dataset_name == 'imagenet_c':
        args.corruption_type = 'gaussian_noise'
        corruption_type = 'gaussian_noise'
    if not hasattr(args, 'severity') and dataset_name == 'imagenet_c':
        args.severity = 1
        severity = 1
    if not hasattr(args, 'train_loss'):
        args.train_loss = 'cross_entropy'
    
    # Load model using the helper function
    try:
        model = create_model(args, args.model, args.dataset, device)
        print(f"Loaded model: {model.__class__.__name__}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Set model to eval mode
    model.eval()
    
    # Fix for accessing classifier or fc attributes
    if hasattr(model, 'module'):  # Check if model is wrapped with DataParallel
        if hasattr(model.module, 'classifier'):
            model.classifier = model.module.classifier
        elif hasattr(model.module, 'fc'):
            model.fc = model.module.fc
    
    # Get model-specific normalization and input size
    norm_mean, norm_std = get_model_normalization(model_name)
    input_size = get_model_input_size(model_name)
    print(f"Using normalization for {model_name}: mean={norm_mean}, std={norm_std}")
    print(f"Using input size for {model_name}: {input_size}x{input_size}")

    # Get appropriate data loaders based on dataset
    try:
        val_loader = None
        test_loader = None

        if args.dataset == 'imagenet':
            val_loader = dataset_loader['imagenet'].get_data_loader(
                root=args.dataset_root,
                split='val',
                batch_size=args.test_batch_size,
                shuffle=True,
                valid_size=args.valid_size,
                num_workers=16,
                pin_memory=True,
                random_seed=args.random_seed,
                mean=norm_mean,
                std=norm_std,
                image_size=input_size
            )

            test_loader = dataset_loader['imagenet'].get_data_loader(
                root=args.dataset_root,
                split='test',
                batch_size=args.test_batch_size,
                shuffle=True,
                valid_size=args.valid_size,
                num_workers=16,
                pin_memory=True,
                random_seed=args.random_seed,
                mean=norm_mean,
                std=norm_std,
                image_size=input_size
            )
        elif args.dataset in ['cifar10', 'cifar100']:
            _, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
                root=args.dataset_root,
                batch_size=args.train_batch_size,
                shuffle=True,
                random_seed=1,
                augment=True
            )

            test_loader = dataset_loader[args.dataset].get_test_loader(
                root=args.dataset_root,
                batch_size=args.test_batch_size,
                shuffle=False
            )
        elif args.dataset == 'imagenet_c':
            # For ImageNet-C, we have both validation and test sets
            val_loader, test_loader = dataset_loader[args.dataset].get_imagenet_c_data_loader(
                root="/hdd/haolan/datasets/ImageNet-C/",
                batch_size=args.test_batch_size,
                corruption_type=args.corruption_type,
                severity=args.severity,
                num_workers=16,
                pin_memory=True,
                valid_size=args.valid_size,
                random_seed=args.random_seed,
                mean=norm_mean,
                std=norm_std,
                image_size=input_size
            )
        
        elif args.dataset == 'imagenet_lt':
            val_loader, test_loader = dataset_loader[args.dataset].get_imagenet_lt_data_loader(
                root="/hdd/haolan/datasets/ImageNet-LT/",
                batch_size=args.test_batch_size,
                num_workers=16,
                pin_memory=True,
                valid_size=args.valid_size,
                random_seed=args.random_seed,
                mean=norm_mean,
                std=norm_std,
                image_size=input_size
            )
        elif args.dataset == 'imagenet_sketch':
            val_loader, test_loader = dataset_loader[args.dataset].get_imagenet_sketch_data_loader(
                # root="/hdd/haolan/datasets/ImageNet-Sketch/",
                root=args.dataset_root,
                batch_size=args.test_batch_size,
                # num_workers=16,
                num_workers=8,
                pin_memory=True,
                valid_size=args.valid_size,
                random_seed=args.random_seed,
                mean=norm_mean,
                std=norm_std,
                image_size=input_size
            )
        elif args.dataset == 'iwildcam':
            val_loader = dataset_loader[args.dataset].get_data_loader(
                root="/hdd/datasets/wilds/iwildcam_v2.0",
                split='val',
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                mean=norm_mean,
                std=norm_std,
                image_size=input_size
            )

            test_loader = dataset_loader[args.dataset].get_data_loader(
                root="/hdd/datasets/wilds/iwildcam_v2.0",
                split='test',
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                mean=norm_mean,
                std=norm_std,
                image_size=input_size
            )
        else:
            raise ValueError(f"Dataset {args.dataset} not supported")
        
        # Check if loaders were created successfully
        if val_loader is None or test_loader is None:
            raise ValueError(f"Failed to create data loaders for dataset {args.dataset}")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        raise
    
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
    
    # Save logits with parameter-specific filenames
    save_logits(val_logits, val_labels, test_logits, test_labels, val_features, test_features,
               dataset_name, model_name, seed_value, valid_size, loss_fn, corruption_type, severity, train_loss)
    
    return val_logits, val_labels, test_logits, test_labels, val_features, test_features


def compute_and_print_metrics(logits, labels, method_name, bins_list=None, device='cuda', enabled_metrics=None):
    """
    Compute and print metrics for a calibration method
    
    Args:
        logits: torch.Tensor - calibrated logits or probabilities
        labels: torch.Tensor - ground truth labels  
        method_name: str - name of the calibration method
        bins_list: list - list of bin sizes for evaluation
        device: str - device to use for computation
        enabled_metrics: list - list of metrics to compute and print
    
    Returns:
        dict: computed metrics
    """
    if bins_list is None:
        bins_list = BINS_LIST
    if enabled_metrics is None:
        enabled_metrics = DEFAULT_METRICS
        
    # Move tensors to device
    logits = logits.to(device)
    labels = labels.to(device)
    
    # Determine if logits are actually probabilities
    is_probs = (logits.dim() == 2 and 
                torch.allclose(logits.sum(dim=1), torch.ones(logits.size(0), device=device), atol=1e-3))
    
    # Use get_all_metrics_multi_bins function from utils to calculate all metrics
    all_metrics = get_all_metrics_multi_bins(
        labels=labels,
        logits=logits if not is_probs else None,
        probs=logits if is_probs else None,
        bins_list=bins_list
    )
    
    # Print metrics based on enabled_metrics
    metric_strs = []
    if 'accuracy' in enabled_metrics:
        metric_strs.append(f"Accuracy: {all_metrics['accuracy']:.4f}")

    if any(metric.startswith('ece') for metric in enabled_metrics):
        ece_strs = []
        for bins in bins_list:
            if 'ece' in enabled_metrics and f'ece_{bins}' in all_metrics:
                ece_strs.append(f"ECE_{bins}: {all_metrics[f'ece_{bins}']:.4f}")
        if ece_strs:
            metric_strs.append(", ".join(ece_strs))
    
    if any(metric.startswith('adaece') for metric in enabled_metrics):
        adaece_strs = []
        for bins in bins_list:
            if 'adaece' in enabled_metrics and f'adaece_{bins}' in all_metrics:
                adaece_strs.append(f"AdaECE_{bins}: {all_metrics[f'adaece_{bins}']:.4f}")
        if adaece_strs and len(bins_list) <= 3:  # Only show for small bin lists to avoid clutter
            metric_strs.append(", ".join(adaece_strs))
    
    if any(metric.startswith('cece') for metric in enabled_metrics):
        cece_strs = []
        for bins in bins_list:
            if 'cece' in enabled_metrics and f'cece_{bins}' in all_metrics:
                cece_strs.append(f"CECE_{bins}: {all_metrics[f'cece_{bins}']:.4f}")
        if cece_strs and len(bins_list) <= 3:  # Only show for small bin lists to avoid clutter
            metric_strs.append(", ".join(cece_strs))
            
    if any(metric.startswith('ece_debiased') for metric in enabled_metrics):
        debiased_strs = []
        for bins in bins_list:
            if 'ece_debiased' in enabled_metrics and f'ece_debiased_{bins}' in all_metrics:
                debiased_strs.append(f"ECE_Debiased_{bins}: {all_metrics[f'ece_debiased_{bins}']:.4f}")
        if debiased_strs:
            metric_strs.append(", ".join(debiased_strs))
    
    if 'ece_sweep' in enabled_metrics and 'ece_sweep' in all_metrics:
        metric_strs.append(f"ECE_Sweep: {all_metrics['ece_sweep']:.4f}")
        
    if 'kde_ece' in enabled_metrics:
        # Add KDE ECE metric if requested
        try:
            kde_ece_metric = KDEECE(p=1, mc_type='top_label', bandwidth=0.0268)
            if not is_probs:  # logits
                kde_ece_value = kde_ece_metric(logits=logits, labels=labels)
            else:  # probabilities - convert to pseudo-logits
                pseudo_logits = torch.log(logits + 1e-8)
                kde_ece_value = kde_ece_metric(logits=pseudo_logits, labels=labels)
            all_metrics['kde_ece'] = float(kde_ece_value.item())
            metric_strs.append(f"KDE ECE: {all_metrics['kde_ece']:.4f}")
        except:
            all_metrics['kde_ece'] = -1
            if 'kde_ece' in enabled_metrics:
                metric_strs.append("KDE ECE: -1.0000 (failed)")
    
    if 'nll' in enabled_metrics:
        metric_strs.append(f"NLL: {all_metrics['nll']:.4f}")
    
    if 'rbs' in enabled_metrics and 'rbs' in all_metrics:
        metric_strs.append(f"RBS: {all_metrics['rbs']:.4f}")
    
    print(f"{method_name} - {', '.join(metric_strs)}")
    return all_metrics


def store_method_results(overall_results, method_key, all_metrics, bins_list=None, loss_fn=None, additional_params=None):
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
        bins_list = BINS_LIST
        
    method_results = {
        'acc': float(all_metrics['accuracy']),
        'nll': float(all_metrics['nll']),
        'loss_fn': loss_fn or 'none'
    }
    
    # Add additional parameters
    if additional_params:
        method_results.update(additional_params)
    
    # Add bin-specific metrics
    for bins in bins_list:
        if f'ece_{bins}' in all_metrics:
            method_results[f'ece_{bins}'] = float(all_metrics[f'ece_{bins}'])
        if f'adaece_{bins}' in all_metrics:
            method_results[f'adaece_{bins}'] = float(all_metrics[f'adaece_{bins}'])
        if f'cece_{bins}' in all_metrics:
            method_results[f'cece_{bins}'] = float(all_metrics[f'cece_{bins}'])
        if f'ece_debiased_{bins}' in all_metrics:
            method_results[f'ece_debiased_{bins}'] = float(all_metrics[f'ece_debiased_{bins}'])
    
    # Add non-bin specific metrics
    if 'ece_sweep' in all_metrics:
        method_results['ece_sweep'] = float(all_metrics['ece_sweep'])
    if 'kde_ece' in all_metrics:
        method_results['kde_ece'] = float(all_metrics['kde_ece'])
    if 'rbs' in all_metrics:
        method_results['rbs'] = float(all_metrics['rbs'])
    
    # Keep backward compatibility metrics (using 15 bins)
    method_results['ece'] = float(all_metrics.get('ece_15', 0))
    method_results['adaece'] = float(all_metrics.get('adaece_15', 0))
    method_results['cece'] = float(all_metrics.get('cece_15', 0))
    method_results['ece_debiased'] = float(all_metrics.get('ece_debiased_15', 0))
    
    overall_results['overall'][method_key] = method_results


def generate_visualizations(test_logits, test_labels, calibration_plots, logitsgap_values=None, 
                            temperatures=None, optimal_temp=None, methods_run=None, model_name="Model", 
                            plot_dir="plots"):
    """
    Generate visualization plots for calibration methods
    
    Args:
        test_logits: Logits for the test set
        test_labels: Labels for the test set
        calibration_plots: Dictionary mapping method names to probability arrays
        logitsgap_values: List of sample logitsgap values (optional)
        temperatures: List of temperatures generated by SMART (optional)
        optimal_temp: Optimal temperature found by TS method (optional)
        methods_run: List of calibration methods run
        model_name: Name of the model
        plot_dir: Directory to save plots
    """
    if methods_run is None or len(methods_run) == 0:
        print("No calibration methods to visualize")
        return
    
    # Ensure directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Plot calibration curves for each method individually
    for method in methods_run:
        method_key = None
        if method == "uncalibrated" and "Uncalibrated" in calibration_plots:
            method_key = "Uncalibrated"
            filename = "Uncalibrated"
        elif method == "TS" and "TS" in calibration_plots:
            method_key = "TS"
            filename = "Temperature_Scaling"
        elif method == "PTS" and "PTS" in calibration_plots:
            method_key = "PTS"
            filename = "Parametric_Temperature_Scaling"
        elif method == "CTS" and "CTS" in calibration_plots:
            method_key = "CTS"
            filename = "Class_Temperature_Scaling"
        elif method == "ETS" and "ETS" in calibration_plots:
            method_key = "ETS"
            filename = "Ensemble_Temperature_Scaling"
        elif method == "SMART" and "SMART" in calibration_plots:
            method_key = "SMART"
            filename = "SMART"

        if method_key:
            plot_enhanced_calibration_curve(
                calibration_plots[method_key],
                test_labels,
                filename,
                plot_dir
            )

    # 2. If logitsgap values exist, plot logitsgap analysis
    if logitsgap_values is not None and len(logitsgap_values) > 0:
        plot_logitsgap_analysis(logitsgap_values, model_name, plot_dir)

        # 3. If both SMART-generated temperatures and logitsgap exist, plot their relationship
        if temperatures is not None and len(temperatures) > 0:
            plot_temperature_distribution(temperatures, model_name, optimal_temp, plot_dir)
            plot_logitsgap_temperature_relationship(logitsgap_values, temperatures, model_name, optimal_temp, plot_dir)

    # 4. Compare confidence distributions across different methods
    if len(calibration_plots) > 1:
        plot_confidence_distribution(calibration_plots, model_name, plot_dir)

        # 5. If uncalibrated and calibrated probabilities exist, compare confidence changes
        if "Uncalibrated" in calibration_plots:
            # Create a dictionary containing only calibration methods
            calibrated_plots = {k: v for k, v in calibration_plots.items() if k != "Uncalibrated"}
            if logitsgap_values is not None:
                plot_confidence_change(
                    calibration_plots["Uncalibrated"],
                    calibrated_plots,
                    logitsgap_values,
                    model_name,
                    plot_dir
                )

    # 6. If logitsgap values exist, analyze logitsgap by correctness
    if logitsgap_values is not None and "Uncalibrated" in calibration_plots:
        plot_logitsgap_by_correctness(
            logitsgap_values,
            calibration_plots["Uncalibrated"],
            test_labels,
            model_name,
            plot_dir
        )

    print("Visualization generation complete!")


def evaluate_calibration_methods(val_logits, val_labels, test_logits, test_labels, val_features, test_features, args,
                                batch_size=100, smart_epochs=200,
                                dataset_name='imagenet', model_name='resnet50', seed_value=1,
                                valid_size=0.2, loss_fn='CE',
                                run_methods=None, patience=20, min_delta=0.0001,
                                eval_metrics=None, eval_bins=None):
    """
    Evaluate different calibration methods
    
    Args:
        val_logits: Logits for the calibration set
        val_labels: Labels for the calibration set
        test_logits: Logits for the test set
        test_labels: Labels for the test set
        val_features: Features for the calibration set (for ProCal methods)
        test_features: Features for the test set (for ProCal methods)
        args: Original command line arguments
        batch_size: Batch size for processing
        smart_epochs: Number of epochs for smart training
        dataset_name: Name of the dataset
        model_name: Name of the model
        seed_value: Random seed used
        valid_size: Validation set size
        loss_fn: Loss function used for training (only used for CIFAR models)
        run_methods: List of calibration methods to run, defaults to ["uncalibrated", "TS", "PTS", "CTS", "SMART"]
        patience: Early stopping patience (number of epochs without improvement)
        min_delta: Minimum change in loss to qualify as improvement for early stopping
    """
    # Set default parameters if not provided
    if run_methods is None:
        run_methods = ["uncalibrated", "TS", "PTS", "CTS", "ETS", "SMART", "HB", "BBQ", "VS", "GC"]
    if eval_metrics is None:
        eval_metrics = DEFAULT_METRICS
    if eval_bins is None:
        eval_bins = BINS_LIST

    # Extract corruption type and severity from args for ImageNet-C
    corruption_type = getattr(args, 'corruption_type', None) if dataset_name == 'imagenet_c' else None
    severity = getattr(args, 'severity', None) if dataset_name == 'imagenet_c' else None
    
    # Get train_loss for CIFAR datasets
    train_loss = args.train_loss if dataset_name.startswith('cifar') else None

    # Create results directory with parameter-specific subfolder
    if dataset_name.startswith('imagenet'):
        result_dir = f"results/{dataset_name}_{model_name}_seed{seed_value}_vs{valid_size}"
    elif dataset_name.startswith('cifar'):
        # Always include train_loss for CIFAR datasets
        result_dir = f"results/{dataset_name}_{model_name}_{train_loss}_seed{seed_value}"
    else:
        result_dir = f"results/{dataset_name}_{model_name}_seed{seed_value}"
        
    if dataset_name == 'imagenet_c' and corruption_type is not None and severity is not None:
        result_dir = f"results/{dataset_name}_{corruption_type}_s{severity}_{model_name}_seed{seed_value}_vs{valid_size}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Create plots directory with parameter-specific subfolder
    if dataset_name.startswith('imagenet'):
        plot_dir = f"plots/{dataset_name}_{model_name}_seed{seed_value}_vs{valid_size}"
    elif dataset_name.startswith('cifar'):
        # Always include train_loss for CIFAR datasets  
        plot_dir = f"plots/{dataset_name}_{model_name}_{train_loss}_seed{seed_value}"
    else:
        plot_dir = f"plots/{dataset_name}_{model_name}_seed{seed_value}"
        
    if dataset_name == 'imagenet_c' and corruption_type is not None and severity is not None:
        plot_dir = f"plots/{dataset_name}_{corruption_type}_s{severity}_{model_name}_seed{seed_value}_vs{valid_size}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize overall_results dictionary
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
            'seed': seed_value,
            'valid_size': valid_size,
            'overall': {}
        }
    
    # Convert numpy arrays to torch tensors
    val_logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    test_logits_tensor = torch.tensor(test_logits, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    
    # Store results for all methods
    results = {
        'cal': [],
        'ece': [],
        'adaece': [],
        'cece': [],
        'nll': [],
        'accuracy': [],
        'ece_debiased': []
    }
    
    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dictionary to store all calibration results
    calibration_plots = {}
    
    # Get loss functions for each calibration method
    ts_loss = getattr(args, 'ts_loss', 'CE')
    pts_loss = getattr(args, 'pts_loss', 'MSE')
    cts_loss = getattr(args, 'cts_loss', 'CE')
    ets_loss = getattr(args, 'ets_loss', 'mse')
    smart_loss = getattr(args, 'smart_loss', 'soft_ece')
    
    # Variables for visualization
    logitsgap_values = []  # Store logitsgap values
    temperatures = []     # Store temperatures generated by SMART
    optimal_temp = None   # Store optimal temperature from TS
    
    # Prepare to load or compute logitsgap values
    if "SMART" in run_methods:
        # Get paths based on parameters
        paths = get_logit_paths(dataset_name, model_name, seed_value, 
                               valid_size, smart_loss, corruption_type, severity, train_loss)
        logitsgap_file = paths['test_logitsgap_values']
        
        # Check if logitsgap values file exists
        if os.path.exists(logitsgap_file):
            print(f"Loading cached test logitsgap values from {logitsgap_file}")
            with open(logitsgap_file, "r") as f:
                logitsgap_dict = json.load(f)
            logitsgap_values = logitsgap_dict["logitsgap"] if "logitsgap" in logitsgap_dict else logitsgap_dict["hardness"]
        else:
            # If not exists, will be computed in SMART section
            print("logitsgap values will be computed during SMART calibration")
    
    # Check which methods already exist in the results
    methods_to_run = []
    for method in run_methods:
        method_key = method.replace("-", "_").replace(" ", "_")
        
        # Determine the appropriate loss function for this method
        method_loss = None
        if method_key == 'TS':
            method_loss = ts_loss
        elif method_key == 'PTS':
            method_loss = pts_loss
        elif method_key == 'CTS':
            method_loss = cts_loss
        elif method_key == 'ETS':
            method_loss = ets_loss
        elif method_key == 'SMART':
            method_loss = smart_loss
        elif method_key == 'uncalibrated':
            # Uncalibrated doesn't use a loss function
            method_loss = 'none'
        elif method_key == 'LC':
            # Logit Clipping uses its own clipping mechanism
            method_loss = 'logit_clipping'
        elif method_key == 'GC':
            # Group Calibration uses group calibration mechanism
            method_loss = 'group_calibration'
        elif method_key == 'FC':
            # Feature Clipping uses feature clipping mechanism
            method_loss = 'feature_clipping'
        elif method_key == 'HB':
            # Histogram Binning uses uniform binning
            method_loss = 'uniform_binning'
        elif method_key == 'BBQ':
            # BBQ uses bayesian binning
            method_loss = 'bayesian_binning'
        elif method_key == 'VS':
            # Vector Scaling uses vector scaling
            method_loss = 'vector_scaling'
        elif method_key == 'ProCal_DR':
            # ProCal Density-Ratio uses density ratio
            method_loss = 'density_ratio'
        elif method_key == 'ProCal_BMS':
            # ProCal Bin-Mean-Shift uses bin mean shift
            method_loss = 'bin_mean_shift'
        
        # Check if method exists in overall results with current loss_fn
        method_exists = False
        if method_key in overall_results.get('overall', {}) and not args.overwrite:
            # If the method data has loss_fn info
            if isinstance(overall_results['overall'][method_key], dict) and 'loss_fn' in overall_results['overall'][method_key]:
                if overall_results['overall'][method_key]['loss_fn'] == method_loss:
                    method_exists = True
                    print(f"Method {method} with loss function {method_loss} already exists in results, skipping...")
            else:
                # For backward compatibility with old format results
                method_exists = True
                print(f"Method {method} already exists in results (old format), skipping...")
        # Special handling for SMART with different loss functions
        elif method_key == 'SMART' and not args.overwrite:
            specific_key = f'SMART_{method_loss}'
            if specific_key in overall_results.get('overall', {}):
                method_exists = True
                print(f"Method SMART with loss function {method_loss} already exists in results under key {specific_key}, skipping...")
        elif method_key == 'TS' and not args.overwrite:
            specific_key = f'TS_{method_loss}'
            if specific_key in overall_results.get('overall', {}):
                method_exists = True
                print(f"Method TS with loss function {method_loss} already exists in results under key {specific_key}, skipping...")
        elif method_key == 'PTS' and not args.overwrite:
            specific_key = f'PTS_{method_loss}'
            if specific_key in overall_results.get('overall', {}):
                method_exists = True
                print(f"Method PTS with loss function {method_loss} already exists in results under key {specific_key}, skipping...")
        elif method_key == 'CTS' and not args.overwrite:
            specific_key = f'CTS_{method_loss}'
            if specific_key in overall_results.get('overall', {}):
                method_exists = True
                print(f"Method CTS with loss function {method_loss} already exists in results under key {specific_key}, skipping...")
        elif method_key == 'ETS' and not args.overwrite:
            specific_key = f'ETS_{method_loss}'
            if specific_key in overall_results.get('overall', {}):
                method_exists = True
                print(f"Method ETS with loss function {method_loss} already exists in results under key {specific_key}, skipping...")
        elif method_key in overall_results.get('overall', {}) and args.overwrite:
            print(f"Method {method} already exists but overwrite=True, will run and overwrite existing results...")
        
        if not method_exists:
            methods_to_run.append(method)
    
    print(f"Methods to run: {methods_to_run}")
    
    # Uncalibrated
    if "uncalibrated" in methods_to_run:
        uncal_probs = F.softmax(test_logits_tensor, dim=1)

        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=test_logits_tensor,
            labels=test_labels_tensor,
            method_name="Uncalibrated",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Add results (using default 15 bins for backward compatibility)
        results['cal'].append('uncalibrated')
        results['ece'].append(all_metrics['ece_15'])
        results['accuracy'].append(all_metrics['accuracy'])
        results['adaece'].append(all_metrics['adaece_15'])
        results['cece'].append(all_metrics['cece_15'])
        results['nll'].append(all_metrics['nll'])
        results['ece_debiased'].append(all_metrics['ece_debiased_15'])
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='uncalibrated',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn='none'
        )
        
        # Add to calibration plots for visualization
        calibration_plots['Uncalibrated'] = uncal_probs.detach().numpy()
    
    # Temperature Scaling 
    if "TS" in methods_to_run:
        print("\nTraining Temperature Scaling...")
        
        # Update args for TS calibration
        args.cal = 'TS'
        if not hasattr(args, 'dataset'):
            args.dataset = dataset_name
        if not hasattr(args, 'device'):
            args.device = device
        args.n_class = dataset_num_classes.get(dataset_name, 1000)

        ts_loss = getattr(args, 'ts_loss', 'CE')
        args.loss = ts_loss
        print(f"Using loss function: {args.loss}")
        
        # Set seed before creating and training TS calibrator
        set_seed(args.random_seed)
        
        # Initialize and train the calibrator
        ts_calibrator = TemperatureScalingCalibrator(
            loss_type=args.loss,
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
        all_metrics = compute_and_print_metrics(
            logits=calibrated_logits,
            labels=test_labels_tensor,
            method_name="Temperature Scaling",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        print(f"Optimal temperature: {optimal_temp:.4f}")
        
        # Save optimal temperature value for visualization
        optimal_temp = float(optimal_temp)
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key=f'TS_{ts_loss}',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn=ts_loss,
            additional_params={'temp': float(optimal_temp)}
        )
        
        # Add to calibration plots for visualization
        calibration_plots['TS'] = ts_probs
    
    # Ensemble Temperature Scaling (ETS)
    if "ETS" in methods_to_run:
        print("\nTraining Ensemble Temperature Scaling...")
        
        # Get number of classes from dataset
        n_classes = dataset_num_classes.get(dataset_name, 1000)
        
        # Set seed before creating and training ETS calibrator
        set_seed(args.random_seed)
        
        # Initialize ETS calibrator
        ets_calibrator = ETSCalibrator(loss_type=ets_loss, n_classes=n_classes)
        
        # Fit the calibrator on validation data
        ets_calibrator.fit(val_logits, val_labels)
        
        # Calibrate test logits
        ets_probs = ets_calibrator.calibrate(test_logits)
        ets_probs_tensor = torch.tensor(ets_probs, dtype=torch.float32)
        
        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=ets_probs_tensor,  # ETS returns probabilities
            labels=test_labels_tensor,
            method_name="Ensemble Temperature Scaling",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        print(f"Optimal temperature: {ets_calibrator.get_temperature():.4f}")
        print(f"Optimal weights: {ets_calibrator.get_weights()}")
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key=f'ETS_{ets_loss}',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn=ets_loss,
            additional_params={
                'temp': float(ets_calibrator.get_temperature()),
                'weights': ets_calibrator.get_weights()
            }
        )
        
        # Add to calibration plots for visualization
        calibration_plots['ETS'] = ets_probs
    
    # SMART
    if "SMART" in methods_to_run:
        print("\nTraining Sample logitsgap Aware Temperature Scaling...")
        # Use SMART loss function from argparse
        smart_loss = getattr(args, 'smart_loss', 'soft_ece')

        print("Training SMART with loss function:", smart_loss)
        smart = SMART(epochs=smart_epochs, dataset_name=dataset_name, 
                     model_name=model_name, seed_value=seed_value,
                     valid_size=valid_size, loss_fn=smart_loss,
                     patience=patience, min_delta=min_delta,
                     corruption_type=corruption_type, severity=severity, 
                     train_loss=train_loss)
        
        # Try to load existing SMART model
        if not smart.load_model():
            # Train new model if loading failed
            smart.fit(val_logits, val_labels)
        
        smart_probs = smart.calibrate(test_logits)
        smart_probs_tensor = torch.tensor(smart_probs, dtype=torch.float32)

        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=smart_probs_tensor,  # SMART returns probabilities
            labels=test_labels_tensor,
            method_name="SMART",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Get logitsgap and temperature values for visualization
        paths = get_logit_paths(dataset_name, model_name, seed_value, 
                              valid_size, smart_loss, corruption_type, severity, train_loss)
        logitsgap_file = paths['test_logitsgap_values']
        
        # Load logitsgap and temperature values for visualization
        if os.path.exists(logitsgap_file) and len(logitsgap_values) == 0:
            # If logitsgap values were not loaded before, load them now
            print(f"Loading cached test logitsgap values from {logitsgap_file}")
            with open(logitsgap_file, "r") as f:
                logitsgap_dict = json.load(f)
            logitsgap_values = logitsgap_dict["logitsgap"] if "logitsgap" in logitsgap_dict else logitsgap_dict["hardness"]
        
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
            bins_list=eval_bins,
            loss_fn=smart_loss
        )
        
        # Add to calibration plots for visualization
        calibration_plots['SMART'] = smart_probs
    
    # Parametric Temperature Scaling (PTS)
    if "PTS" in methods_to_run:
        print("\nTraining Parametric Temperature Scaling...")
        
        # Update args for PTS calibration
        args.cal = 'PTS'
        if not hasattr(args, 'dataset'):
            args.dataset = dataset_name
        if not hasattr(args, 'device'):
            args.device = device
        args.n_class = dataset_num_classes.get(dataset_name, 1000)
        pts_loss = getattr(args, 'pts_loss', 'MSE')
        
        # Initialize PTSCalibrator with overwrite flag and fixed seed
        pts_calibrator = PTSCalibrator(
            steps=10000,
            lr=0.00005,
            nlayers=2,
            n_nodes=5,
            loss_fn=pts_loss,
            top_k_logits=10,
            seed=args.random_seed  # Use the same seed for consistency
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
        all_metrics = compute_and_print_metrics(
            logits=calibrated_logits,
            labels=test_labels_tensor,
            method_name="Parametric Temperature Scaling",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key=f'PTS_{pts_loss}',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn=pts_loss
        )
        
        # Add to calibration plots for visualization
        calibration_plots['PTS'] = pts_probs
    
    # Class-specific Temperature Scaling (CTS)
    if "CTS" in methods_to_run:
        print("\nTraining Class-based Temperature Scaling...")
        
        # Update args for CTS calibration
        args.cal = 'CTS'
        if not hasattr(args, 'dataset'):
            args.dataset = dataset_name
        if not hasattr(args, 'device'):
            args.device = device
        args.n_class = dataset_num_classes.get(dataset_name, 1000)
        cts_loss = getattr(args, 'cts_loss', 'CE')
        args.loss = cts_loss
        print(f"Using loss function: {args.loss}")
        
        # Ensure labels are the correct data type for the loss function
        if cts_loss == 'soft_ece':
            # Convert labels to int64 for soft_ece loss
            val_labels_for_ts = val_labels_tensor.long()
        else:
            val_labels_for_ts = val_labels_tensor

        cts_calibrator = CTSCalibrator(
            n_class=args.n_class,      # Number of classes
            n_bins=15,
            n_iter=5,                 # Number of bins for ECE computation
        ).to(device)
        
        # Set seed before fitting to ensure reproducibility
        set_seed(args.random_seed)
        
        val_logits_device = val_logits_tensor.to(device)
        val_labels_device = val_labels_for_ts.to(device)

        # Fit the calibrator on validation data
        cts_calibrator.fit(val_logits_device, val_labels_device, ts_loss=args.loss)
        
        # Calibrate test logits
        test_logits_device = test_logits_tensor.to(device)
        
        cts_probs = cts_calibrator.calibrate(test_logits_device).cpu().detach().numpy()
        
        # Get calibrated logits for metric calculation
        calibrated_logits = cts_calibrator.calibrate(test_logits_device, return_logits=True)
        
        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=calibrated_logits,
            labels=test_labels_tensor,
            method_name="Class-based Temperature Scaling",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key=f'CTS_{cts_loss}',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn=cts_loss
        )
        
        # Add to calibration plots for visualization
        calibration_plots['CTS'] = cts_probs
    
    # Histogram Binning
    if "HB" in methods_to_run:
        print("\nTraining Histogram Binning...")
        set_seed(args.random_seed)
        
        # Initialize Histogram Binning calibrator
        hb_calibrator = HistogramBinningCalibrator(n_bins=15, strategy='uniform')
        
        # Fit the calibrator on validation data
        hb_calibrator.fit(val_logits_tensor, val_labels_tensor)
        
        # Apply calibration to test set
        hb_logits = hb_calibrator.calibrate(test_logits_tensor, return_logits=True)
        hb_probs = F.softmax(hb_logits, dim=1)
        
        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=hb_logits,
            labels=test_labels_tensor,
            method_name="Histogram Binning",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='HB',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn='uniform_binning'
        )
        
        # Add to calibration plots for visualization
        calibration_plots['HB'] = hb_probs.detach().numpy()
    
    # BBQ (Bayesian Binning into Quantiles)
    if "BBQ" in methods_to_run:
        print("\nTraining BBQ...")
        set_seed(args.random_seed)
        
        # Initialize BBQ calibrator
        bbq_calibrator = BBQCalibrator(score_type='max_prob', n_bins_max=20)
        
        # Fit the calibrator on validation data
        bbq_calibrator.fit(val_logits_tensor, val_labels_tensor)
        
        # Apply calibration to test set
        bbq_logits = bbq_calibrator.calibrate(test_logits_tensor, return_logits=True)
        bbq_probs = F.softmax(bbq_logits, dim=1)
        
        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=bbq_logits,
            labels=test_labels_tensor,
            method_name="BBQ",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='BBQ',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn='bayesian_binning'
        )
        
        # Add to calibration plots for visualization
        calibration_plots['BBQ'] = bbq_probs.detach().numpy()
    
    # Vector Scaling
    if "VS" in methods_to_run:
        print("\nTraining Vector Scaling...")
        set_seed(args.random_seed)
        
        # Initialize Vector Scaling calibrator
        vs_calibrator = VectorScalingCalibrator(loss_type='nll', bias=True)
        
        # Fit the calibrator on validation data
        vs_calibrator.fit(val_logits_tensor, val_labels_tensor)
        
        # Apply calibration to test set
        vs_logits = vs_calibrator.calibrate(test_logits_tensor, return_logits=True)
        vs_probs = F.softmax(vs_logits, dim=1)
        
        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=vs_logits,
            labels=test_labels_tensor,
            method_name="Vector Scaling",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='VS',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn='vector_scaling'
        )
        
        # Add to calibration plots for visualization
        calibration_plots['VS'] = vs_probs.detach().numpy()
    
    # Group Calibration
    if "GC" in methods_to_run:
        print("\nTraining Group Calibration...")
        set_seed(args.random_seed)
        
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
        all_metrics = compute_and_print_metrics(
            logits=gc_logits,
            labels=test_labels_tensor,
            method_name="Group Calibration",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='GC',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn='group_calibration'
        )
        
        # Add to calibration plots for visualization
        calibration_plots['GC'] = gc_probs.detach().numpy()
    
    # ProCal Density-Ratio Calibration
    if "ProCal_DR" in methods_to_run:
        print("\nTraining ProCal Density-Ratio Calibration...")
        set_seed(args.random_seed)
        
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
        all_metrics = compute_and_print_metrics(
            logits=procal_dr_probs,
            labels=test_labels_tensor,
            method_name="ProCal Density-Ratio",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='ProCal_DR',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn='density_ratio'
        )
        
        # Add to calibration plots for visualization
        calibration_plots['ProCal_DR'] = procal_dr_probs.detach().numpy()
    
    # ProCal Bin-Mean-Shift Calibration
    if "ProCal_BMS" in methods_to_run:
        print("\nTraining ProCal Bin-Mean-Shift Calibration...")
        set_seed(args.random_seed)
        
        from sklearn.isotonic import IsotonicRegression
        
        # Initialize ProCal Bin-Mean-Shift calibrator
        procal_bms_calibrator = ProCalBinMeanShiftCalibrator(
            base_calibrator_class=IsotonicRegression,
            k_neighbors=10,
            proximity_bins=10,
            bin_strategy='quantile',
            distance_measure='L2',
            normalize_features=True,
            out_of_bounds='clip'  # Parameter for IsotonicRegression
        )
        
        # Fit the calibrator on validation data with features
        procal_bms_calibrator.fit(val_logits_tensor, val_labels_tensor, val_features_tensor)
        
        # Apply calibration to test set with features
        procal_bms_probs = procal_bms_calibrator.calibrate(test_logits_tensor, test_features_tensor)
        
        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=procal_bms_probs,
            labels=test_labels_tensor,
            method_name="ProCal Bin-Mean-Shift",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='ProCal_BMS',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn='bin_mean_shift'
        )
        
        # Add to calibration plots for visualization
        calibration_plots['ProCal_BMS'] = procal_bms_probs.detach().numpy()
    
    # Feature Clipping Calibration
    if "FC" in methods_to_run:
        print("\nTraining Feature Clipping Calibration...")
        set_seed(args.random_seed)

        # Create the same model that was used to extract features
        model = create_model(args, model_name, dataset_name, device)
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
            val_features_tensor, val_logits_tensor, val_labels_tensor, classifier_fn
        )
        
        print(f"Optimal clipping parameter: {optimal_clip:.4f}")
        
        # Apply feature clipping to test features and get calibrated logits
        clipped_test_features = fc_calibrator.feature_clipping(test_features_tensor, optimal_clip)
        fc_logits = classifier_fn(clipped_test_features)
        fc_probs = F.softmax(fc_logits, dim=1)
        
        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=fc_logits,
            labels=test_labels_tensor,
            method_name="Feature Clipping",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )
        
        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='FC',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn='feature_clipping'
        )
        
        # Add to calibration plots for visualization
        calibration_plots['FC'] = fc_probs.detach().cpu().numpy()

    # Logit Clipping Calibration
    if "LC" in methods_to_run:
        print("\nTraining Logit Clipping Calibration...")
        set_seed(args.random_seed)

        # Initialize Logit Clipping calibrator
        lc_calibrator = LogitClippingCalibrator()

        # Fit the calibrator on validation data using ECE cross-validation
        optimal_clip = lc_calibrator.fit(val_logits_tensor, val_labels_tensor, cross_validate='ece')

        print(f"Optimal clipping parameter: {optimal_clip:.4f}")

        # Apply calibration to test set
        lc_probs = lc_calibrator.calibrate(test_logits_tensor, return_logits=False)
        lc_logits = lc_calibrator.calibrate(test_logits_tensor, return_logits=True)

        # Compute and print metrics
        all_metrics = compute_and_print_metrics(
            logits=lc_logits,
            labels=test_labels_tensor,
            method_name="Logit Clipping",
            bins_list=eval_bins,
            device=device,
            enabled_metrics=eval_metrics
        )

        # Store results
        store_method_results(
            overall_results=overall_results,
            method_key='LC',
            all_metrics=all_metrics,
            bins_list=eval_bins,
            loss_fn='logit_clipping',
            additional_params={'clip_value': float(optimal_clip)}
        )

        # Add to calibration plots for visualization
        calibration_plots['LC'] = lc_probs.detach().cpu().numpy()

    # Density Aware Calibration (DAC)
    if "DAC" in methods_to_run:
        print("\nTraining Density Aware Calibration...")

        # Get the base calibration method from args
        dac_base_method = getattr(args, 'dac_base_method', 'TS')
        print(f"Using DAC with base method: {dac_base_method}")

        set_seed(args.random_seed)

        try:
            # Create base calibrator based on specified method (TS, PTS, or SMART supported)
            base_calibrator = None
            if dac_base_method == 'TS':
                ts_loss = getattr(args, 'ts_loss', 'CE')
                base_calibrator = TemperatureScalingCalibrator(loss_type=ts_loss)
            elif dac_base_method == 'PTS':
                pts_loss = getattr(args, 'pts_loss', 'MSE')
                base_calibrator = PTSCalibrator(
                    steps=10000, lr=0.00005, nlayers=2, n_nodes=5,
                    loss_fn=pts_loss, top_k_logits=10, seed=args.random_seed
                ).to(device)
            elif dac_base_method == 'SMART':
                smart_loss = getattr(args, 'smart_loss', 'smooth_soft_ece')
                print(f"Creating SMART base calibrator with loss function: {smart_loss}")

                # Extract corruption parameters if available
                corruption_type = getattr(args, 'corruption_type', None) if dataset_name == 'imagenet_c' else None
                severity = getattr(args, 'severity', None) if dataset_name == 'imagenet_c' else None
                train_loss = getattr(args, 'train_loss', None)

                base_calibrator = SMART(
                    epochs=smart_epochs,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed_value=args.random_seed,
                    valid_size=valid_size,
                    loss_fn=smart_loss,
                    patience=patience,
                    min_delta=min_delta,
                    corruption_type=corruption_type,
                    severity=severity,
                    train_loss=train_loss
                )
            else:
                raise ValueError(f"Unsupported DAC base method: {dac_base_method}. Supported methods: TS, PTS, SMART")



            # Determine loss type for DAC optimization based on base calibrator
            if dac_base_method == 'TS':
                dac_loss_type = 'ce'
            elif dac_base_method == 'PTS':
                dac_loss_type = 'mse'
            elif dac_base_method == 'SMART':
                # For SMART, use MSE as it works well with its optimization
                dac_loss_type = 'mse'
            else:
                dac_loss_type = 'ce'  # default
            dac_calibrator = DensityAwareCalibrator(
                ood_values_num=1,         # TODO: Should be 6 for multi-layer (requires model changes)
                loss_type=dac_loss_type,  # Match base calibrator
                knn_k=10,                # Paper specification for ImageNet (was 50)
                avg_top_k=False,         # default
                gpu=False,               # use CPU to avoid compatibility issues
                base_calibrator=base_calibrator
            )

            # Convert features to tensors if they're numpy arrays
            val_features_tensor = torch.tensor(val_features, dtype=torch.float32) if isinstance(val_features, np.ndarray) else val_features
            test_features_tensor = torch.tensor(test_features, dtype=torch.float32) if isinstance(test_features, np.ndarray) else test_features

            # CRITICAL FIX: Extract actual training features for KNN density estimation
            # The original code incorrectly used validation features as training features,
            # which fundamentally breaks the density estimation (each sample becomes its own nearest neighbor)
            print("Extracting training features for DAC KNN density estimation...")

            # Create model and train_loader for feature extraction
            dac_model = create_model(args, model_name, dataset_name, device)
            dac_model.eval()
            train_loader = create_train_loader_for_dac(args, dataset_name)

            # Extract reference features for KNN density estimation
            # For ImageNet-C: using original uncorrupted val set (50k images)
            # Extract 10k samples (~20% of val set, or ~0.8% of full training set)
            max_samples = 10000 if dataset_name.startswith('imagenet') else 5000
            train_features_tensor = extract_train_features_for_dac(
                dac_model, train_loader, device, max_samples=max_samples
            )
            print(f"Extracted {train_features_tensor.shape[0]} reference features for DAC KNN")

            # Clean up
            del dac_model
            torch.cuda.empty_cache()

            # Fit the DAC calibrator
            dac_result = dac_calibrator.fit(
                val_logits=val_logits_tensor,
                val_labels=val_labels_tensor,
                val_features=val_features_tensor,
                train_features=train_features_tensor  # Use actual training features
            )

            print(f"DAC fitting completed. Result: {dac_result}")

            # Apply calibration to test set
            dac_probs = dac_calibrator.calibrate(
                test_logits=test_logits_tensor,
                test_features=test_features_tensor,
                return_logits=False
            )

            dac_logits = dac_calibrator.calibrate(
                test_logits=test_logits_tensor,
                test_features=test_features_tensor,
                return_logits=True
            )

            # Compute and print metrics
            all_metrics = compute_and_print_metrics(
                logits=dac_logits,
                labels=test_labels_tensor,
                method_name=f"DAC + {dac_base_method}",
                bins_list=eval_bins,
                device=device,
                enabled_metrics=eval_metrics
            )

            # Store results
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(dac_result, dict):
                dac_weights = dac_result['dac_weights']
                if hasattr(dac_weights, 'tolist'):
                    dac_weights = dac_weights.tolist()
                base_result = dac_result.get('base_result', None)
            else:
                dac_weights = dac_result.tolist() if hasattr(dac_result, 'tolist') else dac_result
                base_result = None

            store_method_results(
                overall_results=overall_results,
                method_key=f'DAC_{dac_base_method}',
                all_metrics=all_metrics,
                bins_list=eval_bins,
                loss_fn=f'dac_{dac_base_method.lower()}',
                additional_params={
                    'dac_weights': dac_weights,
                    'base_result': base_result,
                    'base_method': dac_base_method
                }
            )

            # Add to calibration plots for visualization
            calibration_plots[f'DAC_{dac_base_method}'] = dac_probs.detach().cpu().numpy()

        except ImportError as e:
            print(f"Warning: Could not use DAC due to missing dependencies: {e}")
            print("Please install faiss-cpu or faiss-gpu to use Density Aware Calibration")
        except Exception as e:
            print(f"Error in DAC calibration: {e}")
            import traceback
            traceback.print_exc()

    # Save the updated results
    with open(overall_results_file, "w") as f:
        json.dump(overall_results, f, indent=4)
    
    print(f"Saved updated results to {overall_results_file}")
    
    # Generate visualizations
    generate_visualizations(test_logits, test_labels, calibration_plots, logitsgap_values, 
                           temperatures, optimal_temp, methods_to_run, model_name, plot_dir)
    
    return overall_results


