import sys
import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

# Import new loss functions for SMART
from calibrator.Component.metrics import (
    BrierLoss, FocalLoss, LabelSmoothingLoss, 
    CrossEntropyLoss, MSELoss, SoftECE, ECE
)


# Utils imports
from utils.utils import (
    get_model_and_logits,
    compute_and_print_metrics,
    store_method_results,
    evaluate_calibration_methods,
    get_logit_paths,
    BINS_LIST,
    AVAILABLE_METRICS,
    DEFAULT_METRICS
)

# SMART calibration imports
from utils.smart_calibrator import (
    soft_tail_proxy_g_tau,
    participation_ratio_k_eff,
    set_seed,
    compute_logitsgap
)

# Import models dictionary from utils
from utils import models_dict, dataset_loader, dataset_num_classes



def main():
    # Set batch size based on GPU memory
    batch_size = 1024
    
    # Get available models for each dataset
    available_models = {dataset: list(models.keys()) for dataset, models in models_dict.items()}
    all_models = set()
    for models_list in available_models.values():
        all_models.update(models_list)
    all_models = sorted(list(all_models))
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run smart on ImageNet and other datasets')
    parser.add_argument('--model', type=str, default='resnet50', 
                      choices=all_models,
                      help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                      help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--smart_epochs', type=int, default=2000,
                      help='Number of epochs for SMART training')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                      help='Use CUDA (GPU) for computation')
    parser.add_argument('--dataset', type=str, default='imagenet',
                      choices=list(dataset_loader.keys()),
                      help='Dataset name')
    parser.add_argument("--train_loss", type=str, default='cross_entropy',
                      help='Training loss type for loading pre-trained weights (only used for CIFAR models)')
    parser.add_argument("--use_underfitted", action='store_true', default=False,
                      help='Use underfitted models (5 or 10 epochs) for evaluation')
    parser.add_argument("--underfitted_epochs", type=int, default=5, choices=[5, 10],
                      help='Number of epochs for underfitted models (5 or 10)')
    parser.add_argument("--ts_loss", type=str, default='CE',
                      choices=['CE', 'LS', 'brier', 'soft_ece', 'weighted_soft_ece', 'smooth_soft_ece', 'gap_indexed_soft_ece', 'MSE'],
                      help='Loss function for Temperature Scaling')
    parser.add_argument("--pts_loss", type=str, default='CE',
                      choices=['CE', 'LS', 'brier', 'soft_ece', 'weighted_soft_ece', 'smooth_soft_ece', 'gap_indexed_soft_ece', 'MSE'],
                      help='Loss function for Parametric Temperature Scaling')
    parser.add_argument("--cts_loss", type=str, default='CE',
                      choices=['CE', 'LS', 'brier', 'soft_ece', 'weighted_soft_ece', 'smooth_soft_ece', 'gap_indexed_soft_ece', 'MSE'],
                      help='Loss function for Class-based Temperature Scaling')
    parser.add_argument("--ets_loss", type=str, default='mse',
                      choices=['mse', 'ce'],
                      help='Loss function for Ensemble Temperature Scaling')
    parser.add_argument("--smart_loss", type=str, default='smooth_soft_ece',
                      choices=['CE', 'LS', 'brier', 'soft_ece', 'weighted_soft_ece', 'smooth_soft_ece', 'gap_indexed_soft_ece', 'MSE'],
                      help='Loss function for SMART')
    parser.add_argument("--dac_base_method", type=str, default='TS',
                      choices=['TS', 'PTS', 'SMART'],
                      help='Base calibration method to combine with DAC (Density Aware Calibration)')
    parser.add_argument("--dataset_root", type=str, default='/hdd/haolan/datasets',
                      help='Root directory for datasets')
    parser.add_argument("--train_batch_size", type=int, default=128,
                      help='Batch size for training')
    parser.add_argument("--test_batch_size", type=int, default=128,
                      help='Batch size for testing')
    parser.add_argument("--valid_size", type=float, default=0.2,
                      help='Proportion of validation set (when splitting test set)')
    parser.add_argument("--random_seed", type=int, default=1,
                      help='Random seed for dataset splitting')
    parser.add_argument("--corruption_type", type=str, default='gaussian_noise',
                      help='Corruption type for ImageNet-C dataset')
    parser.add_argument("--severity", type=int, default=5, choices=[1, 2, 3, 4, 5],
                      help='Corruption severity level for ImageNet-C dataset')
    parser.add_argument('--patience', type=int, default=200,
                      help='Early stopping patience (number of epochs without improvement)')
    parser.add_argument('--min_delta', type=float, default=0.0001,
                      help='Minimum change in loss to qualify as improvement for early stopping')
    parser.add_argument('--disable_early_stopping', action='store_true',
                      help='Disable early stopping (run for full number of epochs)')
    parser.add_argument('--run_methods', type=str, default="uncalibrated,TS,PTS,CTS,ETS,SMART,HB,BBQ,VS,GC,ProCal_DR,FC,LC,DAC",
                      help='Comma-separated list of calibration methods to run, choose from: uncalibrated,TS,PTS,CTS,ETS,SMART,HB,BBQ,VS,GC,ProCal_DR,ProCal_BMS,FC,LC,DAC')
    parser.add_argument('--overwrite', action='store_true', default=False,
                      help='Overwrite existing results')
    parser.add_argument('--eval_metrics', type=str, default="ece,adaece,cece,nll,accuracy,rbs",
                      help=f'Comma-separated list of evaluation metrics to compute and display. Available: {",".join(AVAILABLE_METRICS)}')
    parser.add_argument('--eval_bins', type=str, default="5,10,15,20,25,30",
                      help='Comma-separated list of bin sizes for ECE evaluation')

    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.random_seed)
    
    # Process run_methods argument
    run_methods = args.run_methods.split(',')
    print(f"Running calibration methods: {', '.join(run_methods)}")
    
    # Process eval_metrics argument
    eval_metrics = [m.strip() for m in args.eval_metrics.split(',')]
    invalid_metrics = [m for m in eval_metrics if m not in AVAILABLE_METRICS]
    if invalid_metrics:
        print(f"Warning: Invalid metrics {invalid_metrics}. Available: {AVAILABLE_METRICS}")
        eval_metrics = [m for m in eval_metrics if m in AVAILABLE_METRICS]
    print(f"Evaluation metrics: {', '.join(eval_metrics)}")
    
    # Process eval_bins argument
    try:
        eval_bins = [int(b.strip()) for b in args.eval_bins.split(',')]
        print(f"Evaluation bin sizes: {eval_bins}")
    except ValueError:
        print(f"Warning: Invalid bin sizes in '{args.eval_bins}', using default")
        eval_bins = BINS_LIST
    
    # Verify model availability for the selected dataset
    if args.model not in models_dict.get(args.dataset, {}):
        print(f"Warning: Model {args.model} might not be available for dataset {args.dataset}")
        print(f"Available models for {args.dataset}: {available_models.get(args.dataset, ['none'])}")
    
    print(f"Starting {args.dataset} calibration experiment with selected methods")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.random_seed}")
    print(f"Valid size: {args.valid_size}")
    if args.dataset.startswith('cifar'):
        print(f"Loss function: {args.train_loss}") 
    
    # Get model and compute logits (or load existing ones)
    val_logits, val_labels, test_logits, test_labels, val_features, test_features = get_model_and_logits(
        args=args,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cuda=args.use_cuda,
        dataset_name=args.dataset,
        seed_value=args.random_seed,
        valid_size=args.valid_size,
        loss_fn=args.train_loss
    )
    
    # Set up early stopping parameters
    patience = args.patience if not args.disable_early_stopping else float('inf')
    min_delta = args.min_delta
    if args.disable_early_stopping:
        print("Early stopping disabled - will run for full number of epochs")
    else:
        print(f"Early stopping enabled with patience={patience}, min_delta={min_delta}")
    
    # Evaluate calibration methods
    evaluate_calibration_methods(
        val_logits, val_labels, test_logits, test_labels, val_features, test_features,
        args=args,
        batch_size=args.batch_size,
        smart_epochs=args.smart_epochs,
        dataset_name=args.dataset,
        model_name=args.model,
        seed_value=args.random_seed,
        valid_size=args.valid_size,
        loss_fn=args.smart_loss, 
        patience=patience,
        min_delta=min_delta,
        run_methods=run_methods,
        eval_metrics=eval_metrics,
        eval_bins=eval_bins
    )

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time/60:.2f} minutes") 
