import random
import numpy as np
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
from typing import Dict, Optional, Union
import os
import torch
import logging
import json
from Component.metrics import *
from bench.utils import models_dict


def validate_dataset_and_model(dataset_name, model_name):
    if dataset_name not in models_dict:
        available_datasets = ", ".join(models_dict.keys())
        raise ValueError(
            f"❌ Dataset '{dataset_name}' not found! "
            f"Available datasets are: [{available_datasets}]"
        )

    if model_name not in models_dict[dataset_name]:
        available_models = ", ".join(models_dict[dataset_name].keys())
        raise ValueError(
            f"❌ Model '{model_name}' is not supported for dataset '{dataset_name}'. "
            f"Supported models for this dataset are: [{available_models}]"
        )

def compute_all_metrics(
        device,
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
        'ece': ECE(n_bins=n_bins).to(device),
        'accuracy': Accuracy().to(device),
        'adaptive_ece': AdaptiveECE(n_bins=n_bins).to(device),
        'classwise_ece': ClasswiseECE(n_bins=n_bins).to(device),
        'nll': NLL().to(device),
        'ece_debiased': ECEDebiased(n_bins=n_bins).to(device),
        'ece_sweep': ECESweep().to(device),
        'Brier': BrierLoss().to(device),
        'rbs': RBS().to(device)
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
        device,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        n_bins: int = 15,

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
    metrics = compute_all_metrics(device,labels=labels, logits=logits if not is_probs else None,
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
