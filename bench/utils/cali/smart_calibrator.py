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

from bench.utils.data_utils import get_logit_paths

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

# Metrics imports
from Component.metrics import (
    BrierLoss, FocalLoss, LabelSmoothingLoss,
    CrossEntropyLoss, MSELoss, SoftECE, ECE, AdaptiveECE, ClasswiseECE,
    Accuracy, NLL, KDEECE, ECEDebiased, ECESweep
)
from Component.metrics.WeightedSoftECE import WeightedSoftECE
from Component.metrics.SmoothSoftECE import SmoothSoftECE
from Component.metrics.GapIndexedSoftECE import GapIndexedSoftECE


# BINS = [5, 10, 15, 20, 25, 30]

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def soft_tail_proxy_g_tau(logits: torch.Tensor, tau: float = 5.0) -> torch.Tensor:
    """
    g_tau = z_(1) - (1/tau) * logsumexp(tau * z_(j), j>=2), per-sample.
    Stable form: g_tau = -(1/tau) * logsumexp(tau * (z_tail - z1)).
    Returns [B], larger => stronger separation of winner vs competitors.
    """
    z_sorted, _ = torch.sort(logits, dim=1, descending=True)  # [B, C]
    z1 = z_sorted[:, :1]                                      # [B, 1]
    tail = z_sorted[:, 1:]                                    # [B, C-1]
    u = tau * (tail - z1)                                     # shift by z1 to avoid overflow
    g_tau = - torch.logsumexp(u, dim=1) / tau                 # [B]
    return g_tau


@torch.no_grad()
def participation_ratio_k_eff(logits: torch.Tensor, tau: float = 5.0) -> torch.Tensor:
    """
    k_eff = exp( 2*LSE(u) - LSE(2u) ), u = tau * (z_tail - z1), tail j>=2.
    Numerically stable; returns [B] in [1, K-1].
    """
    z_sorted, _ = torch.sort(logits, dim=1, descending=True)
    z1 = z_sorted[:, :1]
    tail = z_sorted[:, 1:]
    u = tau * (tail - z1)                    # shift by top logit
    lse_u = torch.logsumexp(u, dim=1)        # [B]
    lse_2u = torch.logsumexp(2.0 * u, dim=1) # [B]
    k_eff = torch.exp(2.0 * lse_u - lse_2u)
    return k_eff


def compute_logitsgap(logits):
    """
    Compute sample logitsgap as the gap between top logit and runner-up

    Args:
        logits: Tensor of shape [1, num_classes] containing the model's logits
        attack_steps: Not used, kept for API compatibility
        initial_lr: Not used, kept for API compatibility
        min_lr: Not used, kept for API compatibility

    Returns:
        Scalar logitsgap value (logit gap)
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float32, requires_grad=True)

    # Calculate logit gap (margin between predicted class and runner-up)
    sorted_logits, _ = torch.sort(logits_tensor, descending=True)
    logit_gap = (sorted_logits[0][0] - sorted_logits[0][1]).item()

    return logit_gap



# class logitsgapToTemp(nn.Module):
#     """
#     Neural network that maps sample logitsgap to optimal temperature.
#     """
#     def __init__(self, hidden_dim=16, nlayers=2):
#         super(logitsgapToTemp, self).__init__()

#         layers = []
#         input_dim = 1  # logitsgap is a scalar

#         for i in range(nlayers):
#             if i == 0:
#                 layers.append(nn.Linear(input_dim, hidden_dim))
#             else:
#                 layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())

#         # Output layer to predict temperature (should be > 0)
#         layers.append(nn.Linear(hidden_dim, 1))
#         layers.append(nn.Softplus())  # Ensures positive output

#         self.network = nn.Sequential(*layers)

#     def forward(self, logitsgap):
#         """
#         Args:
#             logitsgap: Tensor of shape [B] containing logitsgap values

#         Returns:
#             temperatures: Tensor of shape [B] containing predicted temperatures
#         """
#         # Reshape logitsgap to [B, 1] for the linear layer
#         logitsgap = logitsgap.view(-1, 1)
#         return self.network(logitsgap).squeeze()

# Define the logitsgap to Temperature mapping network
class logitsgapToTemp(nn.Module):
    """
    Neural network that maps sample logitsgap to optimal temperature.
    """
    def __init__(self, additional_features=0, hidden_dim=16, nlayers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(1 + additional_features, hidden_dim))  # Input: logitsgap + any additional features
        
        for _ in range(nlayers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.layers.append(nn.Linear(hidden_dim, 1))  # Output: temperature
        
    def forward(self, logitsgap, additional_features=None):
        """
        Map logitsgap (and potentially other features) to temperature.
        
        Args:
            logitsgap: Tensor of shape [batch_size]
            additional_features: Optional tensor with additional features
            
        Returns:
            Tensor of temperatures, shape [batch_size]
        """
        if additional_features is not None:
            x = torch.cat([logitsgap.view(-1, 1), additional_features], dim=1)
        else:
            x = logitsgap.view(-1, 1)
            
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        # Ensure temperature is positive
        return F.softplus(self.layers[-1](x)) + 0.1


class SMART:
    """
    Sample logitsgap Aware Temperature Scaling (smart).
    Maps sample logitsgap to a specific temperature for each sample.
    """
    def __init__(self, lr=0.005, epochs=200, hidden_dim=16, nlayers=2,
                dataset_name='imagenet', model_name='resnet50', seed_value=1,
                patience=20, min_delta=0.0001, valid_size=0.2, loss_fn='CE',
                corruption_type=None, severity=None, train_loss=None, normalize_logitsgap=True):
        # Set seed for reproducibility
        set_seed(seed_value)

        self.temp_model = logitsgapToTemp(hidden_dim=hidden_dim, nlayers=nlayers)
        self.optimizer = optim.Adam(self.temp_model.parameters(), lr=lr)
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.seed_value = seed_value
        self.valid_size = valid_size
        self.loss_fn_type = loss_fn if isinstance(loss_fn, str) else None
        self.loss_fn = self._get_loss_function(loss_fn)
        self.corruption_type = corruption_type
        self.severity = severity
        self.train_loss = train_loss  # Add train_loss parameter
        self.normalize_logitsgap = normalize_logitsgap  # Control normalization
        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta

    def _get_loss_function(self, loss_fn):
        """
        Get the appropriate loss function based on the input.

        Args:
            loss_fn (str or callable): Loss function specification

        Returns:
            callable: Loss function
        """
        if loss_fn is None:
            return SoftECE()
        elif isinstance(loss_fn, str):
            loss_fn_lower = loss_fn.lower()
            if loss_fn_lower in {"mse", "mean_squared_error"}:
                return MSELoss()
            elif loss_fn_lower in {"crossentropy", "cross_entropy", "ce"}:
                return CrossEntropyLoss()
            elif loss_fn_lower in {"l1", "l1loss", "mean_absolute_error"}:
                return nn.L1Loss()
            elif loss_fn_lower in {"soft_ece", "softece"}:
                return SoftECE()
            elif loss_fn_lower in {"weighted_soft_ece", "weightedsoftece"}:
                return WeightedSoftECE()
            elif loss_fn_lower in {"smooth_soft_ece", "smoothsoftece"}:
                return SmoothSoftECE(mode="softabs")
            elif loss_fn_lower in {"gap_indexed_soft_ece", "gapindexedsoftece"}:
                return GapIndexedSoftECE()
            elif loss_fn_lower in {"brier", "brier_score"}:
                return BrierLoss()
            elif loss_fn_lower in {"focal", "focal_loss", "fl"}:
                return FocalLoss()
            elif loss_fn_lower in {"label_smoothing", "ls"}:
                return LabelSmoothingLoss(alpha=0.005)
            elif loss_fn_lower in {"ece"}:
                return ECE(n_bins=15)
            else:
                raise ValueError(f"Unsupported loss function: {loss_fn}")
        else:
            # If loss_fn is a callable, then use it directly.
            return loss_fn

    def fit(self, logits, labels):
        """
        Train the logitsgap-to-temperature mapping

        Args:
            logits: numpy array of shape [n_samples, n_classes] with model logits
            labels: numpy array of shape [n_samples] with ground truth labels

        Returns:
            None if model was loaded from cache, or training loss if model was trained
        """
        # Try to load existing model first
        if self.load_model():
            print("Loaded existing SMART model from cache. Skipping training.")
            return None

        print("No cached SMART model found. Training new model...")

        # Reset seed before training
        set_seed(self.seed_value)

        # Get paths based on parameters
        paths = get_logit_paths(self.dataset_name, self.model_name, self.seed_value,
                               self.valid_size, self.loss_fn_type,
                               self.corruption_type, self.severity, self.train_loss)
        logitsgap_file = paths['logitsgap_values']
        smart_model_file = paths['smart_model']
        
        # Check if logitsgap values already exist
        if os.path.exists(logitsgap_file):
            print(f"Loading cached logitsgap values from {logitsgap_file}")
            with open(logitsgap_file, "r") as f:
                logitsgap_dict = json.load(f)
            logitsgap_values = logitsgap_dict["logitsgap"] if "logitsgap" in logitsgap_dict else logitsgap_dict["hardness"]
        else:
            # Compute logitsgap for each validation sample
            print("Computing logitsgap values for each sample...")
            logitsgap_values = []
            
            for i in tqdm(range(len(logits))):
                logitsgap = compute_logitsgap(logits[i:i+1])
                logitsgap_values.append(logitsgap)
            
            
            # Save logitsgap values to avoid recomputation
            logitsgap_dict = {
                "logitsgap": logitsgap_values
            }
            os.makedirs(os.path.dirname(logitsgap_file), exist_ok=True)
            with open(logitsgap_file, "w") as f:
                json.dump(logitsgap_dict, f)
        
        # Normalize logitsgap values for stable training (if enabled)
        logitsgap_tensor = torch.tensor(logitsgap_values, dtype=torch.float32)
        if self.normalize_logitsgap:
            self.logitsgap_mean = logitsgap_tensor.mean().item()
            self.logitsgap_std = logitsgap_tensor.std().item()
            logitsgap_tensor = (logitsgap_tensor - self.logitsgap_mean) / (self.logitsgap_std + 1e-8)
        else:
            self.logitsgap_mean = 0.0
            self.logitsgap_std = 1.0
        
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Early stopping variables
        best_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        early_stop = False
        
        # Training loop
        print(f"Training logitsgap-to-temperature mapping with {self.loss_fn_type} loss...")
        print(f"Early stopping with patience={self.patience}, min_delta={self.min_delta}")
        
        for epoch in tqdm(range(self.epochs)):
            self.temp_model.train()
            
            # Forward pass: get temperatures for each sample
            temps = self.temp_model(logitsgap_tensor)
            
            # Apply sample-specific temperature scaling
            batch_size = logits_tensor.size(0)
            scaled_logits = logits_tensor / temps.view(batch_size, 1)
            
            # Compute loss using the selected loss function
            # loss = self.loss_fn(scaled_logits, labels_tensor)
            probs = F.softmax(scaled_logits, dim=1) 

            if isinstance(self.loss_fn, SoftECE):
                loss = self.loss_fn(logits=scaled_logits, labels=labels_tensor)

            elif isinstance(self.loss_fn, BrierLoss):
                loss = self.loss_fn(logits=scaled_logits, labels=labels_tensor)

            elif isinstance(self.loss_fn, (FocalLoss, LabelSmoothingLoss)):
                # 这两个希望得到概率
                loss = self.loss_fn(softmaxes=probs, labels=labels_tensor)

            elif isinstance(self.loss_fn, CrossEntropyLoss):
                # CE 需要 logits + class index
                loss = self.loss_fn(scaled_logits, labels_tensor)

            elif isinstance(self.loss_fn, MSELoss):
                # MSE
                loss = self.loss_fn(scaled_logits, labels_tensor)

            else:                                    # 其它自定义 callable
                loss = self.loss_fn(scaled_logits, labels_tensor)
                
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Check for improvement for early stopping
            current_loss = loss.item()
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {current_loss:.4f}")
            
            # Check if this is the best loss so far
            if current_loss < best_loss - self.min_delta:
                best_loss = current_loss
                best_model_state = self.temp_model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            # Check for early stopping
            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best loss: {best_loss:.6f}")
                early_stop = True
                break
        
        # Restore best model if early stopping occurred
        if early_stop and best_model_state is not None:
            self.temp_model.load_state_dict(best_model_state)
            print(f"Restored model to best state with loss: {best_loss:.6f}")
        
        # Save the trained model
        torch.save({
            'model_state_dict': self.temp_model.state_dict(),
            'logitsgap_mean': self.logitsgap_mean,
            'logitsgap_std': self.logitsgap_std,
            'best_loss': best_loss,
            'epochs_trained': epoch + 1 if early_stop else self.epochs,
            'early_stopped': early_stop,
            'loss_fn': self.loss_fn_type,
            'valid_size': self.valid_size,
            'train_loss': self.train_loss  # Save train_loss in checkpoint
        }, smart_model_file)
        print(f"Saved smart model to {smart_model_file}")
    
    
    def load_model(self):
        """Load a previously trained smart model"""

        paths = get_logit_paths(self.dataset_name, self.model_name, self.seed_value, 
                              self.valid_size, self.loss_fn_type,
                              self.corruption_type, self.severity, self.train_loss)
        smart_model_file = paths['smart_model']
        
        if os.path.exists(smart_model_file):
            checkpoint = torch.load(smart_model_file)
            self.temp_model.load_state_dict(checkpoint['model_state_dict'])
            self.logitsgap_mean = checkpoint['logitsgap_mean']
            self.logitsgap_std = checkpoint['logitsgap_std']
            
            # Print additional info if available in the checkpoint
            if 'early_stopped' in checkpoint:
                early_stopped = checkpoint['early_stopped']
                epochs_trained = checkpoint.get('epochs_trained', 'unknown')
                best_loss = checkpoint.get('best_loss', 'unknown')
                loaded_loss_fn = checkpoint.get('loss_fn', 'unknown')
                loaded_valid_size = checkpoint.get('valid_size', 'unknown')
                loaded_train_loss = checkpoint.get('train_loss', 'unknown')
                print(f"Model was{'' if early_stopped else ' not'} early stopped after {epochs_trained} epochs. Best loss: {best_loss}")
                print(f"Model trained with loss function: {loaded_loss_fn}, valid_size: {loaded_valid_size}, train_loss: {loaded_train_loss}")
            
            dataset_info = f"{self.dataset_name}"
            if self.dataset_name == 'imagenet_c' and self.corruption_type is not None and self.severity is not None:
                dataset_info = f"{self.dataset_name} (corruption: {self.corruption_type}, severity: {self.severity})"
            print(f"Loaded smart model from {smart_model_file} for {dataset_info}")
            return True
        else:
            print(f"smart model file {smart_model_file} not found")
            return False
    
    def calibrate(self, logits, return_logits=False):
        """
        Calibrate logits using the learned logitsgap-to-temperature mapping

        Args:
            logits: numpy array or torch.Tensor of shape [n_samples, n_classes] with model logits
            return_logits: if True, return calibrated logits instead of probabilities

        Returns:
            Calibrated probabilities (if return_logits=False) or calibrated logits (if return_logits=True)
            - If input is torch.Tensor: returns torch.Tensor on same device
            - If input is numpy array: returns numpy array
        """
        # Get paths based on parameters
        paths = get_logit_paths(self.dataset_name, self.model_name, self.seed_value,
                              self.valid_size, self.loss_fn_type,
                              self.corruption_type, self.severity, self.train_loss)
        cache_dir = os.path.dirname(paths['logitsgap_values'])
        logitsgap_test_file = paths['test_logitsgap_values']

        # Check if test logitsgap values already exist AND match the input size
        if os.path.exists(logitsgap_test_file):
            with open(logitsgap_test_file, "r") as f:
                logitsgap_dict = json.load(f)
            cached_logitsgap = logitsgap_dict["logitsgap"] if "logitsgap" in logitsgap_dict else logitsgap_dict["hardness"]
            
            # Only use cached values if the size matches
            if len(cached_logitsgap) == len(logits):
                print(f"Loading cached test logitsgap values from {logitsgap_test_file}")
                logitsgap_values = cached_logitsgap
            else:
                # Size mismatch - compute logitsgap for these samples
                print(f"Cached logitsgap size ({len(cached_logitsgap)}) doesn't match input size ({len(logits)}). Computing logitsgap...")
                logitsgap_values = []
                for i in tqdm(range(len(logits))):
                    logitsgap = compute_logitsgap(logits[i:i+1])
                    logitsgap_values.append(logitsgap)
        else:
            # Compute logitsgap for test samples
            print("Computing logitsgap for test samples...")
            logitsgap_values = []

            for i in tqdm(range(len(logits))):
                logitsgap = compute_logitsgap(logits[i:i+1])
                logitsgap_values.append(logitsgap)

            # Save test logitsgap values (only if this looks like the test set)
            # We can identify test set by checking common test set sizes
            if len(logits) > 20000:  # Likely test set
                logitsgap_dict = {
                    "logitsgap": logitsgap_values
                }
                os.makedirs(os.path.dirname(logitsgap_test_file), exist_ok=True)
                with open(logitsgap_test_file, "w") as f:
                    json.dump(logitsgap_dict, f, indent=4)
                print(f"Saved test logitsgap values to {logitsgap_test_file}")

        # Normalize logitsgap (if enabled)
        logitsgap_tensor = torch.tensor(logitsgap_values, dtype=torch.float32)
        if self.normalize_logitsgap:
            logitsgap_tensor = (logitsgap_tensor - self.logitsgap_mean) / (self.logitsgap_std + 1e-8)

        # Get temperatures
        self.temp_model.eval()
        with torch.no_grad():
            temps = self.temp_model(logitsgap_tensor).detach().numpy().flatten()

        # Apply temperature scaling
        # Handle both numpy arrays and torch tensors
        original_is_tensor = isinstance(logits, torch.Tensor)
        original_device = logits.device if original_is_tensor else torch.device('cpu')

        if original_is_tensor:
            logits_tensor = logits.detach().cpu()
        else:
            logits_tensor = torch.tensor(logits, dtype=torch.float32)

        batch_size = logits_tensor.size(0)
        scaled_logits = logits_tensor / torch.tensor(temps, dtype=torch.float32).view(batch_size, 1)

        # Return logits or probabilities based on return_logits flag
        if return_logits:
            # Return as torch tensor on original device
            if original_is_tensor:
                return scaled_logits.to(original_device)
            else:
                return scaled_logits.detach().numpy()
        else:
            # Return probabilities as torch tensor on original device
            probs = F.softmax(scaled_logits, dim=1)
            if original_is_tensor:
                return probs.to(original_device)
            else:
                return probs.detach().numpy()