# Network Calibration by Class-based Temperature Scaling (GPU Batch-Optimized Version)
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict
from tqdm import tqdm

from .calibrator import Calibrator
from ..metrics import (
    ECE, AdaptiveECE, ClasswiseECE, NLL, Accuracy,
    BrierLoss, FocalLoss, LabelSmoothingLoss,
    CrossEntropyLoss, MSELoss, SoftECE
)
from .temperature_scaling import TemperatureScalingCalibrator

# ------------------------------------------------------------------ #
# logging setup
# ------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)           # DEBUG for full details
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


class CTSCalibrator(Calibrator):
    """
    Class-based Temperature Scaling (CTS) with accuracy constraint and GPU batch optimization.
    
    When updating the temperature for a class, the update is accepted only if:
    (1) ECE improves and (2) accuracy does not decrease (ΔAcc ≥ -ε).
    
    This implementation uses GPU batch processing to evaluate all candidate temperatures
    simultaneously, significantly reducing computation time compared to the sequential version.
    """

    def __init__(self,
                 n_class: int,
                 n_iter: int = 5,
                 n_bins: int = 15,
                 acc_epsilon: float = 0.0,
                 grid=None):
        """
        Initialize the CTS calibrator.
        
        Args:
            n_class (int): Number of classes
            n_iter (int): Number of optimization iterations, default 5
            n_bins (int): Number of bins for ECE calculation, default 15
            acc_epsilon (float): Maximum allowed accuracy decrease, default 0.0
            grid (array-like, optional): Temperature grid for search. 
                If None, uses default grid from 0.5 to 5.0 with step 0.1
        """
        super().__init__()
        self.n_class = n_class
        self.n_iter = n_iter
        self.n_bins = n_bins
        self.acc_epsilon = acc_epsilon
        # per-class temperature (buffer → no grad)
        self.register_buffer("T", torch.ones(n_class))
        
        # Metrics
        self.metrics = {
            'ece': ECE(n_bins=n_bins),
            'adaptive_ece': AdaptiveECE(n_bins=n_bins),
            'classwise_ece': ClasswiseECE(n_bins=n_bins),
            'nll': NLL(),
            'accuracy': Accuracy(),
            'brier': BrierLoss(),
            'focal': FocalLoss(),
            'label_smoothing': LabelSmoothingLoss(),
            'cross_entropy': CrossEntropyLoss(),
            'mse': MSELoss(),
            'soft_ece': SoftECE()
        }

        self.ece_fn = ECE(n_bins=n_bins)
        self.acc_fn = Accuracy()
        # Temperature grid for search
        self.default_grid = grid if grid is not None else np.arange(0.5, 5.1, 0.1)

        logger.info("CTSCalibrator (batch-optimized) initialised: classes=%d iters=%d bins=%d ε=%.3f",
                    n_class, n_iter, n_bins, acc_epsilon)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor):
        """
        Forward pass: scale logits by per-class temperatures.
        
        Args:
            x (torch.Tensor): Input logits of shape (N, C)
            
        Returns:
            torch.Tensor: Scaled logits of shape (N, C)
        """
        return x / self.T

    def get_class_temperatures(self):
        """
        Get the current per-class temperature values.
        
        Returns:
            list: List of temperature values for each class
        """
        return self.T.detach().cpu().tolist()

    # ------------------------------------------------------------------ #
    def _ece_and_acc(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Compute ECE and Accuracy for given logits and labels.
        
        Args:
            logits (torch.Tensor): Input logits of shape (N, C)
            labels (torch.Tensor): Target labels of shape (N,)
            
        Returns:
            tuple: (ece, acc) both as float values
        """
        probs = F.softmax(logits, dim=1)
        ece = self.ece_fn(softmaxes=probs, labels=labels).item()
        acc = (probs.argmax(1) == labels).float().mean().item()
        return ece, acc

    # ------------------------------------------------------------------ #
    def _compute_ece_gpu_batch(self, probs: torch.Tensor, labels: torch.Tensor, n_bins: int):
        """
        GPU batch-optimized ECE computation (fully on GPU, avoids CPU-GPU transfers).
        Uses vectorized operations to compute ECE for multiple temperature candidates simultaneously.
        
        Args:
            probs (torch.Tensor): Probability tensor of shape (grid_size, N, C)
            labels (torch.Tensor): Target labels of shape (N,)
            n_bins (int): Number of bins for ECE calculation
        
        Returns:
            numpy.ndarray: Array of ECE values of shape (grid_size,)
        """
        grid_size, n_samples, n_classes = probs.shape
        labels_expanded = labels.unsqueeze(0).expand(grid_size, -1)  # (grid_size, N)
        
        # Compute confidences and predictions
        confidences = probs.max(dim=2)[0]  # (grid_size, N)
        predictions = probs.argmax(dim=2)  # (grid_size, N)
        accuracies = (predictions == labels_expanded).float()  # (grid_size, N)
        
        # Create bin boundaries
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)  # (n_bins+1,)
        bin_lowers = bin_boundaries[:-1].unsqueeze(0).unsqueeze(2)  # (1, n_bins, 1)
        bin_uppers = bin_boundaries[1:].unsqueeze(0).unsqueeze(2)   # (1, n_bins, 1)
        
        # Expand dimensions for batch processing: (grid_size, N) -> (grid_size, 1, N)
        confidences_exp = confidences.unsqueeze(1)  # (grid_size, 1, N)
        
        # Batch compute bin assignments: (grid_size, n_bins, N)
        in_bin = (confidences_exp > bin_lowers) & (confidences_exp <= bin_uppers)  # (grid_size, n_bins, N)
        prop_in_bin = in_bin.float().mean(dim=2)  # (grid_size, n_bins)
        
        # Vectorized computation of bin-wise accuracy and confidence
        # Expand dimensions: (grid_size, N) -> (grid_size, n_bins, N)
        confidences_3d = confidences.unsqueeze(1).expand(-1, n_bins, -1)  # (grid_size, n_bins, N)
        accuracies_3d = accuracies.unsqueeze(1).expand(-1, n_bins, -1)   # (grid_size, n_bins, N)
        
        # Compute weighted accuracy and confidence per bin using masks
        bin_mask_float = in_bin.float()  # (grid_size, n_bins, N)
        bin_counts = bin_mask_float.sum(dim=2)  # (grid_size, n_bins) - number of samples per bin
        
        # Avoid division by zero: only compute for bins with samples
        valid_bins = bin_counts > 0  # (grid_size, n_bins)
        
        # Compute bin-wise accuracy: sum(accuracy * mask) / count
        accuracy_in_bins = (accuracies_3d * bin_mask_float).sum(dim=2)  # (grid_size, n_bins)
        accuracy_in_bins = torch.where(valid_bins, accuracy_in_bins / bin_counts, torch.zeros_like(accuracy_in_bins))
        
        # Compute bin-wise average confidence: sum(confidence * mask) / count
        confidence_in_bins = (confidences_3d * bin_mask_float).sum(dim=2)  # (grid_size, n_bins)
        confidence_in_bins = torch.where(valid_bins, confidence_in_bins / bin_counts, torch.zeros_like(confidence_in_bins))
        
        # Compute contribution of each bin: |confidence - accuracy| * prop
        bin_contributions = torch.abs(confidence_in_bins - accuracy_in_bins) * prop_in_bin  # (grid_size, n_bins)
        
        # Sum over all bins to get ECE
        eces = bin_contributions.sum(dim=1)  # (grid_size,)
        
        return eces.cpu().numpy()

    # ------------------------------------------------------------------ #
    def _batch_evaluate_temperatures(self, 
                                     val_logits: torch.Tensor, 
                                     val_labels: torch.Tensor, 
                                     cls: int, 
                                     grid: np.ndarray, 
                                     device: torch.device):
        """
        Batch evaluate all candidate temperature values for a given class (GPU-optimized version).
        
        This method evaluates all temperature candidates in a single batch operation on GPU,
        significantly reducing computation time compared to sequential evaluation.
        
        Args:
            val_logits (torch.Tensor): Validation logits of shape (N, C)
            val_labels (torch.Tensor): Validation labels of shape (N,)
            cls (int): Current class index to optimize
            grid (np.ndarray): Array of candidate temperature values
            device (torch.device): Device to perform computation on
        
        Returns:
            tuple: (cand_eces, cand_accs) where
                - cand_eces: Tensor of ECE values for each candidate, shape (grid_size,)
                - cand_accs: Tensor of accuracy values for each candidate, shape (grid_size,)
        """
        grid_size = len(grid)
        n_samples, n_classes = val_logits.shape
        
        # Create all candidate temperature vectors (grid_size, n_classes)
        temp_base = self.T.clone()  # (n_classes,)
        temp_candidates = temp_base.unsqueeze(0).repeat(grid_size, 1)  # (grid_size, n_classes)
        temp_candidates[:, cls] = torch.tensor(grid, device=device, dtype=torch.float32)
        
        # Batch compute scaled logits: (grid_size, n_samples, n_classes)
        val_logits_expanded = val_logits.unsqueeze(0)  # (1, n_samples, n_classes)
        temp_expanded = temp_candidates.unsqueeze(1)    # (grid_size, 1, n_classes)
        cand_logits_all = val_logits_expanded / temp_expanded  # (grid_size, n_samples, n_classes)
        
        # Batch compute softmax probabilities
        cand_probs_all = F.softmax(cand_logits_all, dim=2)  # (grid_size, n_samples, n_classes)
        
        # Batch compute accuracy
        preds_all = cand_probs_all.argmax(dim=2)  # (grid_size, n_samples)
        labels_expanded = val_labels.unsqueeze(0).expand(grid_size, -1)  # (grid_size, n_samples)
        cand_accs = (preds_all == labels_expanded).float().mean(dim=1)  # (grid_size,)
        
        # GPU batch compute ECE (fully on GPU, avoids CPU-GPU transfers)
        cand_eces = self._compute_ece_gpu_batch(cand_probs_all, val_labels, self.n_bins)
        
        return torch.tensor(cand_eces, device=device, dtype=torch.float32), cand_accs

    # ------------------------------------------------------------------ #
    def fit(self,
            val_logits: torch.Tensor,
            val_labels: torch.Tensor,
            ts_loss: str = "nll",
            **kwargs) -> Dict[str, float]:
        """
        Fit the CTS calibrator on validation data.
        
        Args:
            val_logits (torch.Tensor): Validation logits of shape (N, C)
            val_labels (torch.Tensor): Validation labels of shape (N,)
            ts_loss (str): Loss function for initial TS, default "nll"
            **kwargs: Additional arguments
                - grid: Custom temperature grid for search
                - loss_fn: Custom loss function for TS
        
        Returns:
            Dict[str, float]: Dictionary containing final ECE and accuracy
        """
        # Select device, prefer GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Ensure input tensors are on the correct device
        val_logits = val_logits.to(device)
        val_labels = val_labels.to(device)
        
        verbose = kwargs.get("verbose", True)
        
        if verbose:
            logger.info("Using device: %s", device)
            logger.debug("Logit range: %.4f to %.4f", val_logits.min().item(), val_logits.max().item())

        # Evaluate raw model performance
        raw_probs = F.softmax(val_logits, dim=1)
        raw_acc = (raw_probs.argmax(1) == val_labels).float().mean().item()
        if verbose:
            logger.info("Raw model accuracy: %.4f", raw_acc)
            logger.info("CTS fit started")

        # ---------- 1-D TS initialization ----------
        logger.debug("TS loss type: %s", ts_loss)
        ts = TemperatureScalingCalibrator(loss_type=ts_loss)
        ts.to(device)  # Ensure TS model is on GPU
        if "loss_fn" in kwargs:
            ts.loss_fn = kwargs["loss_fn"]
        ts.fit(val_logits, val_labels)

        with torch.no_grad():
            init_T = ts.temperature.clamp(0.5, 5.0)  # Clamp to prevent extreme underconfidence
            self.T.fill_(init_T.item())
        
        if verbose:
            logger.info("Initialised all class temps with TS value %.4f", init_T.item())

        # ---------- Compute initial metrics ----------
        cur_logits = val_logits / self.T  # (N, C)
        best_ece, best_acc = self._ece_and_acc(cur_logits, val_labels)
        if verbose:
            logger.info("Initial ECE: %.6f | Acc: %.4f", best_ece, best_acc)

        # ---------- Greedy optimization ----------
        grid = kwargs.get("grid", self.default_grid)
        
        # Count classes with samples (do this once before the loop)
        valid_classes = []
        for cls in range(self.n_class):
            mask = (val_labels == cls)
            if mask.sum().item() > 0:
                valid_classes.append(cls)
        
        total_updates = 0  # Track number of temperature updates
        
        # Outer loop: iterations with progress bar
        iter_pbar = tqdm(range(self.n_iter), desc="CTS", disable=not verbose, 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
        
        for it in iter_pbar:
            # Inner loop: classes with progress bar
            class_pbar = tqdm(valid_classes, desc=f"Iter {it+1}/{self.n_iter}", 
                            leave=False, disable=not verbose,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for cls in class_pbar:
                cls_best_temp = self.T[cls].item()   # Current temperature
                cls_best_ece = best_ece
                cls_best_acc = best_acc

                # GPU batch optimization: evaluate all candidate temperatures at once
                cand_eces, cand_accs = self._batch_evaluate_temperatures(
                    val_logits, val_labels, cls, grid, device
                )
                cand_eces_np = cand_eces.cpu().numpy()
                cand_accs_np = cand_accs.cpu().numpy()
                
                # Find optimal value
                for i, (cand_ece, cand_acc) in enumerate(zip(cand_eces_np, cand_accs_np)):
                    # Condition: ECE improves & Acc does not decrease beyond epsilon
                    if (cand_ece < cls_best_ece) and (cand_acc >= best_acc - self.acc_epsilon):
                        cls_best_ece, cls_best_acc, cls_best_temp = cand_ece, cand_acc, grid[i]
                
                # Update if better solution found
                if cls_best_temp != self.T[cls].item():
                    with torch.no_grad():
                        self.T[cls] = cls_best_temp
                    best_ece, best_acc = cls_best_ece, cls_best_acc
                    total_updates += 1
                    # Update inner progress bar
                    if verbose:
                        class_pbar.set_postfix({
                            'ECE': f'{best_ece:.6f}',
                            'Acc': f'{best_acc:.4f}'
                        })
            
            # Update outer progress bar with summary
            if verbose:
                iter_pbar.set_postfix({
                    'ECE': f'{best_ece:.6f}',
                    'Acc': f'{best_acc:.4f}',
                    'Updates': total_updates
                })

        # ---------- Final report ----------
        if verbose:
            logger.info("CTS fit complete | Final ECE=%.6f Acc=%.4f", best_ece, best_acc)
            temps = self.T.cpu().numpy()
            logger.info("Temperature stats -> min:%.3f max:%.3f mean:%.3f std:%.3f",
                        temps.min(), temps.max(), temps.mean(), temps.std())

        return {"final_ece": best_ece, "final_accuracy": best_acc}

    # ------------------------------------------------------------------ #
    def calibrate(self, test_logits, return_logits=False, **_):
        """
        Calibrate test logits using learned per-class temperatures.
        
        Args:
            test_logits (torch.Tensor): Test logits of shape (N, C)
            return_logits (bool): If True, return scaled logits; if False, return probabilities
            **_: Additional keyword arguments (ignored)
        
        Returns:
            torch.Tensor: Calibrated probabilities or logits
        """
        # Select device, prefer GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        if not torch.is_tensor(test_logits):
            test_logits = torch.tensor(test_logits, dtype=torch.float32, device=device)
        else:
            test_logits = test_logits.to(device)
            
        logits = self.forward(test_logits)
        return logits if return_logits else F.softmax(logits, dim=1)

    # ------------------------------------------------------------------ #
    def save(self, path="./"):
        """
        Save the CTS model to disk.
        
        Args:
            path (str): Directory path to save the model
        """
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/cts_model.pth")
        logger.info("CTS model saved to %s/cts_model.pth", path)

    def load(self, path="./"):
        """
        Load the CTS model from disk.
        
        Args:
            path (str): Directory path to load the model from
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(f"{path}/cts_model.pth", map_location=device))
        self.to(device)
        logger.info("CTS model loaded from %s/cts_model.pth and moved to %s", path, device)

    def compute_all_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute all available metrics for the given logits and labels.
        
        Args:
            logits (torch.Tensor): Input logits of shape (N, C)
            labels (torch.Tensor): Target labels of shape (N,)
            
        Returns:
            Dict[str, float]: Dictionary containing all metric values
        """
        # Select device, prefer GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Ensure input tensors are on the correct device
        logits = logits.to(device)
        labels = labels.to(device)
        
        probs = F.softmax(logits / self.T, dim=1)
        
        results = {}
        for name, metric in self.metrics.items():
            metric = metric.to(device)
            try:
                if name in ['nll', 'cross_entropy']:
                    value = metric(logits=logits / self.T, labels=labels)
                elif name in ['brier', 'focal', 'label_smoothing', 'mse']:
                    value = metric(softmaxes=probs, labels=labels)
                elif name in ['ece', 'adaptive_ece', 'classwise_ece', 'soft_ece']:
                    value = metric(softmaxes=probs, labels=labels)
                elif name == 'accuracy':
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

    def get_all_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Get all metrics in a dictionary format compatible with the results structure.
        
        Args:
            logits (torch.Tensor): Input logits of shape (N, C)
            labels (torch.Tensor): Target labels of shape (N,)
            
        Returns:
            Dict[str, float]: Dictionary containing all metric values in the format:
            {
                'ece': float,
                'accuracy': float,
                'adaece': float,
                'cece': float,
                'nll': float
            }
        """
        metrics = self.compute_all_metrics(logits, labels)
        return {
            'ece': metrics.get('ece', None),
            'accuracy': metrics.get('accuracy', None),
            'adaece': metrics.get('adaptive_ece', None),
            'cece': metrics.get('classwise_ece', None),
            'nll': metrics.get('nll', None)
        }
