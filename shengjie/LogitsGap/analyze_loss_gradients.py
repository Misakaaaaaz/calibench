"""
Gradient Analysis for Different Loss Functions in SMART Calibration

This script analyzes and visualizes how different loss functions (SmoothSoftECE, SoftECE, NLL/CE)
produce different gradient behaviors during SMART calibration training.

Analyses include:
1. Gradient field visualization (how gradients change with predictions)
2. Gradient magnitude comparison across confidence levels
3. Convergence behavior during SMART training
4. Sample-wise gradient distribution
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import json

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

# Import loss functions
from calibrator.Component.metrics import (
    CrossEntropyLoss, SoftECE, NLL
)
from calibrator.Component.metrics.SmoothSoftECE import SmoothSoftECE

# Import SMART utilities
from utils.smart_calibrator import compute_logitsgap


def compute_gradients_for_loss(loss_fn, logits, labels, temperature):
    """
    Compute gradients of loss with respect to temperature parameter.
    
    Args:
        loss_fn: Loss function object
        logits: Input logits (N, C)
        labels: True labels (N,)
        temperature: Temperature value (scalar)
    
    Returns:
        gradients: Gradient tensor with respect to temperature
        loss_value: Scalar loss value
    """
    # Ensure temperature requires grad
    temp = torch.tensor([temperature], dtype=torch.float32, requires_grad=True)
    
    # Scale logits by temperature
    scaled_logits = logits / temp
    
    # Compute loss
    loss = loss_fn(logits=scaled_logits, labels=labels)
    
    # Compute gradient
    loss.backward()
    
    return temp.grad.item(), loss.item()


def analyze_gradient_field(logits, labels, loss_functions, temp_range=(0.5, 3.0), num_points=50):
    """
    Analyze how gradients change across different temperature values.
    
    Args:
        logits: Sample logits (N, C)
        labels: Sample labels (N,)
        loss_functions: Dict of {name: loss_fn}
        temp_range: Range of temperatures to analyze
        num_points: Number of temperature points to sample
    
    Returns:
        results: Dict containing temperature values, gradients, and losses for each loss function
    """
    temps = np.linspace(temp_range[0], temp_range[1], num_points)
    
    results = {}
    for loss_name, loss_fn in loss_functions.items():
        gradients = []
        losses = []
        
        for temp in tqdm(temps, desc=f"Computing gradients for {loss_name}"):
            grad, loss_val = compute_gradients_for_loss(
                loss_fn, logits.clone(), labels.clone(), temp
            )
            gradients.append(grad)
            losses.append(loss_val)
        
        results[loss_name] = {
            'temperatures': temps,
            'gradients': np.array(gradients),
            'losses': np.array(losses)
        }
    
    return results


def plot_gradient_fields(results, save_path):
    """
    Plot gradient field analysis comparing different loss functions.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    loss_names = list(results.keys())
    colors = {'SmoothSoftECE': '#2E86AB', 'SoftECE': '#A23B72', 'NLL': '#F18F01'}
    
    # 1. Gradient magnitude vs Temperature
    ax1 = fig.add_subplot(gs[0, :])
    for loss_name in loss_names:
        temps = results[loss_name]['temperatures']
        grads = results[loss_name]['gradients']
        ax1.plot(temps, grads, label=loss_name, linewidth=2.5, 
                color=colors.get(loss_name, 'gray'), alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Temperature', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Gradient (∂L/∂T)', fontsize=14, fontweight='bold')
    ax1.set_title('Gradient Magnitude vs Temperature', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # 2. Loss landscape
    ax2 = fig.add_subplot(gs[1, :])
    for loss_name in loss_names:
        temps = results[loss_name]['temperatures']
        losses = results[loss_name]['losses']
        ax2.plot(temps, losses, label=loss_name, linewidth=2.5,
                color=colors.get(loss_name, 'gray'), alpha=0.8)
    
    ax2.set_xlabel('Temperature', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss Value', fontsize=14, fontweight='bold')
    ax2.set_title('Loss Landscape', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # 3-5. Individual gradient magnitude plots for each loss
    for idx, loss_name in enumerate(loss_names):
        ax = fig.add_subplot(gs[2, idx])
        temps = results[loss_name]['temperatures']
        grads = np.abs(results[loss_name]['gradients'])
        
        ax.fill_between(temps, 0, grads, alpha=0.4, color=colors.get(loss_name, 'gray'))
        ax.plot(temps, grads, linewidth=2, color=colors.get(loss_name, 'gray'))
        
        ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
        ax.set_ylabel('|∂L/∂T|', fontsize=12, fontweight='bold')
        ax.set_title(f'{loss_name}\nGradient Magnitude', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved gradient field plot to {save_path}")


def analyze_sample_wise_gradients(logits, labels, loss_functions, temperature=1.5):
    """
    Analyze gradients for individual samples to understand sensitivity.
    
    Args:
        logits: Sample logits (N, C)
        labels: Sample labels (N,)
        loss_functions: Dict of {name: loss_fn}
        temperature: Temperature value for analysis
    
    Returns:
        sample_gradients: Dict containing per-sample gradient information
    """
    n_samples = logits.shape[0]
    
    results = {}
    for loss_name, loss_fn in loss_functions.items():
        sample_grads = []
        sample_losses = []
        
        # Analyze each sample individually
        for i in tqdm(range(n_samples), desc=f"Analyzing samples for {loss_name}"):
            sample_logit = logits[i:i+1]
            sample_label = labels[i:i+1]
            
            grad, loss_val = compute_gradients_for_loss(
                loss_fn, sample_logit, sample_label, temperature
            )
            sample_grads.append(grad)
            sample_losses.append(loss_val)
        
        # Compute confidence and correctness
        probs = F.softmax(logits / temperature, dim=1)
        confidences = probs.max(dim=1)[0].numpy()
        predictions = probs.argmax(dim=1).numpy()
        correct = (predictions == labels.numpy()).astype(int)
        
        results[loss_name] = {
            'gradients': np.array(sample_grads),
            'losses': np.array(sample_losses),
            'confidences': confidences,
            'correct': correct
        }
    
    return results


def plot_sample_wise_analysis(sample_results, save_path):
    """
    Plot sample-wise gradient analysis.
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    loss_names = list(sample_results.keys())
    colors = {'SmoothSoftECE': '#2E86AB', 'SoftECE': '#A23B72', 'NLL': '#F18F01'}
    
    # 1. Gradient distribution comparison
    ax1 = fig.add_subplot(gs[0, :])
    for loss_name in loss_names:
        grads = sample_results[loss_name]['gradients']
        ax1.hist(grads, bins=50, alpha=0.5, label=loss_name, 
                color=colors.get(loss_name, 'gray'), edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Gradient Value (∂L/∂T)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax1.set_title('Sample-wise Gradient Distribution', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(labelsize=11)
    
    # 2-4. Gradients vs Confidence for each loss
    for idx, loss_name in enumerate(loss_names):
        ax = fig.add_subplot(gs[1, idx])
        
        data = sample_results[loss_name]
        confidences = data['confidences']
        gradients = data['gradients']
        correct = data['correct']
        
        # Separate correct and incorrect predictions
        correct_mask = correct == 1
        incorrect_mask = correct == 0
        
        ax.scatter(confidences[correct_mask], gradients[correct_mask], 
                  alpha=0.4, s=20, c='green', label='Correct', edgecolors='none')
        ax.scatter(confidences[incorrect_mask], gradients[incorrect_mask],
                  alpha=0.4, s=20, c='red', label='Incorrect', edgecolors='none')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gradient (∂L/∂T)', fontsize=12, fontweight='bold')
        ax.set_title(f'{loss_name}\nGradients vs Confidence', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    # 5-7. Box plots: Gradient magnitude by correctness
    for idx, loss_name in enumerate(loss_names):
        ax = fig.add_subplot(gs[2, idx])
        
        data = sample_results[loss_name]
        gradients = np.abs(data['gradients'])
        correct = data['correct']
        
        box_data = [gradients[correct == 1], gradients[correct == 0]]
        bp = ax.boxplot(box_data, labels=['Correct', 'Incorrect'],
                       patch_artist=True, widths=0.6)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor(colors.get(loss_name, 'gray'))
            patch.set_alpha(0.6)
        
        ax.set_ylabel('|∂L/∂T|', fontsize=12, fontweight='bold')
        ax.set_title(f'{loss_name}\nGradient by Correctness', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sample-wise analysis plot to {save_path}")


def train_smart_with_different_losses(val_logits, val_labels, test_logits, test_labels, 
                                      loss_functions, epochs=500, lr=0.001):
    """
    Train SMART with different loss functions and track convergence.
    
    Args:
        val_logits: Validation logits for computing logitsgap
        val_labels: Validation labels
        test_logits: Test logits
        test_labels: Test labels
        loss_functions: Dict of {name: loss_fn}
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        training_history: Dict containing training curves for each loss function
    """
    # Compute logitsgap values
    print("Computing logitsgap values...")
    val_logitsgap = compute_logitsgap(val_logits)
    test_logitsgap = compute_logitsgap(test_logits)
    
    # Normalize logitsgap
    logitsgap_mean = np.mean(val_logitsgap)
    logitsgap_std = np.std(val_logitsgap)
    
    val_logitsgap_norm = (val_logitsgap - logitsgap_mean) / (logitsgap_std + 1e-8)
    test_logitsgap_norm = (test_logitsgap - logitsgap_mean) / (logitsgap_std + 1e-8)
    
    # Convert to tensors
    val_logitsgap_tensor = torch.tensor(val_logitsgap_norm, dtype=torch.float32)
    test_logitsgap_tensor = torch.tensor(test_logitsgap_norm, dtype=torch.float32)
    val_logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    test_logits_tensor = torch.tensor(test_logits, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    
    training_history = {}
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining SMART with {loss_name}...")
        
        # Simple MLP for temperature prediction
        class TempModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(1, 32)
                self.fc2 = torch.nn.Linear(32, 32)
                self.fc3 = torch.nn.Linear(32, 1)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                # Temperature should be positive
                return F.softplus(x) + 0.1
        
        model = TempModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        test_losses = []
        grad_norms = []
        temperature_values = []
        
        for epoch in tqdm(range(epochs), desc=f"Training {loss_name}"):
            model.train()
            optimizer.zero_grad()
            
            # Predict temperatures
            temps = model(val_logitsgap_tensor).squeeze()
            
            # Scale logits by predicted temperatures
            scaled_logits = val_logits_tensor / temps.unsqueeze(1)
            
            # Compute loss
            loss = loss_fn(logits=scaled_logits, labels=val_labels_tensor)
            
            # Backward pass
            loss.backward()
            
            # Track gradient norm
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_grad_norm ** 0.5
            
            optimizer.step()
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_temps = model(test_logitsgap_tensor).squeeze()
                test_scaled_logits = test_logits_tensor / test_temps.unsqueeze(1)
                test_loss = loss_fn(logits=test_scaled_logits, labels=test_labels_tensor)
                
                train_losses.append(loss.item())
                test_losses.append(test_loss.item())
                grad_norms.append(grad_norm)
                temperature_values.append(temps.mean().item())
        
        training_history[loss_name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'grad_norms': grad_norms,
            'temperature_values': temperature_values,
            'final_model': model
        }
    
    return training_history


def plot_convergence_analysis(training_history, save_path):
    """
    Plot convergence behavior for different loss functions.
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    loss_names = list(training_history.keys())
    colors = {'SmoothSoftECE': '#2E86AB', 'SoftECE': '#A23B72', 'NLL': '#F18F01'}
    
    # 1. Training loss convergence
    ax1 = fig.add_subplot(gs[0, 0])
    for loss_name in loss_names:
        losses = training_history[loss_name]['train_losses']
        ax1.plot(losses, label=loss_name, linewidth=2, 
                color=colors.get(loss_name, 'gray'), alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training Loss Convergence', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.tick_params(labelsize=11)
    
    # 2. Test loss convergence
    ax2 = fig.add_subplot(gs[0, 1])
    for loss_name in loss_names:
        losses = training_history[loss_name]['test_losses']
        ax2.plot(losses, label=loss_name, linewidth=2,
                color=colors.get(loss_name, 'gray'), alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=14, fontweight='bold')
    ax2.set_title('Test Loss Convergence', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.tick_params(labelsize=11)
    
    # 3. Gradient norm evolution
    ax3 = fig.add_subplot(gs[1, 0])
    for loss_name in loss_names:
        grad_norms = training_history[loss_name]['grad_norms']
        ax3.plot(grad_norms, label=loss_name, linewidth=2,
                color=colors.get(loss_name, 'gray'), alpha=0.8)
    
    ax3.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Gradient Norm', fontsize=14, fontweight='bold')
    ax3.set_title('Gradient Norm Evolution', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.tick_params(labelsize=11)
    
    # 4. Mean temperature evolution
    ax4 = fig.add_subplot(gs[1, 1])
    for loss_name in loss_names:
        temps = training_history[loss_name]['temperature_values']
        ax4.plot(temps, label=loss_name, linewidth=2,
                color=colors.get(loss_name, 'gray'), alpha=0.8)
    
    ax4.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Mean Temperature', fontsize=14, fontweight='bold')
    ax4.set_title('Mean Temperature Evolution', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=11)
    
    # 5. Training loss - last 100 epochs (zoomed)
    ax5 = fig.add_subplot(gs[2, 0])
    for loss_name in loss_names:
        losses = training_history[loss_name]['train_losses'][-100:]
        ax5.plot(losses, label=loss_name, linewidth=2,
                color=colors.get(loss_name, 'gray'), alpha=0.8)
    
    ax5.set_xlabel('Epoch (last 100)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax5.set_title('Training Loss (Final 100 Epochs)', fontsize=16, fontweight='bold')
    ax5.legend(fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=11)
    
    # 6. Convergence speed comparison
    ax6 = fig.add_subplot(gs[2, 1])
    convergence_data = []
    for loss_name in loss_names:
        train_losses = np.array(training_history[loss_name]['train_losses'])
        # Find epoch where loss drops to 90%, 50%, 10% of initial value
        initial_loss = train_losses[0]
        epochs_to_90 = np.argmax(train_losses < 0.9 * initial_loss) if np.any(train_losses < 0.9 * initial_loss) else len(train_losses)
        epochs_to_50 = np.argmax(train_losses < 0.5 * initial_loss) if np.any(train_losses < 0.5 * initial_loss) else len(train_losses)
        epochs_to_10 = np.argmax(train_losses < 0.1 * initial_loss) if np.any(train_losses < 0.1 * initial_loss) else len(train_losses)
        
        convergence_data.append([epochs_to_90, epochs_to_50, epochs_to_10])
    
    x = np.arange(len(loss_names))
    width = 0.25
    
    ax6.bar(x - width, [d[0] for d in convergence_data], width, 
           label='90% of initial', alpha=0.8, color='#3A86FF')
    ax6.bar(x, [d[1] for d in convergence_data], width,
           label='50% of initial', alpha=0.8, color='#FB5607')
    ax6.bar(x + width, [d[2] for d in convergence_data], width,
           label='10% of initial', alpha=0.8, color='#06FFA5')
    
    ax6.set_xlabel('Loss Function', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Epochs to Convergence', fontsize=14, fontweight='bold')
    ax6.set_title('Convergence Speed Comparison', fontsize=16, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(loss_names, fontsize=11)
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.tick_params(labelsize=11)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence analysis plot to {save_path}")


def create_gradient_summary_table(gradient_results, sample_results, training_history, save_path):
    """
    Create a summary table comparing gradient characteristics.
    """
    loss_names = list(gradient_results.keys())
    
    summary = {}
    for loss_name in loss_names:
        # Gradient field statistics
        grads = gradient_results[loss_name]['gradients']
        grad_mean = np.mean(np.abs(grads))
        grad_std = np.std(grads)
        grad_max = np.max(np.abs(grads))
        
        # Sample-wise statistics
        sample_grads = sample_results[loss_name]['gradients']
        sample_grad_mean = np.mean(np.abs(sample_grads))
        sample_grad_std = np.std(sample_grads)
        
        # Training statistics
        final_loss = training_history[loss_name]['train_losses'][-1]
        final_grad_norm = training_history[loss_name]['grad_norms'][-1]
        
        summary[loss_name] = {
            'Mean |∂L/∂T|': f"{grad_mean:.4f}",
            'Std ∂L/∂T': f"{grad_std:.4f}",
            'Max |∂L/∂T|': f"{grad_max:.4f}",
            'Sample Mean |∂L/∂T|': f"{sample_grad_mean:.4f}",
            'Sample Std ∂L/∂T': f"{sample_grad_std:.4f}",
            'Final Training Loss': f"{final_loss:.6f}",
            'Final Grad Norm': f"{final_grad_norm:.4f}"
        }
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nGradient Summary Table:")
    print("=" * 100)
    for loss_name in loss_names:
        print(f"\n{loss_name}:")
        for key, value in summary[loss_name].items():
            print(f"  {key}: {value}")
    print("=" * 100)
    
    print(f"\nSaved summary table to {save_path}")


def main():
    """
    Main function to run gradient analysis.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Analyze loss function gradients for SMART')
    parser.add_argument('--dataset', type=str, default='imagenet_sketch',
                       help='Dataset to use for analysis')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model to use')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed')
    parser.add_argument('--valid_size', type=float, default=0.2,
                       help='Validation size')
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='Number of samples to use for analysis')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of epochs for training analysis')
    parser.add_argument('--output_dir', type=str, default='gradient_analysis',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading logits for {args.dataset} {args.model}...")
    
    # Load logits from cache
    from utils.utils import load_logits, get_logit_paths
    
    try:
        val_logits, val_labels, test_logits, test_labels, val_features, test_features = load_logits(
            dataset_name=args.dataset,
            model_name=args.model,
            seed_value=args.seed,
            valid_size=args.valid_size
        )
    except FileNotFoundError:
        print("Error: Logits not found. Please run smart_logitgap.py first to generate logits.")
        return
    
    # Subsample for faster analysis
    if args.num_samples < len(test_logits):
        indices = np.random.choice(len(test_logits), args.num_samples, replace=False)
        test_logits = test_logits[indices]
        test_labels = test_labels[indices]
    
    print(f"Using {len(test_logits)} samples for analysis")
    
    # Convert to tensors
    logits_tensor = torch.tensor(test_logits, dtype=torch.float32)
    labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    
    # Initialize loss functions
    loss_functions = {
        'SmoothSoftECE': SmoothSoftECE(n_bins=15, kernel_bandwidth=0.1),
        'SoftECE': SoftECE(n_bins=15),
        'NLL': NLL()
    }
    
    print("\n" + "="*80)
    print("GRADIENT ANALYSIS FOR SMART CALIBRATION")
    print("="*80)
    
    # 1. Gradient field analysis
    print("\n[1/4] Analyzing gradient fields...")
    gradient_results = analyze_gradient_field(
        logits_tensor, labels_tensor, loss_functions,
        temp_range=(0.5, 3.0), num_points=50
    )
    
    plot_gradient_fields(
        gradient_results,
        os.path.join(args.output_dir, 'gradient_field_analysis.png')
    )
    
    # 2. Sample-wise gradient analysis
    print("\n[2/4] Analyzing sample-wise gradients...")
    sample_results = analyze_sample_wise_gradients(
        logits_tensor, labels_tensor, loss_functions,
        temperature=1.5
    )
    
    plot_sample_wise_analysis(
        sample_results,
        os.path.join(args.output_dir, 'sample_wise_gradient_analysis.png')
    )
    
    # 3. Training convergence analysis
    print("\n[3/4] Analyzing training convergence...")
    training_history = train_smart_with_different_losses(
        val_logits, val_labels, test_logits, test_labels,
        loss_functions, epochs=args.epochs, lr=0.001
    )
    
    plot_convergence_analysis(
        training_history,
        os.path.join(args.output_dir, 'convergence_analysis.png')
    )
    
    # 4. Create summary table
    print("\n[4/4] Creating summary table...")
    create_gradient_summary_table(
        gradient_results, sample_results, training_history,
        os.path.join(args.output_dir, 'gradient_summary.json')
    )
    
    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print("="*80)
    
    print("\nKey Insights:")
    print("-" * 80)
    print("1. SmoothSoftECE uses kernel smoothing, producing smoother gradients")
    print("2. SoftECE has discrete bin boundaries, causing gradient discontinuities")
    print("3. NLL focuses on log-likelihood, with different sensitivity to confidence")
    print("4. Check the plots for detailed gradient behavior and convergence patterns")
    print("-" * 80)


if __name__ == "__main__":
    main()

