import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def compute_ece(probs, labels, n_bins=15):
    """
    Compute ECE (Expected Calibration Error)
    
    Args:
        probs: numpy array of shape [n_samples, n_classes] with probabilities
        labels: numpy array of shape [n_samples] with ground truth labels
        n_bins: number of bins for confidence histogram
        
    Returns:
        Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def plot_enhanced_calibration_curve(probs, labels, method_name, plot_dir=None):
    """
    Plot enhanced reliability diagram with confidence histograms
    
    Args:
        probs: numpy array of shape [n_samples, n_classes] with probabilities
        labels: numpy array of shape [n_samples] with ground truth labels
        method_name: string name of the calibration method for the title
        plot_dir: directory to save the plot, if None, saves to "plots/"
    """
    if plot_dir is None:
        plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    n_bins = 15
    
    # Get predicted confidence and check if prediction is correct
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    # Create figure with confidence histogram
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Plot calibration curve
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    true_probs = []
    mean_confidences = []
    sample_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_count = np.sum(in_bin)
        sample_counts.append(bin_count)
        
        if bin_count > 0:
            true_prob = np.mean(accuracies[in_bin])
            mean_conf = np.mean(confidences[in_bin])
            true_probs.append(true_prob)
            mean_confidences.append(mean_conf)
        else:
            true_probs.append(0)
            mean_confidences.append(bin_centers[len(mean_confidences)])
    
    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    # Plot calibration curve
    ax1.plot(mean_confidences, true_probs, 's-', label='Calibration curve')
    
    # Add calibration metrics
    ece = compute_ece(probs, labels)
    ax1.set_xlabel('Mean predicted probability', fontsize=24)
    ax1.set_ylabel('Fraction of positives', fontsize=24)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(loc='lower right', fontsize=24)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=24)
    
    # Create confidence histogram on the same plot
    ax2 = ax1.twinx()
    
    # Plot separate histograms for correct and incorrect predictions
    correct_confidences = confidences[accuracies]
    incorrect_confidences = confidences[~accuracies]
    
    ax2.hist([correct_confidences, incorrect_confidences], bins=20,
             color=['green', 'red'], alpha=0.3, label=['Correct', 'Incorrect'],
             stacked=True)

    ax2.set_ylabel('Count', fontsize=24)
    ax2.legend(loc='upper right', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=24)
    
    # Save the plot
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'calibration_{method_name.replace(" ", "_")}.pdf')
    plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()
    
    return ece

def plot_logitsgap_analysis(test_logitsgap, model_name, plot_dir=None):
    """
    Plot logitsgap distribution with CDF
    
    Args:
        test_logitsgap: array of logitsgap values
        model_name: name of the model
        plot_dir: directory to save the plot
    """
    if plot_dir is None:
        plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot detailed logitsgap distribution as a bar chart
    plt.figure(figsize=(14, 8))
    
    # Create bins
    num_bins = 20
    logitsgap_hist, logitsgap_bin_edges = np.histogram(test_logitsgap, bins=num_bins, range=(0, 10))
    logitsgap_bin_centers = 0.5 * (logitsgap_bin_edges[1:] + logitsgap_bin_edges[:-1])
    
    # Create bin labels
    logitsgap_bin_labels = [f"{logitsgap_bin_edges[i]:.1f}-{logitsgap_bin_edges[i+1]:.1f}" 
                          for i in range(len(logitsgap_bin_edges)-1)]
    
    plt.bar(logitsgap_bin_centers, logitsgap_hist,
            width=(logitsgap_bin_edges[1] - logitsgap_bin_edges[0]) * 0.8,
            alpha=0.7, label=f'{model_name}')
    plt.xlabel('Logit Margin', fontsize=24)
    plt.ylabel('Count', fontsize=24)
    plt.grid(alpha=0.3)
    plt.xticks(logitsgap_bin_centers, logitsgap_bin_labels, rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'detailed_logitsgap_distribution.pdf')
    plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()
    
    return {
        'bin_labels': logitsgap_bin_labels,
        'bin_centers': logitsgap_bin_centers.tolist(),
        'bin_edges': logitsgap_bin_edges.tolist(),
        'counts': logitsgap_hist.tolist(),
        'percentage': (logitsgap_hist / len(test_logitsgap) * 100).tolist()
    }

def plot_temperature_distribution(temps, model_name, optimal_temp=None, plot_dir=None):
    """
    Plot temperature distribution
    
    Args:
        temps: array of temperature values
        model_name: name of the model
        optimal_temp: optimal temperature from TS (optional)
        plot_dir: directory to save the plot
    """
    if plot_dir is None:
        plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calculate range
    temp_min = float(np.min(temps))
    temp_max = float(np.max(temps))
    temp_range = (max(0.1, temp_min - 0.1), temp_max + 0.1)
    
    # Create histogram
    temp_hist, temp_bin_edges = np.histogram(temps, bins=20, range=temp_range)
    temp_bin_centers = 0.5 * (temp_bin_edges[1:] + temp_bin_edges[:-1])
    
    # Create bin labels
    temp_bin_labels = [f"{temp_bin_edges[i]:.2f}-{temp_bin_edges[i+1]:.2f}" 
                      for i in range(len(temp_bin_edges)-1)]
    
    # Plot temperature distribution
    plt.figure(figsize=(14, 8))
    plt.bar(temp_bin_centers, temp_hist,
            width=(temp_bin_edges[1] - temp_bin_edges[0]) * 0.8,
            alpha=0.7, label=f'{model_name}')
    plt.xlabel('Temperature', fontsize=24)
    plt.ylabel('Count', fontsize=24)
    plt.grid(alpha=0.3)
    plt.xticks(temp_bin_centers, temp_bin_labels, rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    
    if optimal_temp is not None:
        plt.axvline(x=optimal_temp, color='r', linestyle='--',
                   label=f'TS temp: {optimal_temp:.4f}')

    plt.legend(fontsize=24)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'detailed_temperature_distribution.pdf')
    plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()
    
    return {
        'bin_labels': temp_bin_labels,
        'bin_centers': temp_bin_centers.tolist(),
        'bin_edges': temp_bin_edges.tolist(),
        'counts': temp_hist.tolist(),
        'percentage': (temp_hist / len(temps) * 100).tolist()
    }

def plot_logitsgap_temperature_relationship(test_logitsgap, temps, model_name, optimal_temp=None, plot_dir=None):
    """
    Plot relationship between logitsgap and temperature
    
    Args:
        test_logitsgap: array of logitsgap values
        temps: array of temperature values
        model_name: name of the model
        optimal_temp: optimal temperature from TS (optional)
        plot_dir: directory to save the plot
    """
    if plot_dir is None:
        plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calculate correlation and regression
    logitsgap_temp_corr, logitsgap_temp_pvalue = stats.pearsonr(test_logitsgap, temps)
    slope, intercept, r_value, p_value, std_err = stats.linregress(test_logitsgap, temps)
    
    # Plot density heatmap
    plt.figure(figsize=(14, 10))
    plt.hexbin(test_logitsgap, temps, gridsize=30, cmap='viridis', mincnt=1)
    cbar = plt.colorbar(label='Count')
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Count', fontsize=24)
    
    # Add regression line
    plt.plot(np.array([min(test_logitsgap), max(test_logitsgap)]),
            intercept + slope * np.array([min(test_logitsgap), max(test_logitsgap)]),
            'r-', linewidth=3, label=f'Regression line (r={r_value:.2f})')

    plt.xlabel('Logit Margin', fontsize=24)
    plt.ylabel('Temperature', fontsize=24)

    if optimal_temp is not None:
        plt.axhline(y=optimal_temp, color='r', linestyle='--', linewidth=2,
                   label=f'TS temp={optimal_temp:.2f}')

    plt.legend(fontsize=24)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'logitsgap_temperature_density_map.pdf')
    plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()
    
    # Plot regular scatter plot with CDF
    plt.figure(figsize=(12, 10))
    
    # logitsgap distribution
    plt.subplot(2, 1, 1)
    counts, bins, _ = plt.hist(test_logitsgap, bins=30, alpha=0.7, label=f'{model_name}')
    plt.xlabel('Logit Margin', fontsize=24)
    plt.ylabel('Count', fontsize=24)
    plt.legend(fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    # Add Quantiles
    quantiles = [0.25, 0.5, 0.75]
    quantile_values = np.quantile(test_logitsgap, quantiles)
    for i, q in enumerate(quantiles):
        plt.axvline(x=quantile_values[i], color='r', linestyle='--',
                  alpha=0.5, label=f'{int(q*100)}th percentile' if i == 0 else "")

    # Plot CDF
    ax2 = plt.gca().twinx()
    cdf = np.cumsum(counts) / np.sum(counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax2.plot(bin_centers, cdf, 'g-', label='CDF')
    ax2.set_ylabel('Cumulative Probability', fontsize=24)
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(loc='upper left', fontsize=24)
    
    # logitsgap vs temperature
    plt.subplot(2, 1, 2)
    plt.scatter(test_logitsgap, temps, alpha=0.3, label=f'{model_name} samples')
    
    # Add regression line
    plt.plot(np.array([min(test_logitsgap), max(test_logitsgap)]),
            intercept + slope * np.array([min(test_logitsgap), max(test_logitsgap)]),
            'r', label=f'Regression line (r={r_value:.2f})')

    plt.xlabel('Logit Margin', fontsize=24)
    plt.ylabel('Temperature', fontsize=24)

    if optimal_temp is not None:
        plt.axhline(y=optimal_temp, color='r', linestyle='-', label=f'TS temp={optimal_temp:.2f}')

    plt.legend(fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'logitsgap_temperature_mapping.pdf')
    plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()
    
    return {
        'logitsgap_temperature_correlation': float(logitsgap_temp_corr),
        'logitsgap_temperature_pvalue': float(logitsgap_temp_pvalue),
        'regression': {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'std_err': float(std_err)
        }
    }

def plot_confidence_distribution(probs_dict, model_name, plot_dir=None):
    """
    Plot confidence distribution for different methods
    
    Args:
        probs_dict: dictionary mapping method names to probability arrays
        model_name: name of the model
        plot_dir: directory to save the plot
    """
    if plot_dir is None:
        plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    hist_data = []
    labels = []
    
    for method, probs in probs_dict.items():
        hist_data.append(np.max(probs, axis=1))
        labels.append(method)

    plt.hist(hist_data, bins=20, alpha=0.7, label=labels)
    plt.xlabel('Confidence', fontsize=24)
    plt.ylabel('Count', fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    save_path = os.path.join(plot_dir, 'confidence_distribution.pdf')
    plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()

def plot_confidence_change(uncal_probs, calibrated_probs_dict, test_logitsgap, model_name, plot_dir=None, 
                          color_by_prediction_groups=False, labels=None, top_k=2):
    """
    Plot confidence change vs logitsgap for different methods
    
    Args:
        uncal_probs: uncalibrated probabilities
        calibrated_probs_dict: dictionary mapping method names to calibrated probability arrays
        test_logitsgap: array of logitsgap values
        model_name: name of the model
        plot_dir: directory to save the plot
        color_by_prediction_groups: whether to color points by prediction correctness and logits gap groups
        labels: true labels (required if color_by_prediction_groups=True)
        top_k: number of top predictions to consider (required if color_by_prediction_groups=True)
    """
    if plot_dir is None:
        plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Validation for color grouping
    if color_by_prediction_groups and (labels is None):
        raise ValueError("labels must be provided when color_by_prediction_groups=True")
    
    plt.figure(figsize=(12, 10))
    
    num_methods = len(calibrated_probs_dict)
    method_idx = 0
    
    for method, probs in calibrated_probs_dict.items():
        conf_change = np.max(probs, axis=1) - np.max(uncal_probs, axis=1)
        
        plt.subplot(num_methods, 1, method_idx + 1)
        
        if not color_by_prediction_groups:
            # Original coloring - single color for all points
            plt.scatter(test_logitsgap, conf_change, alpha=0.3, label=method)
        else:
            # New coloring scheme based on prediction correctness and logits gap
            # Convert logitsgap to array and handle different formats
            test_logitsgap_array = np.array(test_logitsgap)
            
            # If logitsgap is a list of lists (gap vectors), take the first element of each
            if test_logitsgap_array.ndim > 1:
                # For gap vectors, take the first gap (largest gap)
                test_logitsgap_array = test_logitsgap_array[:, 0]
            
            print(f"Processing method: {method}")
            print(f"Logitsgap array shape: {test_logitsgap_array.shape}")
            print(f"Logitsgap sample values: {test_logitsgap_array[:5]}")
            print(f"Confidence change shape: {conf_change.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Top_k: {top_k}")
            
            # Get predictions
            predictions = np.argmax(uncal_probs, axis=1)
            
            # Sort predictions by confidence to get top-k
            sorted_indices = np.argsort(uncal_probs, axis=1)[:, ::-1]  # Sort in descending order
            
            # Determine prediction correctness
            correct_predictions = (predictions == labels)
            
            # Group by logits gap size (3 groups)
            n_samples = len(test_logitsgap_array)
            n_groups = 3
            group_size = n_samples // n_groups
            
            # Sort indices by logits gap for grouping
            gap_sorted_indices = np.argsort(test_logitsgap_array)
            
            # Define color schemes
            # For correct predictions: use blue shades (light blue for small gap, dark blue for large gap)
            correct_small_gap_color = '#ADD8E6'  # Light blue
            correct_large_gap_color = '#000080'  # Navy blue
            
            # For incorrect predictions in top k: use green shades
            incorrect_in_topk_small_gap_color = '#90EE90'  # Light green
            incorrect_in_topk_large_gap_color = '#006400'  # Dark green
            
            # For incorrect predictions not in top k: use red shades
            incorrect_not_in_topk_small_gap_color = '#FFB6C1'  # Light pink
            incorrect_not_in_topk_large_gap_color = '#8B0000'  # Dark red
            
            # Create mapping from sample index to gap group
            gap_groups = {}
            for rank_idx in range(n_samples):
                sample_idx = int(gap_sorted_indices[rank_idx])  # Convert to int to ensure hashable
                if rank_idx < group_size:
                    gap_groups[sample_idx] = 1  # Small gap
                elif rank_idx >= n_samples - group_size:
                    gap_groups[sample_idx] = 3  # Large gap
                else:
                    gap_groups[sample_idx] = 2  # Middle group (will be skipped)
            
            # Collect points by category for batch plotting
            categories = {
                'Correct (Small Gap)': {'x': [], 'y': [], 'color': correct_small_gap_color},
                'Correct (Large Gap)': {'x': [], 'y': [], 'color': correct_large_gap_color},
                'Incorrect (In Top-k, Small Gap)': {'x': [], 'y': [], 'color': incorrect_in_topk_small_gap_color},
                'Incorrect (In Top-k, Large Gap)': {'x': [], 'y': [], 'color': incorrect_in_topk_large_gap_color},
                'Incorrect (Not in Top-k, Small Gap)': {'x': [], 'y': [], 'color': incorrect_not_in_topk_small_gap_color},
                'Incorrect (Not in Top-k, Large Gap)': {'x': [], 'y': [], 'color': incorrect_not_in_topk_large_gap_color}
            }
            
            # Categorize points
            for sample_idx in range(n_samples):
                gap_value = test_logitsgap_array[sample_idx]
                conf_change_value = conf_change[sample_idx]
                
                # Get gap group for this sample
                gap_group = gap_groups.get(sample_idx, 2)
                if gap_group == 2:
                    continue  # Skip middle group as requested
                
                if correct_predictions[sample_idx]:
                    # Correct prediction - use blue shades
                    if gap_group == 1:  # Small gap
                        category = 'Correct (Small Gap)'
                    else:  # Large gap (group 3)
                        category = 'Correct (Large Gap)'
                else:
                    # Incorrect prediction - check if true label is in top k
                    top_k_predictions = sorted_indices[sample_idx, :top_k]
                    true_label_in_topk = labels[sample_idx] in top_k_predictions
                    
                    if true_label_in_topk:
                        # True label in top k - use green shades
                        if gap_group == 1:  # Small gap
                            category = 'Incorrect (In Top-k, Small Gap)'
                        else:  # Large gap (group 3)
                            category = 'Incorrect (In Top-k, Large Gap)'
                    else:
                        # True label not in top k - use red shades
                        if gap_group == 1:  # Small gap
                            category = 'Incorrect (Not in Top-k, Small Gap)'
                        else:  # Large gap (group 3)
                            category = 'Incorrect (Not in Top-k, Large Gap)'
                
                categories[category]['x'].append(gap_value)
                categories[category]['y'].append(conf_change_value)
            
            # Plot each category
            for category, data in categories.items():
                if len(data['x']) > 0:  # Only plot if there are points in this category
                    print(f"Plotting {len(data['x'])} points for category: {category}")
                    plt.scatter(data['x'], data['y'], alpha=0.6, color=data['color'], s=10, label=category)
                else:
                    print(f"No points found for category: {category}")
            
            print(f"Total samples: {n_samples}, Group size: {group_size}")
            print(f"Gap groups distribution: {list(gap_groups.values()).count(1)} small, {list(gap_groups.values()).count(2)} middle, {list(gap_groups.values()).count(3)} large")


        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel('Logit Margin', fontsize=24)
        plt.ylabel('Conf Change', fontsize=24)
        if color_by_prediction_groups:
            plt.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend(fontsize=24)
        plt.grid(alpha=0.3)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.title(f'{method}', fontsize=26)
        
        method_idx += 1
    
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'logitsgap_vs_confidence_change.pdf')
    plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()

def plot_logitsgap_by_correctness(test_logitsgap, probs, labels, model_name, plot_dir=None):
    """
    Plot logitsgap distribution separated by prediction correctness
    
    Args:
        test_logitsgap: array of logitsgap values
        probs: predicted probabilities
        labels: true labels
        model_name: name of the model
        plot_dir: directory to save the plot
    """
    if plot_dir is None:
        plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    correct = (np.argmax(probs, axis=1) == labels)
    
    plt.figure(figsize=(10, 8))
    plt.hist([np.array(test_logitsgap)[correct], np.array(test_logitsgap)[~correct]],
            bins=20, alpha=0.7, label=['Correct', 'Incorrect'])
    plt.xlabel('Logit Margin', fontsize=24)
    plt.ylabel('Count', fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    save_path = os.path.join(plot_dir, 'logitsgap_by_correctness.pdf')
    plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()

def plot_performance_by_logitsgap(ece_by_group, acc_by_group, group_names, plot_dir=None):
    """
    Plot calibration and accuracy performance by logitsgap level
    
    Args:
        ece_by_group: dictionary mapping method names to ECE values by group
        acc_by_group: dictionary mapping method names to accuracy values by group
        group_names: names of logitsgap groups
        plot_dir: directory to save the plot
    """
    if plot_dir is None:
        plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(group_names))
    width = 0.25
    
    # ECE subplot
    plt.subplot(2, 1, 1)
    
    offset = -width
    for method, ece_values in ece_by_group.items():
        plt.bar(x + offset, ece_values, width, label=method)
        offset += width

    plt.xlabel('Logit Margin Group', fontsize=24)
    plt.ylabel('ECE (lower is better)', fontsize=24)
    plt.xticks(x, group_names, fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Accuracy subplot
    plt.subplot(2, 1, 2)
    
    offset = -width
    for method, acc_values in acc_by_group.items():
        plt.bar(x + offset, acc_values, width, label=method)
        offset += width

    plt.xlabel('Logit Margin Group', fontsize=24)
    plt.ylabel('Accuracy (higher is better)', fontsize=24)
    plt.xticks(x, group_names, fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'performance_by_logitsgap.pdf')
    plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close() 