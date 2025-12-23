

import torch
import os

from Component.model.temperature_scaling import TemperatureScalingCalibrator
from bench.utils.util import get_all_metrics, store_method_results


def TS(val_logits_tensor, val_labels_tensor, test_logits_tensor, test_labels_tensor, overall_results,device):
    print("\nTraining Temperature Scaling...")

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

    # Get optimal temperature parameter
    optimal_temp = ts_calibrator.temperature.item()

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
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

if __name__ == "__main__":
    pass

# ECE:  0.016303749985613722
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.016350537538528442
# ClasswiseECE:  0.0007517749909311533
# NLL:  4.495458126068115
# ECEDebiased:  0.021933010883898648
# ECESweep:  0.016406973648035277
# BrierLoss:  0.43513572216033936
# RBS:  0.9328833818435669