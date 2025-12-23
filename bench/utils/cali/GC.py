
import torch
import os

from Component.model.group_calibration import GroupCalibrationCalibrator
from bench.utils.util import get_all_metrics, store_method_results


def GC(val_logits_tensor, val_labels_tensor, test_logits_tensor, test_labels_tensor, device,overall_results):
    print("\nTraining Group Calibration...")

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

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
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

if __name__ == '__main__':
    pass

# ECE:  0.021001603752241048
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.022305700927972794
# ClasswiseECE:  0.0006801192066632211
# NLL:  4.3824286460876465
# ECEDebiased:  0.03526881666805167
# ECESweep:  0.022846114471273653
# BrierLoss:  0.42928507924079895
# RBS:  0.926590621471405