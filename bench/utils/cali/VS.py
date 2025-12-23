
import torch
import os

from Component.model.vector_scaling import VectorScalingCalibrator
from bench.utils.util import get_all_metrics, store_method_results


def VS(val_logits_tensor, val_labels_tensor, test_logits_tensor, test_labels_tensor, device,overall_results):
    print("\nTraining Vector Scaling...")

    # Initialize Vector Scaling calibrator
    vs_calibrator = VectorScalingCalibrator(loss_type='nll', bias=True)

    # Fit the calibrator on validation data
    vs_calibrator.fit(val_logits_tensor, val_labels_tensor)

    # Apply calibration to test set
    vs_logits = vs_calibrator.calibrate(test_logits_tensor, return_logits=True)

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
        logits=vs_logits,
        labels=test_labels_tensor,
    )

    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key='VS',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn='vector_scaling'
    )

if __name__ == '__main__':
    pt = "imagenet_sketch_resnet50_seed.pt"
    VS(pt)

# ECE:  0.11493719512394321
# Accuracy:  0.26653075218200684
# AdaptiveECE:  0.11493688076734543
# ClasswiseECE:  0.0007777303690090775
# NLL:  4.455328464508057
# ECEDebiased:  0.12693450881544888
# ECESweep:  0.11493719541840144
# BrierLoss:  0.42887911200523376
# RBS:  0.9261523485183716