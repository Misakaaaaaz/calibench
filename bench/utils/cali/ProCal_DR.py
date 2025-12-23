
import torch
import os

from Component.model.procal import ProCalDensityRatioCalibrator
from bench.utils.util import get_all_metrics, store_method_results


def ProCal_DR(val_features_tensor, test_features_tensor, val_logits_tensor, test_logits_tensor, val_labels_tensor, test_labels_tensor, device,overall_results):
    print("\nTraining ProCal Density-Ratio Calibration...")

    # Initialize ProCal Density-Ratio calibrator
    procal_dr_calibrator = ProCalDensityRatioCalibrator(
        k_neighbors=10,
        bandwidth='normal_reference',
        kernel='KDEMultivariate',
        distance_measure='L2',
        normalize_features=True
    )

    # Fit the calibrator on validation data with features
    procal_dr_calibrator.fit(val_logits_tensor, val_labels_tensor, val_features_tensor)

    # Apply calibration to test set with features
    procal_dr_probs = procal_dr_calibrator.calibrate(test_logits_tensor, test_features_tensor)

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
        logits=procal_dr_probs,
        labels=test_labels_tensor,
    )

    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key='ProCal_DR',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn='density_ratio'
    )

if __name__ == '__main__':
    pass

# ECE:  0.10080302436492926
# Accuracy:  0.2182648777961731
# AdaptiveECE:  0.09839121997356415
# ClasswiseECE:  0.0009383425931446254
# NLL:  4.873789310455322
# ECEDebiased:  0.10665284853739596
# ECESweep:  0.09882844894180852
# BrierLoss:  0.44716885685920715
# RBS:  0.9456943273544312