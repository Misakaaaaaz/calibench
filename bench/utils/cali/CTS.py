
import torch
import os

from Component.model.cts import CTSCalibrator
from bench.utils.util import get_all_metrics, store_method_results


def CTS(val_labels_tensor, val_logits_tensor,test_logits_tensor,test_labels_tensor,overall_results,device):
    print("\nTraining Class-based Temperature Scaling...")

    # Update args for CTS calibration TODO: Change
    cts_loss = "CE"

    # Ensure labels are the correct data type for the loss function
    if cts_loss == 'soft_ece':
        # Convert labels to int64 for soft_ece loss
        val_labels_for_ts = val_labels_tensor.long()
    else:
        val_labels_for_ts = val_labels_tensor

    cts_calibrator = CTSCalibrator(
        n_class=1000,  # Number of classes
        n_bins=15,
        n_iter=5,  # Number of bins for ECE computation
    ).to(device)

    val_logits_device = val_logits_tensor.to(device)
    val_labels_device = val_labels_for_ts.to(device)

    # Fit the calibrator on validation data
    cts_calibrator.fit(val_logits_device, val_labels_device, ts_loss=cts_loss)

    # Calibrate test logits
    test_logits_device = test_logits_tensor.to(device)

    cts_probs = cts_calibrator.calibrate(test_logits_device).cpu().detach().numpy()

    # Get calibrated logits for metric calculation
    calibrated_logits = cts_calibrator.calibrate(test_logits_device, return_logits=True)

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
        logits=calibrated_logits,
        labels=test_labels_tensor,
    )

    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key=f'CTS_{cts_loss}',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn=cts_loss
    )

if __name__ == "__main__":
    pass

# ECE:  0.011711168481012498
# Accuracy:  0.23862741887569427
# AdaptiveECE:  0.012131850235164165
# ClasswiseECE:  0.0007698740810155869
# NLL:  4.552792072296143
# ECEDebiased:  0.022326507743344805
# ECESweep:  0.01208578674300208
# BrierLoss:  0.43529176712036133
# RBS:  0.9330506324768066