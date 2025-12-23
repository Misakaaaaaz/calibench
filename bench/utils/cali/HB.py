
import torch
import os

from Component.model.histogram_binning import HistogramBinningCalibrator
from bench.utils.util import get_all_metrics, store_method_results


def HB(val_logits_tensor, val_labels_tensor,test_logits_tensor,test_labels_tensor,device,overall_results):
    print("\nTraining Histogram Binning...")

    # Initialize Histogram Binning calibrator
    hb_calibrator = HistogramBinningCalibrator(n_bins=15, strategy='uniform')

    # Fit the calibrator on validation data
    hb_calibrator.fit(val_logits_tensor, val_labels_tensor)

    # Apply calibration to test set
    hb_logits = hb_calibrator.calibrate(test_logits_tensor, return_logits=True)

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
        logits=hb_logits,
        labels=test_labels_tensor,
    )

    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key='HB',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn='uniform_binning'
    )

if __name__ == '__main__':
    pass

# ECE:  0.2242849710501553
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.2242845743894577
# ClasswiseECE:  0.0009912238456308842
# NLL:  5.070596218109131
# ECEDebiased:  0.24621253639173618
# ECESweep:  0.22428497318785995
# BrierLoss:  0.47080621123313904
# RBS:  0.9703671336174011