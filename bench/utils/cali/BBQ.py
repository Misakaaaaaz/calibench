

import torch
import os

from Component.model.bbq import BBQCalibrator
from bench.utils.util import get_all_metrics, store_method_results


def BBQ(val_logits_tensor, val_labels_tensor,test_logits_tensor, test_labels_tensor, overall_results,device):
    print("\nTraining BBQ...")

    # Initialize BBQ calibrator
    bbq_calibrator = BBQCalibrator(score_type='max_prob', n_bins_max=20)

    # Fit the calibrator on validation data
    bbq_calibrator.fit(val_logits_tensor, val_labels_tensor)

    # Apply calibration to test set
    bbq_logits = bbq_calibrator.calibrate(test_logits_tensor, return_logits=True)

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
        logits=bbq_logits,
        labels=test_labels_tensor,
    )

    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key='BBQ',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn='bayesian_binning'
    )

if __name__ == '__main__':
    pass

# ECE:  0.2242920956674851
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.22429172694683075
# ClasswiseECE:  0.0009912484092637897
# NLL:  5.073429584503174
# ECEDebiased:  0.2461973593196547
# ECESweep:  0.22429208917458013
# BrierLoss:  0.4708074927330017
# RBS:  0.970368504524231