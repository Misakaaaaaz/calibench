

import torch
import os

from Component.model.ets import ETSCalibrator
from bench.utils.util import get_all_metrics, store_method_results


def ETS(val_logits,val_labels,test_logits,test_labels_tensor,overall_results,device):
    print("\nTraining Ensemble Temperature Scaling...")

    # Get number of classes from dataset
    n_classes = 1000
    ets_loss = 'mse'
    # Initialize ETS calibrator
    ets_calibrator = ETSCalibrator(loss_type=ets_loss, n_classes=n_classes)

    # Fit the calibrator on validation data
    ets_calibrator.fit(val_logits, val_labels)

    # Calibrate test logits
    ets_probs = ets_calibrator.calibrate(test_logits)
    ets_probs_tensor = torch.tensor(ets_probs, dtype=torch.float32)

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
        logits=ets_probs_tensor,  # ETS returns probabilities
        labels=test_labels_tensor,
    )
    print(f"Optimal temperature: {ets_calibrator.get_temperature():.4f}")
    print(f"Optimal weights: {ets_calibrator.get_weights()}")

    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key=f'ETS_{ets_loss}',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn=ets_loss,
        additional_params={
            'temp': float(ets_calibrator.get_temperature()),
            'weights': ets_calibrator.get_weights()
        }
    )

if __name__ == '__main__':
    pass

# ECE:  0.010077651298029712
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.009272750467061996
# ClasswiseECE:  0.000737900089006871
# NLL:  4.495574951171875
# ECEDebiased:  0.01141324241108635
# ECESweep:  0.009272574557494358
# BrierLoss:  0.43492305278778076
# RBS:  0.932655394077301