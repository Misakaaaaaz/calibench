from calibrator.Component import VectorScalingCalibrator
from metric import metric
import torch
import os

def VS(pt):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    calibrator = VectorScalingCalibrator(loss_type='nll', bias=True)
    calibrator.fit(val_logits, val_labels)
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
    logit = calibrated_logits
    label = test_labels
    print(pt)
    print('VS')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_VS.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

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