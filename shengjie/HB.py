from calibrator.Component import HistogramBinningCalibrator
from metric import metric
import torch
import os

def HB(pt):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    calibrator = HistogramBinningCalibrator(n_bins=15, strategy='uniform')
    calibrator.fit(val_logits, val_labels)
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
    logit = calibrated_logits
    label = test_labels
    print(pt)
    print('HB')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_HB.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == '__main__':
    pt = "imagenet_sketch_resnet50_seed.pt"
    HB(pt)

# ECE:  0.2242849710501553
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.2242845743894577
# ClasswiseECE:  0.0009912238456308842
# NLL:  5.070596218109131
# ECEDebiased:  0.24621253639173618
# ECESweep:  0.22428497318785995
# BrierLoss:  0.47080621123313904
# RBS:  0.9703671336174011