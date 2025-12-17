from calibrator.Component import CTSCalibrator
from metric import metric
import torch
import os

def CTS(pt):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    num_classes = val_logits.shape[1]
    calibrator = CTSCalibrator(
        n_class=num_classes,
        n_bins=15,
        n_iter=5
    )
    calibrator.fit(val_logits, val_labels)
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
    logit = calibrated_logits
    label = test_labels
    print(pt)
    print('CTS')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_CTS.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == "__main__":
    pt = "imagenet_sketch_resnet50_seed.pt"
    CTS(pt)

# ECE:  0.011711168481012498
# Accuracy:  0.23862741887569427
# AdaptiveECE:  0.012131850235164165
# ClasswiseECE:  0.0007698740810155869
# NLL:  4.552792072296143
# ECEDebiased:  0.022326507743344805
# ECESweep:  0.01208578674300208
# BrierLoss:  0.43529176712036133
# RBS:  0.9330506324768066