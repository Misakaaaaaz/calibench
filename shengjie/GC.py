from calibrator.Component import GroupCalibrationCalibrator
from metric import metric
import torch
import os

def GC(pt):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    calibrator = GroupCalibrationCalibrator(
        num_groups=2,
        num_partitions=20,
        weight_decay=0.1
    )
    calibrator.fit(val_logits, val_labels)
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
    logit = calibrated_logits
    label = test_labels
    print(pt)
    print('GC')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_GC.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == '__main__':
    pt = "imagenet_sketch_resnet50_seed.pt"
    GC(pt)

# ECE:  0.021001603752241048
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.022305700927972794
# ClasswiseECE:  0.0006801192066632211
# NLL:  4.3824286460876465
# ECEDebiased:  0.03526881666805167
# ECESweep:  0.022846114471273653
# BrierLoss:  0.42928507924079895
# RBS:  0.926590621471405