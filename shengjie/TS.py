from calibrator.Component import TemperatureScalingCalibrator
from metric import metric
import torch
import os

def TS(pt):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    calibrator = TemperatureScalingCalibrator()
    calibrator.fit(val_logits, val_labels)
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
    logit = calibrated_logits
    label = test_labels
    print(pt)
    print('TS')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_TS.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == "__main__":
    pt = "imagenet_sketch_resnet50_seed.pt"
    TS(pt)

# ECE:  0.016303749985613722
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.016350537538528442
# ClasswiseECE:  0.0007517749909311533
# NLL:  4.495458126068115
# ECEDebiased:  0.021933010883898648
# ECESweep:  0.016406973648035277
# BrierLoss:  0.43513572216033936
# RBS:  0.9328833818435669