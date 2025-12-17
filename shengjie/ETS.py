from calibrator.Component import ETSCalibrator
from metric import metric
import torch
import os

def ETS(pt):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    num_classes = val_logits.shape[1]
    calibrator = ETSCalibrator(n_classes=num_classes)
    calibrator.fit(val_logits, val_labels)
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
    logit = calibrated_logits
    label = test_labels
    print(pt)
    print('ETS')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_ETS.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == '__main__':
    pt = "imagenet_sketch_resnet50_seed.pt"
    ETS(pt)

# ECE:  0.010077651298029712
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.009272750467061996
# ClasswiseECE:  0.000737900089006871
# NLL:  4.495574951171875
# ECEDebiased:  0.01141324241108635
# ECESweep:  0.009272574557494358
# BrierLoss:  0.43492305278778076
# RBS:  0.932655394077301