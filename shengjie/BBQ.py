from calibrator.Component import BBQCalibrator
from metric import metric
import torch
import os

def BBQ(pt):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    calibrator = BBQCalibrator(score_type='max_prob', n_bins_max=20)
    calibrator.fit(val_logits, val_labels)
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
    logit = calibrated_logits
    label = test_labels
    print(pt)
    print('BBQ')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_BBQ.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == '__main__':
    pt = "imagenet_sketch_resnet50_seed.pt"
    BBQ(pt)

# ECE:  0.2242920956674851
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.22429172694683075
# ClasswiseECE:  0.0009912484092637897
# NLL:  5.073429584503174
# ECEDebiased:  0.2461973593196547
# ECESweep:  0.22429208917458013
# BrierLoss:  0.4708074927330017
# RBS:  0.970368504524231