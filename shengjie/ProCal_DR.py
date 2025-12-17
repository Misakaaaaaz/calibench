from calibrator.Component import ProCalDensityRatioCalibrator
from metric import metric
import torch
import os

def ProCal_DR(pt):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    calibrator = ProCalDensityRatioCalibrator(
        k_neighbors=10,
        bandwidth='normal_reference',
        kernel='KDEMultivariate',
        distance_measure='L2',
        normalize_features=True
    )
    calibrator.fit(val_logits, val_labels, val_features)
    calibrated_logits = calibrator.calibrate(test_logits, test_features, return_logits=True)
    logit = calibrated_logits
    label = test_labels
    print(pt)
    print('ProCal_DR')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_ProCal_DR.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == '__main__':
    pt = "imagenet_sketch_resnet50_seed.pt"
    ProCal_DR(pt)

# ECE:  0.10080302436492926
# Accuracy:  0.2182648777961731
# AdaptiveECE:  0.09839121997356415
# ClasswiseECE:  0.0009383425931446254
# NLL:  4.873789310455322
# ECEDebiased:  0.10665284853739596
# ECESweep:  0.09882844894180852
# BrierLoss:  0.44716885685920715
# RBS:  0.9456943273544312