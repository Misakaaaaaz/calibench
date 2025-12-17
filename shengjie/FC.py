from calibrator.Component.model.feature_clipping import FeatureClippingCalibrator
from utils.model_utils import create_model
from metric import metric
from types import SimpleNamespace
import torch
import os

def FC(pt, model, dataset):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    args = SimpleNamespace(dataset_root='data')
    model_name = model
    dataset_name = dataset
    device = 'cuda'
    val_logits = val_logits.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    test_logits = test_logits.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    model = create_model(args, model_name, dataset_name, device)
    model.eval()
    classifier_fn = model.classifier
    calibrator = FeatureClippingCalibrator(cross_validate='ece')
    optimal_clip = calibrator.set_feature_clip(val_features, val_logits, val_labels, classifier_fn)
    clipped_test_features = calibrator.feature_clipping(test_features, optimal_clip)
    fc_logits = classifier_fn(clipped_test_features)
    logit = fc_logits
    label = test_labels
    print(pt)
    print('FC')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_FC.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == '__main__':
    pt = "imagenet_sketch_resnet50_seed.pt"
    model = "resnet50"
    dataset = "imagenet_sketch"
    FC(pt, model, dataset)

# ECE:  0.11224935805773859
# Accuracy:  0.2379642277956009
# AdaptiveECE:  0.11224912106990814
# ClasswiseECE:  0.0008903742418624461
# NLL:  4.662837982177734
# ECEDebiased:  0.12248622580062288
# ECESweep:  0.112249364587245
# BrierLoss:  0.4431566298007965
# RBS:  0.9414421319961548