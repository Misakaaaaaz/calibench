from Component.model.feature_clipping import FeatureClippingCalibrator
from types import SimpleNamespace
import torch
import os
from bench.utils.model_utils import *
from bench.utils.util import get_all_metrics, store_method_results


def FC(args,model_name,dataset_name,device,val_features_tensor, val_logits_tensor, val_labels_tensor,test_features_tensor, test_labels_tensor, overall_results):
    print("\nTraining Feature Clipping Calibration...")

    # Create the same model that was used to extract features
    model = create_model(args,model_name,dataset_name,device)
    model.eval()

    # Get the classifier function - all our models have a classifier method
    classifier_fn = model.classifier

    # Initialize Feature Clipping calibrator
    fc_calibrator = FeatureClippingCalibrator(cross_validate='ece')

    # Set optimal clipping parameter using validation data
    optimal_clip = fc_calibrator.set_feature_clip(
        val_features_tensor, val_logits_tensor, val_labels_tensor, classifier_fn
    )

    print(f"Optimal clipping parameter: {optimal_clip:.4f}")

    # Apply feature clipping to test features and get calibrated logits
    clipped_test_features = fc_calibrator.feature_clipping(test_features_tensor, optimal_clip)
    fc_logits = classifier_fn(clipped_test_features)

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
        logits=fc_logits,
        labels=test_labels_tensor,
    )

    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key='FC',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn='feature_clipping'
    )

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