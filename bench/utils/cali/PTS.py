

import torch
import os

from Component.model.pts import PTSCalibrator
from bench.utils.util import get_all_metrics, store_method_results


def PTS(val_logits_tensor,val_labels_tensor,test_logits_tensor,test_labels_tensor,seed,overall_results,device):
    print("\nTraining Parametric Temperature Scaling...")

    # Initialize PTSCalibrator with overwrite flag and fixed seed
    pts_calibrator = PTSCalibrator(
        steps=10000,
        lr=0.00005,
        nlayers=2,
        n_nodes=5,
        loss_fn="MSE",
        top_k_logits=10,
        seed=seed  # Use the same seed for consistency
    ).to(device)

    val_logits_device = val_logits_tensor.to(device)
    val_labels_device = val_labels_tensor.to(device)
    pts_calibrator.fit(val_logits_device, val_labels_device)
    test_logits_device = test_logits_tensor.to(device)

    # Get calibrated logits for metric calculation
    calibrated_logits = pts_calibrator.calibrate(test_logits_device, return_logits=True)

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
        logits=calibrated_logits,
        labels=test_labels_tensor,
    )

    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key=f'PTS_MSE',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn='MSE'
    )

if __name__ == "__main__":
    pass

# ECE:  0.020306435398926556
# Accuracy:  0.23904499411582947
# AdaptiveECE:  0.01906575821340084
# ClasswiseECE:  0.0008157623233273625
# NLL:  4.548316478729248
# ECEDebiased:  0.021089597205441082
# ECESweep:  0.019557584937984707
# BrierLoss:  0.43728160858154297
# RBS:  0.9351808428764343




# from calibrator.Component import PTSCalibrator
# from metric import metric
# import torch
# import os
#
# def PTS(pt, seed):
#     val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
#     num_classes = val_logits.shape[1]
#     calibrator = PTSCalibrator(
#         length_logits=num_classes,
#         steps=10000,
#         lr=0.00005,
#         nlayers=2,
#         n_nodes=5,
#         loss_fn="ce",
#         top_k_logits=10,
#         seed=seed
#     )
#     calibrator.fit(val_logits, val_labels)
#     calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
#     logit = calibrated_logits
#     label = test_labels
#     print(pt)
#     print('PTS')
#     metric(logit, label)
#     # PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
#     # OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
#     # save_path = os.path.join(
#     #     OUTPUT_DIR, f"{pt}_PTS.json"
#     # )
#     # save_dir = os.path.dirname(save_path)
#     # if save_dir:
#     #     os.makedirs(save_dir, exist_ok=True)
#     # metric(logit, label, save=save_path)
#
# if __name__ == "__main__":
#     pt = "output/imagenet_sketch_swin_b_seed5_vs0.2.pt"
#     seed = 5
#     PTS(pt, seed)
#
# # ECE:  0.020306435398926556
# # Accuracy:  0.23904499411582947
# # AdaptiveECE:  0.01906575821340084
# # ClasswiseECE:  0.0008157623233273625
# # NLL:  4.548316478729248
# # ECEDebiased:  0.021089597205441082
# # ECESweep:  0.019557584937984707
# # BrierLoss:  0.43728160858154297
# # RBS:  0.9351808428764343
#
# # output/imagenet_sketch_swin_b_seed5_vs0.2.pt
# # PTS
# # ECE:  0.05849864142159054
# # Accuracy:  0.31457555294036865
# # AdaptiveECE:  0.05849793925881386
# # ClasswiseECE:  0.0006838978733867407
# # NLL:  4.057199001312256
# # ECEDebiased:  0.06299777036238521
# # ECESweep:  0.058498651647380634
# # BrierLoss:  0.40758126974105835
# # RBS:  0.9028635025024414