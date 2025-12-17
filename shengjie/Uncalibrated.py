from metric import metric
import torch
import os

def Uncalibrated(pt):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    logit = test_logits
    label = test_labels
    print(pt)
    print('Uncalibrated')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_Uncalibrated.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == "__main__":
    pt = "imagenet_sketch_resnet50_seed.pt"
    Uncalibrated(pt)

# ECE:  0.22429245881850476
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.2242920845746994
# ClasswiseECE:  0.0009912487585097551
# NLL:  5.075135707855225
# ECEDebiased:  0.2461977228895761
# ECESweep:  0.22429246433360342
# BrierLoss:  0.47080758213996887
# RBS:  0.9703685641288757