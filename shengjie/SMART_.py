from utils.smart_calibrator import SMART
from metric import metric
from types import SimpleNamespace
import torch
import os

def SMART_(pt, dataset, model, seed, vs, loss):
    val_logits, val_features, val_labels, test_logits, test_features, test_labels = torch.load(pt, weights_only=False)
    args = SimpleNamespace(dataset_root='data')
    smart_epochs = 200
    dataset_name = dataset
    model_name = model
    seed_value = seed
    valid_size = vs
    smart_loss = getattr(args, 'smart_loss', 'soft_ece')
    patience = 20
    min_delta = 0.0001
    corruption_type = getattr(args, 'corruption_type', None) if dataset_name == 'imagenet_c' else None
    severity = getattr(args, 'severity', None) if dataset_name == 'imagenet_c' else None
    train_loss = loss
    smart = SMART(epochs=smart_epochs, dataset_name=dataset_name,
                  model_name=model_name, seed_value=seed_value,
                  valid_size=valid_size, loss_fn=smart_loss,
                  patience=patience, min_delta=min_delta,
                  corruption_type=corruption_type, severity=severity,
                  train_loss=train_loss)
    smart.fit(val_logits, val_labels)
    smart_logits = smart.calibrate(test_logits, return_logits=True)
    logit = smart_logits
    label = test_labels
    print(pt)
    print('SMART')
    # metric(logit, label)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")
    save_path = os.path.join(
        OUTPUT_DIR, f"{pt}_SMART.json"
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metric(logit, label, save=save_path)

if __name__ == '__main__':
    pt = "imagenet_sketch_resnet50_seed.pt"
    dataset = "imagenet_sketch"
    model = "resnet50"
    seed = 1
    vs = 0.2
    loss = "ce"
    SMART_(pt, dataset, model, seed, vs, loss)

# ECE:  0.0097737745467905
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.009374484419822693
# ClasswiseECE:  0.0007760548032820225
# NLL:  4.506536483764648
# ECEDebiased:  0.015255164200498116
# ECESweep:  0.009325494212476327
# BrierLoss:  0.43617013096809387
# RBS:  0.933991551399231