from calibrator.Component.metrics import ECE, Accuracy, AdaptiveECE, ClasswiseECE, NLL, ECEDebiased, ECESweep, BrierLoss, RBS
import torch
import numpy as np
import json

def metric(logit, label, save=None):
    if isinstance(logit, np.ndarray):
        logit = torch.from_numpy(logit)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    device = logit.device
    if device.type == "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    logit = logit.to(device)
    label = label.to(device)
    print("ECE: ", ECE().to(device)(logits=logit, labels=label))
    print("Accuracy: ", Accuracy().to(device)(logits=logit, labels=label))
    print("AdaptiveECE: ", AdaptiveECE().to(device)(logits=logit, labels=label))
    print("ClasswiseECE: ", ClasswiseECE().to(device)(logits=logit, labels=label))
    print("NLL: ", NLL().to(device)(logits=logit, labels=label))
    print("ECEDebiased: ", ECEDebiased().to(device)(logits=logit, labels=label))
    print("ECESweep: ", ECESweep().to(device)(logits=logit, labels=label))
    print("BrierLoss: ", float(BrierLoss().to(device)(logits=logit, labels=label).detach().cpu()))
    print("RBS: ", float(RBS().to(device)(logits=logit, labels=label).detach().cpu()))
    result = {
        "ECE": ECE().to(device)(logits=logit, labels=label),
        "Accuracy": Accuracy().to(device)(logits=logit, labels=label),
        "AdaptiveECE": AdaptiveECE().to(device)(logits=logit, labels=label),
        "ClasswiseECE": ClasswiseECE().to(device)(logits=logit, labels=label),
        "NLL": NLL().to(device)(logits=logit, labels=label),
        "ECEDebiased": ECEDebiased().to(device)(logits=logit, labels=label),
        "ECESweep": ECESweep().to(device)(logits=logit, labels=label),
        "BrierLoss": float(BrierLoss().to(device)(logits=logit, labels=label).detach().cpu()),
        "RBS": float(RBS().to(device)(logits=logit, labels=label).detach().cpu())
    }
    if save is not None:
        with open(save, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)