import os
import json
import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 350
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

BASE_DIR = "./cifar10"

SEEDS = [1, 2, 3, 4, 5]

MODELS = {
    "resnet50": "resnet50",
    "resnet152": "resnet152",
    "densenet121": "densenet121",
    "wide_resnet50": "wide-resnet-50",
    "vit_b16": "vit-base-patch16-224",
    "vit_b32": "vit-base-patch32-224",
    "vit_l16": "vit-large-patch16-224",
    "mlp_mixer_b16": "mixer_b16_224",
    "swin_b": "swin_base_patch4_window7_224",
    "swin_t": "swin_tiny_patch4_window7_224",
}

LOSSES = [
    "NLL",
    "Brier",
    "MMCE",
    "LS-0.05",
    "FLSD-53",
    "FLSD-3",
    "DFL",
    "SoftECE",
    "SmoothSoftECE",
]



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_cifar10_loaders(seed):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    full_train = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

    indices = list(range(len(full_train)))
    random.Random(seed).shuffle(indices)

    val_size = 5000
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_set = Subset(full_train, train_idx)
    val_set = Subset(
        datasets.CIFAR10("./data", train=True, download=False, transform=transform_test),
        val_idx,
    )

    return (
        DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4),
        DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=4),
        DataLoader(test_set, BATCH_SIZE, shuffle=False, num_workers=4),
    )


class ModelWithFeatures(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model = timm.create_model(
            name, pretrained=False, num_classes=NUM_CLASSES
        )
        self.feature_dim = self.model.get_classifier().in_features
        self.model.reset_classifier(0)
        self.classifier_head = nn.Linear(self.feature_dim, NUM_CLASSES)

    def forward(self, x, return_features=False):
        feats = self.model(x)
        logits = self.classifier_head(feats)
        if return_features:
            return logits, feats
        return logits

    def classifier(self, feats):
        return self.classifier_head(feats)



class BrierLoss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))


class LabelSmoothingCE(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze()
        smooth = -log_probs.mean(dim=1)
        return ((1 - self.alpha) * nll + self.alpha * smooth).mean()


class FLSDLoss(nn.Module):
    def __init__(self, gamma_max):
        super().__init__()
        self.gamma_max = gamma_max

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
        gamma = self.gamma_max * (1 - pt)
        ce = -torch.log(pt + 1e-8)
        return ((1 - pt) ** gamma * ce).mean()

class FocalLossSampleDependentGamma(nn.Module):
    def __init__(self, gamma_min: float = 3.0, gamma_max: float = 5.0, eps: float = 1e-12):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(self.eps)
        gamma_i = self.gamma_min + (self.gamma_max - self.gamma_min) * (1.0 - p_t)
        return (-((1.0 - p_t) ** gamma_i) * torch.log(p_t)).mean()

class DualFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=1.0, lam=1.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.lam = lam

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
        pos = -(1 - pt) ** self.gamma_pos * torch.log(pt + 1e-8)
        neg = -pt ** self.gamma_neg * torch.log(1 - pt + 1e-8)
        return (pos + self.lam * neg).mean()


class MMCELoss(nn.Module):
    def __init__(self, sigma=0.4):
        super().__init__()
        self.sigma = sigma

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        acc = (preds == targets).float()
        diff = conf - acc
        dist = (conf.unsqueeze(0) - conf.unsqueeze(1)) ** 2
        kernel = torch.exp(-dist / (2 * self.sigma ** 2))
        return (diff.unsqueeze(0) * diff.unsqueeze(1) * kernel).mean()


class SoftECELoss(nn.Module):
    def __init__(self, bins=15, temp=0.1):
        super().__init__()
        self.centers = torch.linspace(0, 1, bins)
        self.temp = temp

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        acc = (preds == targets).float()
        centers = self.centers.to(conf.device)
        weights = torch.softmax(-(conf.unsqueeze(1) - centers) ** 2 / self.temp, dim=1)
        bin_acc = (weights * acc.unsqueeze(1)).sum(0) / weights.sum(0)
        bin_conf = (weights * conf.unsqueeze(1)).sum(0) / weights.sum(0)
        return torch.abs(bin_acc - bin_conf).mean()


class SmoothSoftECELoss(SoftECELoss):
    pass


def build_loss(name):
    return {
        "NLL": nn.CrossEntropyLoss(),
        "Brier": BrierLoss(),
        "MMCE": MMCELoss(),
        "LS-0.05": LabelSmoothingCE(0.05),
        "FLSD-53": FocalLossSampleDependentGamma(gamma_min=3.0, gamma_max=5.0),
        "FLSD-3": FLSDLoss(3.0),
        "DFL": DualFocalLoss(),
        "SoftECE": SoftECELoss(),
        "SmoothSoftECE": SmoothSoftECELoss(),
    }[name]


def train(model, loss_fn, loader):
    optimizer = optim.SGD(
        model.parameters(), lr=0.1,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[150, 250], gamma=0.1
    )

    model.train()
    for _ in range(EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()


@torch.no_grad()
def collect(model, loader):
    model.eval()
    L, Y, F = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits, feats = model(x, return_features=True)
        L.append(logits.cpu())
        Y.append(y)
        F.append(feats.cpu())
    return torch.cat(L), torch.cat(Y), torch.cat(F)



def main():
    for model_dir, model_name in MODELS.items():
        for seed in SEEDS:
            set_seed(seed)
            train_loader, val_loader, test_loader = get_cifar10_loaders(seed)

            for loss_name in LOSSES:
                print(f"[RUN] {model_dir} | seed={seed} | {loss_name}")

                model = ModelWithFeatures(model_name).to(DEVICE)
                loss_fn = build_loss(loss_name).to(DEVICE)

                train(model, loss_fn, train_loader)

                vL, vY, vF = collect(model, val_loader)
                tL, tY, tF = collect(model, test_loader)

                save_dir = os.path.join(
                    BASE_DIR, model_dir, loss_name, f"seed_{seed}"
                )
                os.makedirs(save_dir, exist_ok=True)

                np.save(f"{save_dir}/val_logits.npy", vL.numpy())
                np.save(f"{save_dir}/val_labels.npy", vY.numpy())
                np.save(f"{save_dir}/val_features.npy", vF.numpy())
                np.save(f"{save_dir}/test_logits.npy", tL.numpy())
                np.save(f"{save_dir}/test_labels.npy", tY.numpy())
                np.save(f"{save_dir}/test_features.npy", tF.numpy())

                print(f"[SAVE] {model_dir} | seed={seed} | {loss_name}")

                RUN_METHODS = ["uncalibrated", "TS", "PTS", "CTS", "ETS", "SMART", "HB", "BBQ", "VS", "GC", "ProCal_DR",
                               "FC", "Spline"]



                with open(f"{save_dir}/results.json", "w") as f:
                    json.dump({"seed": seed, "loss": loss_name}, f)

if __name__ == "__main__":
    main()
