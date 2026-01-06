#!/usr/bin/env python3
"""
CIFAR-100 Training Script with Multiple Loss Functions

训练 CIFAR-100 模型（支持多种架构），使用 9 种不同的损失函数：

支持的模型架构：
- resnet50
- resnet152
- densenet121
- wide_resnet50 (Wide-ResNet-50)
- mlp_mixer_b_16 (MLP-Mixer-B-16)
- mobilenetv2
- NLL (CrossEntropy)
- Brier
- MMCE
- LS-0.05 (Label Smoothing)
- FLSD-53 (Focal Loss Sample-Dependent)
- FLSD-3 (Focal Loss γ=3)
- DFL (Dual Focal Loss)
- SoftECE
- SmoothSoftECE

训练完成后保存 logits 和 features，上传到 Hugging Face，然后删除本地文件。
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet152, ResNet152_Weights,
    densenet121, DenseNet121_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
import timm
import random

# 添加路径
sys.path.insert(0, "/home/ruiqi/calibench")
from Component.metrics import (
    ECE, NLL, Accuracy, AdaptiveECE, ClasswiseECE,
    ECEDebiased, ECESweep, BrierLoss as BrierLoss_metric, RBS
)
from Component.metrics.soft_ece import SoftECE
from Component.metrics.SmoothSoftECE import SmoothSoftECE

# 配置
REPO_ID = "CaliBench/CaliBench"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 损失函数列表
LOSS_FUNCTIONS = [
    'nll', 'brier', 'mmce', 'ls-0.05', 
    'flsd-53', 'flsd-3', 'dfl', 'softece', 'smoothsoftece'
]

# Seeds
SEEDS = [1, 2, 3, 4, 5]

# 支持的模型列表
MODELS = ['resnet50', 'resnet152', 'densenet121', 'wide_resnet50', 'mlp_mixer_b_16', 'mobilenetv2']


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(batch_size=128, data_root='/share/datasets/', seed=1):
    """获取 CIFAR-100 数据加载器（训练集、验证集、测试集）"""
    # 数据预处理 (CIFAR-100 归一化参数)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 加载完整训练集
    full_train = datasets.CIFAR100(
        root=data_root, train=True, download=False, transform=train_transform
    )
    
    # 使用固定种子分割训练集和验证集 (80% train, 20% val)
    indices = list(range(len(full_train)))
    random.seed(seed)
    random.shuffle(indices)
    split = int(0.8 * len(full_train))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)
    
    # 测试集
    test_dataset = datasets.CIFAR100(
        root=data_root, train=False, download=False, transform=test_transform
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_model(model_name, num_classes=100):
    """获取指定模型
    
    Args:
        model_name: 模型名称 ('resnet50', 'resnet152', 'densenet121', 'wide_resnet50', 'mlp_mixer_b_16', 'mobilenetv2')
        num_classes: 分类数量，默认100（CIFAR-100）
    """
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet152':
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'wide_resnet50':
        # Wide-ResNet-50 (使用 timm 的 wide_resnet50_2)
        model = timm.create_model('wide_resnet50_2', pretrained=True, num_classes=num_classes)
    elif model_name == 'mlp_mixer_b_16':
        # MLP-Mixer-B-16，调整输入尺寸为32x32（CIFAR）
        model = timm.create_model('mixer_b16_224', pretrained=False, num_classes=num_classes, img_size=32)
    elif model_name == 'mobilenetv2':
        # MobileNetV2
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {MODELS}")
    
    return model


class BrierLoss(nn.Module):
    """Brier Loss"""
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        return torch.mean(torch.sum((probs - targets_one_hot) ** 2, dim=1))


class MMCELoss(nn.Module):
    """Maximum Mean Calibration Error Loss
    
    MMCE uses a kernel-based approach to measure calibration error.
    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """
    def __init__(self, lambda_mmce=1.0):
        super().__init__()
        self.lambda_mmce = lambda_mmce  # Weight for MMCE term
        self.ce = nn.CrossEntropyLoss()
    
    def gaussian_kernel(self, x, y, sigma=1.0):
        """Compute Gaussian kernel between two tensors"""
        # x: (N,), y: (M,)
        # Returns: (N, M)
        x = x.unsqueeze(1)  # (N, 1)
        y = y.unsqueeze(0)  # (1, M)
        return torch.exp(-(x - y) ** 2 / (2 * sigma ** 2))
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - model logits
            targets: (batch_size,) - true class labels
        Returns:
            loss: scalar tensor combining CE and MMCE
        """
        batch_size = logits.size(0)
        
        # Get predicted probabilities
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)  # (batch_size,)
        
        # Binary correctness (1 if correct, 0 if incorrect)
        correct = (predictions == targets).float()  # (batch_size,)
        
        # Cross-entropy loss
        ce_loss = self.ce(logits, targets)
        
        # Compute MMCE term
        if batch_size == 1:
            mmce_loss = torch.tensor(0.0, device=logits.device)
        else:
            # Compute kernel matrix
            # K[i, j] = exp(-(conf[i] - conf[j])^2 / (2 * sigma^2))
            sigma = 1.0  # Kernel bandwidth
            kernel_matrix = self.gaussian_kernel(confidences, confidences, sigma)
            
            # Compute calibration error using kernel trick
            # MMCE^2 = 1/n^2 * sum_i sum_j (conf[i] - correct[i]) * (conf[j] - correct[j]) * K[i, j]
            conf_minus_correct = confidences - correct  # (batch_size,)
            conf_minus_correct_i = conf_minus_correct.unsqueeze(1)  # (batch_size, 1)
            conf_minus_correct_j = conf_minus_correct.unsqueeze(0)  # (1, batch_size)
            
            # Element-wise product: (conf[i] - correct[i]) * (conf[j] - correct[j]) * K[i, j]
            mmce_matrix = conf_minus_correct_i * conf_minus_correct_j * kernel_matrix
            
            # Sum over all pairs and normalize
            mmce_squared = mmce_matrix.sum() / (batch_size ** 2)
            
            # Take square root to get MMCE
            mmce_loss = torch.sqrt(mmce_squared + 1e-8)  # Add small epsilon for numerical stability
        
        # Combine CE and MMCE
        total_loss = ce_loss + self.lambda_mmce * mmce_loss
        
        return total_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss"""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, logits, targets):
        num_classes = logits.size(1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.alpha) * targets_one_hot + self.alpha / num_classes
        return torch.mean(torch.sum(-targets_smooth * log_probs, dim=1))


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class FLSD53Loss(nn.Module):
    """FLSD-53: Sample-Dependent Focal Loss (γ=5/3)"""
    def __init__(self):
        super().__init__()
        self.gamma = 5.0 / 3.0
    
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class FLSD3Loss(nn.Module):
    """FLSD-3: Focal Loss with γ=3"""
    def __init__(self):
        super().__init__()
        self.gamma = 3.0
    
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DFLLoss(nn.Module):
    """Dual Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()



class SoftECELoss(nn.Module):
    """SoftECE Loss combined with CrossEntropy for training"""
    def __init__(self, n_bins=15, sigma=0.05, lambda_softece=1.0):
        super().__init__()
        self.softece = SoftECE(n_bins=n_bins, sigma=sigma)
        self.ce = nn.CrossEntropyLoss()
        self.lambda_softece = lambda_softece
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        softece_loss = self.softece(logits, targets)
        return ce_loss + self.lambda_softece * softece_loss


class SmoothSoftECELoss(nn.Module):
    """SmoothSoftECE Loss combined with CrossEntropy for training"""
    def __init__(self, n_bins=15, sigma=0.05, delta=0.001, mode="softabs", lambda_smoothsoftece=1.0):
        super().__init__()
        self.smoothsoftece = SmoothSoftECE(n_bins=n_bins, sigma=sigma, delta=delta, mode=mode)
        self.ce = nn.CrossEntropyLoss()
        self.lambda_smoothsoftece = lambda_smoothsoftece
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        smoothsoftece_loss = self.smoothsoftece(logits, targets)
        return ce_loss + self.lambda_smoothsoftece * smoothsoftece_loss


def get_loss_function(loss_name):
    """获取损失函数"""
    loss_name = loss_name.lower()
    if loss_name == 'nll':
        return nn.CrossEntropyLoss()
    elif loss_name == 'brier':
        return BrierLoss()
    elif loss_name == 'mmce':
        return MMCELoss()
    elif loss_name == 'ls-0.05':
        return LabelSmoothingLoss(alpha=0.05)
    elif loss_name == 'flsd-53':
        return FLSD53Loss()
    elif loss_name == 'flsd-3':
        return FLSD3Loss()
    elif loss_name == 'dfl':
        return DFLLoss()
    elif loss_name == 'softece':
        return SoftECELoss(n_bins=15, sigma=0.05, lambda_softece=1.0)
    elif loss_name == 'smoothsoftece':
        return SmoothSoftECELoss(n_bins=15, sigma=0.05, delta=0.001, mode="softabs", lambda_smoothsoftece=1.0)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate_model(model, data_loader, device, model_name='resnet50'):
    """评估模型，返回 logits, features, labels 和指标
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        model_name: 模型名称，用于确定特征提取方式
    """
    model.eval()
    all_logits = []
    all_features = []
    all_labels = []
    
    # 根据模型类型获取特征提取函数（去掉最后的分类层）
    def get_features(model, x, model_name):
        if model_name in ['resnet50', 'resnet152']:
            # ResNet: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> flatten
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
        elif model_name == 'densenet121':
            # DenseNet: features -> global_pool -> flatten
            features = model.features(x)
            # DenseNet 的全局池化
            out = torch.relu(features)
            out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
            x = torch.flatten(out, 1)
        elif model_name == 'wide_resnet50':
            # Wide-ResNet-50 (timm): 使用 forward_features 方法
            x = model.forward_features(x)
            # 如果有 global_pool，应用它
            if hasattr(model, 'global_pool'):
                x = model.global_pool(x)
            if hasattr(model, 'flatten'):
                x = model.flatten(x)
            if x.dim() > 2:
                x = torch.flatten(x, 1)
        elif model_name == 'mlp_mixer_b_16':
            # MLP-Mixer-B-16 (timm): 使用 forward_features
            x = model.forward_features(x)
            # MLP-Mixer 输出 [B, N, C] 格式，使用全局平均池化
            if x.dim() == 3:
                x = x.mean(dim=1)  # 全局平均池化 [B, N, C] -> [B, C]
            if x.dim() > 2:
                x = torch.flatten(x, 1)
        elif model_name == 'mobilenetv2':
            # MobileNetV2: features -> adaptive_avg_pool2d -> flatten
            x = model.features(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
        return x
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 提取特征
            features = get_features(model, inputs, model_name)
            all_features.append(features.cpu())
            
            # 获取 logits
            outputs = model(inputs)
            all_logits.append(outputs.cpu())
            all_labels.append(targets.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # 计算指标
    probs = torch.softmax(logits, dim=1)
    labels_tensor = labels.to(device)
    probs_tensor = probs.to(device)
    
    # 初始化所有metric对象
    ece_metric = ECE(n_bins=15).to(device)
    nll_metric = NLL().to(device)
    acc_metric = Accuracy().to(device)
    ada_ece_metric = AdaptiveECE(n_bins=15).to(device)
    cece_metric = ClasswiseECE(n_bins=15).to(device)
    ece_debiased_metric = ECEDebiased(n_bins=15).to(device)
    ece_sweep_metric = ECESweep().to(device)
    brier_metric = BrierLoss_metric().to(device)
    rbs_metric = RBS().to(device)
    
    # 计算所有metrics
    ece = ece_metric(labels=labels_tensor, softmaxes=probs_tensor)
    nll = nll_metric(labels=labels_tensor, softmaxes=probs_tensor)
    acc = acc_metric(labels=labels_tensor, softmaxes=probs_tensor)
    ada_ece = ada_ece_metric(labels=labels_tensor, softmaxes=probs_tensor)
    cece = cece_metric(labels=labels_tensor, softmaxes=probs_tensor)
    ece_debiased = ece_debiased_metric(labels=labels_tensor, softmaxes=probs_tensor)
    ece_sweep = ece_sweep_metric(labels=labels_tensor, softmaxes=probs_tensor)
    brier = brier_metric(labels=labels_tensor, softmaxes=probs_tensor)
    rbs = rbs_metric(labels=labels_tensor, softmaxes=probs_tensor)
    
    # 处理返回类型（有些返回tensor需要.item()，有些已经是float）
    metrics = {
        'ECE': float(ece),
        'Accuracy': float(acc),
        'AdaECE': float(ada_ece),
        'CECE': float(cece),
        'NLL': float(nll),
        'ECE_debiased': float(ece_debiased),
        'ECE_sweep': float(ece_sweep),
        'Brier': float(brier.item() if torch.is_tensor(brier) else brier),
        'RBS': float(rbs.item() if torch.is_tensor(rbs) else rbs)
    }
    
    return logits, features, labels, metrics


def print_metrics(metrics, split_name):
    """打印metrics"""
    print(f'\n{split_name} Metrics:')
    print(f"  ECE:            {metrics['ECE']:.6f}")
    print(f"  Accuracy:       {metrics['Accuracy']:.6f}")
    print(f"  AdaECE:         {metrics['AdaECE']:.6f}")
    print(f"  CECE:           {metrics['CECE']:.6f}")
    print(f"  NLL:            {metrics['NLL']:.6f}")
    print(f"  ECE_debiased:   {metrics['ECE_debiased']:.6f}")
    print(f"  ECE_sweep:      {metrics['ECE_sweep']:.6f}")
    print(f"  Brier:          {metrics['Brier']:.6f}")
    print(f"  RBS:            {metrics['RBS']:.6f}")


def save_metrics_to_csv(save_dir, val_metrics, test_metrics, loss_name, seed, model_name):
    """保存metrics到CSV文件"""
    # 创建DataFrame
    metrics_data = {
        'Split': ['val', 'test'],
        'Loss': [loss_name, loss_name],
        'Seed': [seed, seed],
        'Model': [model_name, model_name],
        'ECE': [val_metrics['ECE'], test_metrics['ECE']],
        'Accuracy': [val_metrics['Accuracy'], test_metrics['Accuracy']],
        'AdaECE': [val_metrics['AdaECE'], test_metrics['AdaECE']],
        'CECE': [val_metrics['CECE'], test_metrics['CECE']],
        'NLL': [val_metrics['NLL'], test_metrics['NLL']],
        'ECE_debiased': [val_metrics['ECE_debiased'], test_metrics['ECE_debiased']],
        'ECE_sweep': [val_metrics['ECE_sweep'], test_metrics['ECE_sweep']],
        'Brier': [val_metrics['Brier'], test_metrics['Brier']],
        'RBS': [val_metrics['RBS'], test_metrics['RBS']]
    }
    
    df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(save_dir, 'metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f'✓ Metrics saved to {csv_path}')


def run_command(cmd, description):
    """运行命令"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"\n失败 (退出码: {result.returncode})")
        return False
    else:
        print(f"\n成功")
        return True


def upload_to_hf(model_name, loss_name, seed, local_path):
    """上传到 Hugging Face"""
    hf_path = f"cifar100/{model_name}/{loss_name}/seed_{seed}"
    
    if not os.path.exists(local_path):
        print(f"警告: 本地路径不存在: {local_path}")
        return False
    
    cmd = [
        'hf', 'upload',
        '--repo-type', 'dataset',
        REPO_ID,
        local_path,
        hf_path,
    ]
    
    return run_command(cmd, f"上传 {model_name}/{loss_name}/seed_{seed} 到 Hugging Face")


def delete_files(save_dir):
    """删除 logits 和 features 文件"""
    files_to_delete = [
        'val_logits.npy', 'val_features.npy', 'val_labels.npy',
        'test_logits.npy', 'test_features.npy', 'test_labels.npy'
    ]
    
    deleted_count = 0
    for filename in files_to_delete:
        filepath = os.path.join(save_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            deleted_count += 1
    
    print(f"✓ 删除了 {deleted_count} 个文件")
    return deleted_count


def train_and_save(
    loss_name, seed, model_name='resnet50', num_epochs=350, batch_size=128, 
    data_root='/share/datasets/', save_dir_base='./checkpoints',
    upload=True, delete_after_upload=True
):
    """训练模型并保存结果
    
    Args:
        loss_name: 损失函数名称
        seed: 随机种子
        model_name: 模型名称
        num_epochs: 训练轮数
        batch_size: 批次大小
        data_root: 数据根目录
        save_dir_base: 保存目录基础路径
        upload: 是否上传到 Hugging Face
        delete_after_upload: 上传后是否删除文件
    """
    # 设置种子
    set_seed(seed)
    
    # 创建保存目录
    save_dir = os.path.join(save_dir_base, 'cifar100', model_name, loss_name, f'seed_{seed}')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training: {loss_name.upper()} | Seed: {seed}")
    print(f"{'='*80}")
    print(f"Save directory: {save_dir}")
    
    # 数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size, data_root=data_root, seed=seed
    )
    
    # 模型
    model = get_model(model_name, num_classes=100).to(DEVICE)
    
    # 损失函数
    criterion = get_loss_function(loss_name).to(DEVICE)
    
    # 优化器和学习率调度器
    optimizer = optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[150, 250], gamma=0.1
    )
    
    # 训练
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    
    # 评估并保存
    print('\nEvaluating on validation set...')
    val_logits, val_features, val_labels, val_metrics = evaluate_model(model, val_loader, DEVICE, model_name)
    print_metrics(val_metrics, 'Validation')
    
    print('\nEvaluating on test set...')
    test_logits, test_features, test_labels, test_metrics = evaluate_model(model, test_loader, DEVICE, model_name)
    print_metrics(test_metrics, 'Test')
    
    # 保存文件
    print('\nSaving files...')
    np.save(os.path.join(save_dir, 'val_logits.npy'), val_logits.numpy())
    np.save(os.path.join(save_dir, 'val_features.npy'), val_features.numpy())
    np.save(os.path.join(save_dir, 'val_labels.npy'), val_labels.numpy())
    np.save(os.path.join(save_dir, 'test_logits.npy'), test_logits.numpy())
    np.save(os.path.join(save_dir, 'test_features.npy'), test_features.numpy())
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels.numpy())
    
    # 保存结果 JSON
    results = {
        'loss_function': loss_name,
        'seed': seed,
        'model': model_name,
        'dataset': 'cifar100',
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存metrics到CSV
    save_metrics_to_csv(save_dir, val_metrics, test_metrics, loss_name, seed, model_name)
    
    print('✓ All files saved')
    
    # 上传到 Hugging Face
    if upload:
        print('\nUploading to Hugging Face...')
        upload_success = upload_to_hf(model_name, loss_name, seed, save_dir)
        
        if upload_success and delete_after_upload:
            print('\nDeleting local files...')
            delete_files(save_dir)
    
    print(f'\n✓ Training completed: {loss_name.upper()} | Seed: {seed}')
    return results


def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-100 with different loss functions')
    parser.add_argument('--loss', type=str, required=True, choices=LOSS_FUNCTIONS,
                        help='Loss function to use')
    parser.add_argument('--seed', type=int, required=True, choices=SEEDS,
                        help='Random seed (1-5)')
    parser.add_argument('--model', type=str, required=True, choices=MODELS,
                        help=f'Model architecture ({", ".join(MODELS)})')
    parser.add_argument('--data_root', type=str, default='/share/datasets/',
                        help='Root directory for CIFAR-100 dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Base directory to save results')
    parser.add_argument('--num_epochs', type=int, default=350,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--no_upload', action='store_true',
                        help='Skip uploading to Hugging Face')
    parser.add_argument('--keep_files', action='store_true',
                        help='Keep files after uploading')
    
    args = parser.parse_args()
    
    train_and_save(
        loss_name=args.loss,
        seed=args.seed,
        model_name=args.model,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        data_root=args.data_root,
        save_dir_base=args.save_dir,
        upload=not args.no_upload,
        delete_after_upload=not args.keep_files
    )


if __name__ == '__main__':
    main()

