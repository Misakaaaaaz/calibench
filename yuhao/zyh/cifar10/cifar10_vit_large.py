import os
import sys

import datetime



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from yuhao.zyh.cifar10.models.eva02_cifar import EVA02_Base_CIFAR10, EVA02_Small_CIFAR10, EVA02_Large_CIFAR10

from collections import deque
class TeeLogger(object):
    def __init__(self, log_dir="logs", max_lines=2000):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        self.filename = os.path.join(log_dir, f"train_{time_str}_pid{pid}.log")
        self.log = open(self.filename, "w", buffering=1)
        self.terminal = sys.stdout
        self.max_lines = max_lines
        self.lines_buffer = deque(maxlen=max_lines)
        self.temp_buffer = ""

        print(f"[Logger Init] Save logs to: {self.filename} (Keeping last {max_lines} lines)")

    def write(self, message):
        self.terminal.write(message)
        self.temp_buffer += message

        if '\n' in self.temp_buffer:

            parts = self.temp_buffer.split('\n')
            for part in parts[:-1]:
                self.lines_buffer.append(part + '\n')
            self.temp_buffer = parts[-1]
            self._rewrite_file()

    def _rewrite_file(self):
        self.log.seek(0)
        self.log.truncate()
        self.log.writelines(self.lines_buffer)
        self.log.write(self.temp_buffer)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger()
sys.stderr = sys.stdout

import argparse
import json

from typing import Optional, Dict
import random
import numpy as np
import copy

from Component.metrics import *
import torch
import torch.backends.cudnn as cudnn
from Datasets.cifar10 import get_train_valid_loader as cifar10_train_valid_loader
from Datasets.cifar10 import get_test_loader as cifar10_test_loader
from metrics import test_classification_net
from models.densenet_cifar import DenseNet121_CIFAR
from models.resnet_cifar import ResNet50_CIFAR,  ResNet152_CIFAR
from models.wide_resnet_cifar import WideResNet_CIFAR
from torch import optim
import torch.nn.functional as F
from train_utils import train_single_epoch, test_single_epoch
import logging
from functools import partial
from yuhao.zyh.cifar10.models.vit_cifar import ViT_B_16_CIFAR, ViT_B_32_CIFAR, ViT_L_16_CIFAR
models={
    "resnet50": ResNet50_CIFAR,
    "resnet152": ResNet152_CIFAR,
    "densenet121": DenseNet121_CIFAR,
    "wide_resnet": WideResNet_CIFAR,
    "vit_b_16": partial(ViT_B_16_CIFAR, image_size=224),
    "vit_b_32": partial(ViT_B_32_CIFAR, image_size=224),
    "vit_l_16": partial(ViT_L_16_CIFAR, image_size=224),
    "eva02_base": EVA02_Base_CIFAR10,
    "eva02_small": EVA02_Small_CIFAR10,
    "eva02_large": EVA02_Large_CIFAR10
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_all_metrics(
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        n_bins: int = 15
) -> Dict[str, float]:
    """
    Compute all available metrics for the given logits/probs and labels.

    Args:
        labels (torch.Tensor): Target labels
        logits (torch.Tensor, optional): Input logits before softmax
        probs (torch.Tensor, optional): Probability distributions (softmax outputs)
        n_bins (int, optional): Number of bins for ECE calculation. Defaults to 15.

    Returns:
        Dict[str, float]: Dictionary containing all metric values
    """
    if logits is None and probs is None:
        raise ValueError("Either logits or probs must be provided")

    device = labels.device

    # If probs not provided, compute from logits
    if probs is None:
        probs = F.softmax(logits, dim=1)

    # Initialize metrics
    metrics = {
        'ece': ECE(n_bins=n_bins),
        'accuracy': Accuracy(),
        'adaptive_ece': AdaptiveECE(n_bins=n_bins),
        'classwise_ece': ClasswiseECE(n_bins=n_bins),
        'nll': NLL(),
        'ece_debiased': ECEDebiased(n_bins=n_bins),
        'ece_sweep': ECESweep(),
        'Brier': BrierLoss(),
        'rbs': RBS()
    }

    # Set up basic logging
    logger = logging.getLogger(__name__)

    results = {}
    for name, metric in metrics.items():
        metric = metric.to(device)
        try:
            if name in ['nll', 'rbs', 'Brier']:
                if probs is not None:
                    value = metric(softmaxes=probs, labels=labels)
                elif logits is not None:
                    value = metric(logits=logits, labels=labels)
            elif name in ['ece', 'adaptive_ece', 'classwise_ece', 'ece_debiased', 'ece_sweep', 'accuracy']:
                value = metric(softmaxes=probs, labels=labels)
            else:
                logger.warning(f"Unknown metric type: {name}")
                continue

            # Convert to float if it's a tensor
            if torch.is_tensor(value):
                value = value.item()
            results[name] = value
        except Exception as e:
            logger.warning(f"Failed to compute {name}: {str(e)}")
            results[name] = None
            continue

    return results


def get_all_metrics(
        device,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        n_bins: int = 15
) -> Dict[str, float]:
    """
    Get all metrics in a dictionary format compatible with the standard results structure.

    Args:
        labels (torch.Tensor): Target labels
        logits (torch.Tensor, optional): Input logits before softmax
        probs (torch.Tensor, optional): Probability distributions (softmax outputs)
        n_bins (int, optional): Number of bins for ECE calculation. Defaults to 15.

    Returns:
        Dict[str, float]: Dictionary containing the 8 standard metrics:
        {
            'ece': float,
            'accuracy': float,
            'adaece': float,
            'cece': float,
            'nll': float,
            'ece_debiased': float,
            'ece_sweep': float,
            'rbs': float
        }
    """
    # Move tensors to device
    logits = logits.to(device)
    labels = labels.to(device)

    # Determine if logits are actually probabilities
    is_probs = (logits.dim() == 2 and
                torch.allclose(logits.sum(dim=1), torch.ones(logits.size(0), device=device), atol=1e-3))
    metrics = compute_all_metrics(labels=labels, logits=logits if not is_probs else None,
                                  probs=logits if is_probs else None, n_bins=n_bins)
    return {
        'ece': metrics.get('ece', None),
        'accuracy': metrics.get('accuracy', None),
        'adaece': metrics.get('adaptive_ece', None),
        'cece': metrics.get('classwise_ece', None),
        'nll': metrics.get('nll', None),
        'ece_debiased': metrics.get('ece_debiased', None),
        'ece_sweep': metrics.get('ece_sweep', None),
        'Brier': metrics.get('Brier', None),
        'rbs': metrics.get('rbs', None)
    }

def loss_function_save_name(loss_function,
                            scheduled=False,
                            gamma=1.0,
                            gamma1=1.0,
                            gamma2=1.0,
                            gamma3=1.0,
                            lamda=1.0):
    res_dict = {
        'cross_entropy': 'cross_entropy',
        'focal_loss': 'focal_loss_gamma_' + str(gamma),
        'focal_loss_adaptive': 'focal_loss_adaptive_gamma_' + str(gamma),
        'mmce': 'mmce_lamda_' + str(lamda),
        'mmce_weighted': 'mmce_weighted_lamda_' + str(lamda),
        'brier_score': 'brier_score'
    }
    if (loss_function == 'focal_loss' and scheduled == True):
        res_str = 'focal_loss_scheduled_gamma_' + str(gamma1) + '_' + str(gamma2) + '_' + str(gamma3)
    else:
        res_str = res_dict[loss_function]
    return res_str

def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = '/ssd/data'
    train_batch_size = 128
    test_batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    loss = "cross_entropy"
    gamma = 1.0
    gamma2 = 1.0
    gamma3 = 1.0
    lamda = 1.0
    weight_decay = 5e-4
    log_interval = 50
    save_interval = 50
    save_loc = "./cifar10"
    model_name = None
    saved_model_name = "resnet50_cross_entropy_350.model"
    load_loc = "./cifar10"
    model = "resnet50"
    epoch = 350
    first_milestone = 150 #Milestone for change in lr
    second_milestone = 250 #Milestone for change in lr
    gamma_schedule_step1 = 100
    gamma_schedule_step2 = 250

    parser = argparse.ArgumentParser(
        description="Training for calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)

    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("--load", action="store_true", dest="load",
                        help="Load from pretrained model")
    parser.set_defaults(load=False)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("-e", type=int, default=epoch, dest="epoch",
                        help='Number of training epochs')
    parser.add_argument("--lr", type=float, default=learning_rate,
                        dest="learning_rate", help='Learning rate')
    parser.add_argument("--mom", type=float, default=momentum,
                        dest="momentum", help='Momentum')
    parser.add_argument("--nesterov", action="store_true", dest="nesterov",
                        help="Whether to use nesterov momentum in SGD")
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=weight_decay,
                        dest="weight_decay", help="Weight Decay")
    parser.add_argument("--opt", type=str, default=optimiser,
                        dest="optimiser",
                        help='Choice of optimisation algorithm')

    parser.add_argument("--loss", type=str, default=loss, dest="loss_function",
                        help="Loss function to be used for training")
    parser.add_argument("--loss-mean", action="store_true", dest="loss_mean",
                        help="whether to take mean of loss instead of sum to train")
    parser.set_defaults(loss_mean=False)
    parser.add_argument("--gamma", type=float, default=gamma,
                        dest="gamma", help="Gamma for focal components")
    parser.add_argument("--gamma2", type=float, default=gamma2,
                        dest="gamma2", help="Gamma for different focal components")
    parser.add_argument("--gamma3", type=float, default=gamma3,
                        dest="gamma3", help="Gamma for different focal components")
    parser.add_argument("--lamda", type=float, default=lamda,
                        dest="lamda", help="Regularization factor")
    parser.add_argument("--gamma-schedule", type=int, default=0,
                        dest="gamma_schedule", help="Schedule gamma or not")
    parser.add_argument("--gamma-schedule-step1", type=int, default=gamma_schedule_step1,
                        dest="gamma_schedule_step1", help="1st step for gamma schedule")
    parser.add_argument("--gamma-schedule-step2", type=int, default=gamma_schedule_step2,
                        dest="gamma_schedule_step2", help="2nd step for gamma schedule")

    parser.add_argument("--log-interval", type=int, default=log_interval,
                        dest="log_interval", help="Log Interval on Terminal")
    parser.add_argument("--save-interval", type=int, default=save_interval,
                        dest="save_interval", help="Save Interval on Terminal")
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to export the model')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name",
                        help='name of the model')
    parser.add_argument("--load-path", type=str, default=load_loc,
                        dest="load_loc",
                        help='Path to load the model from')

    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to train')
    parser.add_argument("--first-milestone", type=int, default=first_milestone,
                        dest="first_milestone", help="First milestone to change lr")
    parser.add_argument("--second-milestone", type=int, default=second_milestone,
                        dest="second_milestone", help="Second milestone to change lr")

    return parser.parse_args()


import pandas as pd
import os


def save_metrics_to_csv(metrics, dataset, model, loss, seed, file_path='cifar10_results.csv'):
    """
    保存为 CSV
    1. 不截断小数（保留完整精度）
    2. 加上百分号 "%"
    """

    # 需要转换的字段
    percent_keys = ['ece', 'accuracy', 'adaece', 'ece_debiased', 'ece_sweep']

    processed_metrics = {}
    for k, v in metrics.items():
        if v is None:
            processed_metrics[k] = ""
        elif k in percent_keys:
            # 【核心修改】
            # 1. v * 100 算出百分数数值
            # 2. str(...) 转为字符串，保留 Python float 的全部默认精度
            # 3. + '%' 加上百分号
            # 结果示例：0.123456789 -> "12.3456789%"
            processed_metrics[k] = str(v * 100) + '%'
        else:
            # 其他字段保持原样
            processed_metrics[k] = v

    # 准备行数据
    row_data = {
        'Dataset': dataset,
        'Model': model,
        'Loss': loss,
        'Seed': seed,
        'ECE': processed_metrics.get('ece'),
        'Accuracy': processed_metrics.get('accuracy'),
        'AdaECE': processed_metrics.get('adaece'),
        'CECE': processed_metrics.get('cece'),
        'NLL': processed_metrics.get('nll'),
        'ECE_debiased': processed_metrics.get('ece_debiased'),
        'ECE_sweep': processed_metrics.get('ece_sweep'),
        'Brier': processed_metrics.get('Brier'),
        'RBS': processed_metrics.get('rbs')
    }

    # 创建 DataFrame
    df = pd.DataFrame([row_data])

    # 指定列顺序
    columns_order = [
        'Dataset', 'Model', 'Loss', 'Seed',
        'ECE', 'Accuracy', 'AdaECE', 'CECE',
        'NLL', 'ECE_debiased', 'ECE_sweep', 'Brier', 'RBS'
    ]
    df = df[columns_order]

    # 保存
    file_exists = os.path.isfile(file_path)
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)

    print(f"Results saved to {file_path}")


def train_cifar10(args,data_root,seed,model_name,device):
    save_dir = os.path.join(
        str(args.save_loc), str(args.model_name), str(args.loss_function), f"seed_{seed}"
    )
    files_exist = (os.path.exists(os.path.join(save_dir, "val_logits.npy")) and
                   os.path.exists(os.path.join(save_dir, "val_labels.npy")) and
                   os.path.exists(os.path.join(save_dir, "val_features.npy")) and
                   os.path.exists(os.path.join(save_dir, "test_logits.npy")) and
                   os.path.exists(os.path.join(save_dir, "test_labels.npy")) and
                   os.path.exists(os.path.join(save_dir, "test_features.npy")))
    if files_exist:
        print("Jump: " + str(save_dir))
        return
    resize=None
    if args.model_name in ['vit_b_16','vit_b_32','vit_l_16']:
        resize = 224
    if args.model_name in ['eva02_small']:
        resize = 336
    if args.model_name in ['eva02_base','eva02_large']:
        resize = 448

    train_loader, val_loader = cifar10_train_valid_loader(
        root=data_root,
        batch_size=64,
        shuffle=True,
        random_seed=seed,
        augment=True,
        resize=resize
    )

    test_loader = cifar10_test_loader(
        root=data_root,
        batch_size=64,
        shuffle=False,
        resize=resize
    )

    net = models[model_name](num_classes=10)

    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

    start_epoch = 0
    num_epochs = 350

    opt_params = net.parameters()
    optimizer = optim.SGD(opt_params,lr=0.1,momentum=0.9,weight_decay=5e-4,nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250],gamma=0.1)

    training_set_loss = {}
    val_set_loss = {}
    test_set_loss = {}
    val_set_err = {}

    for epoch in range(0, start_epoch):
        scheduler.step()

    best_val_acc = 0
    best_val_logit = None
    best_val_feature = None
    best_val_label = None
    best_test_logit = None
    best_test_feature = None
    best_test_label = None
    best_model_wts = None
    best_epoch_idx = 0
    for epoch in range(start_epoch, num_epochs):

        if (args.loss_function == 'FLSD-3' and args.gamma_schedule == 1):
            if (epoch < args.gamma_schedule_step1):
                gamma = args.gamma
            elif (epoch >= args.gamma_schedule_step1 and epoch < args.gamma_schedule_step2):
                gamma = args.gamma2
            else:
                gamma = args.gamma3
        else:
            gamma = args.gamma

        train_loss = train_single_epoch(epoch,
                                        net,
                                        train_loader,
                                        optimizer,
                                        device,
                                        loss_function=args.loss_function,
                                        gamma=gamma,
                                        lamda=args.lamda,
                                        loss_mean=args.loss_mean)
        scheduler.step()
        val_loss,val_logit, val_feature, val_label = test_single_epoch(epoch,
                                     net,
                                     val_loader,
                                     device,
                                     loss_function=args.loss_function,
                                     gamma=gamma,
                                     lamda=args.lamda)
        test_loss,test_logit, test_feature, test_label = test_single_epoch(epoch,
                                      net,
                                      test_loader,
                                      device,
                                      loss_function=args.loss_function,
                                      gamma=gamma,
                                      lamda=args.lamda)
        _, val_acc, _, _, _ = test_classification_net(net, val_loader, device)

        training_set_loss[epoch] = train_loss
        val_set_loss[epoch] = val_loss
        test_set_loss[epoch] = test_loss
        val_set_err[epoch] = 1 - val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('New best error: %.4f' % (1 - best_val_acc))
            best_val_logit = val_logit
            best_val_feature = val_feature
            best_val_label = val_label
            best_test_logit = test_logit
            best_test_feature = test_feature
            best_test_label = test_label
            best_model_wts = copy.deepcopy(net.state_dict())
            best_epoch_idx = epoch + 1

            '''
            save_name = args.save_loc + \
                        args.model_name + '_' + \
                        loss_function_save_name(args.loss_function, args.gamma_schedule, gamma, args.gamma, args.gamma2,
                                                args.gamma3, args.lamda) + \
                        '_best_' + \
                        str(epoch + 1) + '.model'
            torch.save(net.state_dict(), save_name)
            '''
        '''
        if (epoch + 1) % args.save_interval == 0:
            save_name = args.save_loc + \
                        args.model_name + '_' + \
                        loss_function_save_name(args.loss_function, args.gamma_schedule, gamma, args.gamma, args.gamma2,
                                                args.gamma3, args.lamda) + \
                        '_' + str(epoch + 1) + '.model'
            torch.save(net.state_dict(), save_name)
        '''

    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/val_logits.npy", best_val_logit)
    np.save(f"{save_dir}/val_labels.npy", best_val_label)
    np.save(f"{save_dir}/val_features.npy", best_val_feature)
    np.save(f"{save_dir}/test_logits.npy", best_test_logit)
    np.save(f"{save_dir}/test_labels.npy", best_test_label)
    np.save(f"{save_dir}/test_features.npy", best_test_feature)


    model_loc=os.path.join(
        args.save_loc, args.model_name, args.loss_function, f"seed_{seed}",f"best_{str(best_epoch_idx)}.model"
    )
    torch.save(best_model_wts, str(model_loc))

    print(f"[SAVE] {args.model_name} | seed={seed} | {args.loss_function}")

    val_logits_tensor = torch.tensor(best_val_logit, dtype=torch.float32)
    val_labels_tensor = torch.tensor(best_val_label, dtype=torch.long)
    val_features_tensor = torch.tensor(best_val_feature, dtype=torch.float32)

    test_logits_tensor = torch.tensor(best_test_logit, dtype=torch.float32)
    test_labels_tensor = torch.tensor(best_test_label, dtype=torch.long)
    test_features_tensor = torch.tensor(best_test_feature, dtype=torch.float32)

    all_metrics = get_all_metrics(device,logits=test_logits_tensor, labels=test_labels_tensor, n_bins=15)
    save_metrics_to_csv(all_metrics,'cifar10',args.model_name,args.loss_function,seed,'cifar10_result.csv')


    with open(os.path.join( save_dir, '_train_loss.json'), 'a') as f:
        json.dump(training_set_loss, f)

    with open(os.path.join( save_dir, '_val_loss.json'), 'a') as fv:
        json.dump(val_set_loss, fv)

    with open(os.path.join( save_dir, '_test_loss.json'), 'a') as ft:
        json.dump(test_set_loss, ft)

    with open(os.path.join( save_dir, '_val_err.json'), 'a') as ft:
        json.dump(val_set_err, ft)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for seed in [1]:
        set_seed(seed)
        """
        ['vit_b_16','vit_b_32','vit_l_16','swin_b','beit_base', 'beit_large',  'convnext_tiny', 'convnext_base', 'convnext_large',
                  'eva02_small', 'eva02_base',  'mobilenet_v2', 'mlp_mixer_b16','eva02_large']:
        """
        for model_name in ["vit_l_16"]:
            origin = parseArgs()
            origin.model_name=model_name

            arg1 = copy.deepcopy(origin)
            arg1.loss_function = 'NLL'
            train_cifar10(arg1,arg1.dataset_root,seed,arg1.model_name, device)

            arg2=copy.deepcopy(origin)
            arg2.loss_function = 'Brier'
            train_cifar10(arg2, arg2.dataset_root, seed, arg2.model_name, device)

            arg3 = copy.deepcopy(origin)
            arg3.loss_function = 'MMCE'
            arg3.lamda = 2.0
            train_cifar10(arg3, arg3.dataset_root, seed, arg3.model_name, device)

            arg4 = copy.deepcopy(origin)
            arg4.loss_function = 'LS-0.05'
            train_cifar10(arg4, arg4.dataset_root, seed, arg4.model_name, device)

            arg5 = copy.deepcopy(origin)
            arg5.loss_function = 'FLSD-53'
            arg5.gamma=3.0
            train_cifar10(arg5, arg5.dataset_root, seed, arg5.model_name, device)

            arg6 = copy.deepcopy(origin)
            arg6.loss_function = 'FLSD-3'
            arg6.gamma = 3.0
            train_cifar10(arg6, arg6.dataset_root, seed, arg6.model_name, device)

            arg7 = copy.deepcopy(origin)
            gammamap7 = {
                "resnet_50": 5,
                "wide_resnet": 2.6,
                "densenet121": 5,
            }
            if arg7.model_name in gammamap7:
                arg7.loss_function = 'DFL'
                arg7.gamma = gammamap7.get(model_name, 5)
                train_cifar10(arg7, arg7.dataset_root, seed, arg7.model_name, device)

            """
            arg8 = copy.deepcopy(origin)
            arg8.loss_function = 'SoftECE'
            train_cifar10(arg8, arg8.dataset_root, seed, arg8.model_name, device)

            arg9 = copy.deepcopy(origin)
            arg9.loss_function = 'SmoothSoftECE'
            train_cifar10(arg9, arg9.dataset_root, seed, arg9.model_name, device)
            """
            arg10 = copy.deepcopy(origin)
            arg10.loss_function = 'MSE'
            train_cifar10(arg10, arg10.dataset_root, seed, arg10.model_name, device)

if __name__ == "__main__":
    main()



