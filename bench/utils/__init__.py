import torch
from tqdm import tqdm
import torchvision

# Import own models
from .models.resnet_imagenet import ResNet_ImageNet as own_resnet50, ResNet101_ImageNet as own_resnet101, ResNet152_ImageNet as own_resnet152
from .models.densenet_imagenet import DenseNet121_ImageNet as own_densenet121, DenseNet169_ImageNet as own_densenet169
from .models.wide_resnet_imagenet import WideResNet_ImageNet as own_wide_resnet
from .models.mobilenet_v2_imagenet import MobileNet_V2_ImageNet as own_mobilenet_v2
from .models.vit_imagenet import ViT_B_16_ImageNet as own_vit_b_16, ViT_B_32_ImageNet as own_vit_b_32, ViT_L_16_ImageNet as own_vit_l_16, ViT_L_32_ImageNet as own_vit_l_32
from .models.swin_imagenet import Swin_B_ImageNet as own_swin_b
from .models.beit_imagenet import BEiT_Base_ImageNet as own_beit_base, BEiT_Large_ImageNet as own_beit_large, BEiTv2_Base_ImageNet as own_beitv2_base
from .models.convnext_imagenet import ConvNext_Tiny_ImageNet as own_convnext_tiny, ConvNext_Base_ImageNet as own_convnext_base, ConvNext_Large_ImageNet as own_convnext_large
# EVA02 with ImageNet-1K fine-tuned classification heads (uses 448x448 input)
from .models.eva_imagenet import EVA02_Base_ImageNet as own_eva02_base, EVA02_Large_ImageNet as own_eva02_large, EVA02_Small_ImageNet as own_eva02_small

# Import CIFAR-compatible versions of the same models
from .models.resnet_cifar import ResNet50_CIFAR as own_resnet50_cifar, ResNet110_CIFAR as own_resnet110_cifar

# For WideResNet and DenseNet, use the original implementations that match saved weights
from .Net.wide_resnet import wide_resnet_cifar as own_wide_resnet_cifar
from .Net.densenet import densenet121 as own_densenet121_cifar

# Import dataloaders
from .Datasets import cifar10 as cifar10
from .Datasets import cifar100 as cifar100
from .Datasets import tiny_imagenet as tiny_imagenet
from .Datasets import imagenet as imagenet
from .Datasets import imagenet_original_val as imagenet_original_val
from .Datasets import imagenet_lt as imagenet_lt
from .Datasets import imagenet_c as imagenet_c
from .Datasets import imagenet_sketch as imagenet_sketch
from .Datasets import iwildcam as iwildcam


# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
    'imagenet': 1000,
    'imagenet_lt': 1000,
    'imagenet_c': 1000,
    'imagenet_sketch': 1000,
    'imagenet_original_val': 1000,
    'iwildcam': 206
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet,
    'imagenet': imagenet,
    'imagenet_lt': imagenet_lt,
    'imagenet_c': imagenet_c,
    'imagenet_sketch': imagenet_sketch,
    'imagenet_original_val': imagenet_original_val,
    'iwildcam': iwildcam
}

# Mapping model name to model function
models_dict = {
    "cifar10":{
        'resnet50': own_resnet50_cifar,
        'resnet110': own_resnet110_cifar,
        'wide_resnet': own_wide_resnet_cifar,
        'densenet121': own_densenet121_cifar
    },
    "cifar100":{
        'resnet50': own_resnet50_cifar,
        'resnet110': own_resnet110_cifar,
        'wide_resnet': own_wide_resnet_cifar,
        'densenet121': own_densenet121_cifar
    },
    "imagenet":{
        'resnet50': own_resnet50,
        'resnet101': own_resnet101,
        'resnet152': own_resnet152,
        'densenet121': own_densenet121,
        'densenet169': own_densenet169,
        'wide_resnet': own_wide_resnet,
        'mobilenet_v2': own_mobilenet_v2,
        'vit_l_16': own_vit_l_16,
        'vit_b_16': own_vit_b_16,
        'vit_b_32': own_vit_b_32,
        'vit_l_32': own_vit_l_32,
        'swin_b': own_swin_b,
        'beit_base': own_beit_base,
        'beit_large': own_beit_large,
        'beitv2_base': own_beitv2_base,
        'convnext_tiny': own_convnext_tiny,
        'convnext_base': own_convnext_base,
        'convnext_large': own_convnext_large,
        'eva02_small': own_eva02_small,
        'eva02_base': own_eva02_base,
        'eva02_large': own_eva02_large
    },
    "imagenet_lt": {
        'resnet50': own_resnet50,
        'resnet101': own_resnet101,
        'resnet152': own_resnet152,
        'densenet121': own_densenet121,
        'densenet169': own_densenet169,
        'wide_resnet': own_wide_resnet,
        'mobilenet_v2': own_mobilenet_v2,
        'vit_l_16': own_vit_l_16,
        'vit_b_16': own_vit_b_16,
        'vit_b_32': own_vit_b_32,
        'vit_l_32': own_vit_l_32,
        'swin_b': own_swin_b,
        'beit_base': own_beit_base,
        'beit_large': own_beit_large,
        'beitv2_base': own_beitv2_base,
        'convnext_tiny': own_convnext_tiny,
        'convnext_base': own_convnext_base,
        'convnext_large': own_convnext_large,
        'eva02_small': own_eva02_small,
        'eva02_base': own_eva02_base,
        'eva02_large': own_eva02_large
    },
    "imagenet_c": {
        'resnet50': own_resnet50,
        'resnet101': own_resnet101,
        'resnet152': own_resnet152,
        'densenet121': own_densenet121,
        'densenet169': own_densenet169,
        'wide_resnet': own_wide_resnet,
        'mobilenet_v2': own_mobilenet_v2,
        'vit_l_16': own_vit_l_16,
        'vit_b_16': own_vit_b_16,
        'vit_b_32': own_vit_b_32,
        'vit_l_32': own_vit_l_32,
        'swin_b': own_swin_b,
        'beit_base': own_beit_base,
        'beit_large': own_beit_large,
        'beitv2_base': own_beitv2_base,
        'convnext_tiny': own_convnext_tiny,
        'convnext_base': own_convnext_base,
        'convnext_large': own_convnext_large,
        'eva02_small': own_eva02_small,
        'eva02_base': own_eva02_base,
        'eva02_large': own_eva02_large
    },
    "imagenet_sketch": {
        'resnet50': own_resnet50,
        'resnet101': own_resnet101,
        'resnet152': own_resnet152,
        'densenet121': own_densenet121,
        'densenet169': own_densenet169,
        'wide_resnet': own_wide_resnet,
        'mobilenet_v2': own_mobilenet_v2,
        'vit_l_16': own_vit_l_16,
        'vit_b_16': own_vit_b_16,
        'vit_b_32': own_vit_b_32,
        'vit_l_32': own_vit_l_32,
        'swin_b': own_swin_b,
        'beit_base': own_beit_base,
        'beit_large': own_beit_large,
        'beitv2_base': own_beitv2_base,
        'convnext_tiny': own_convnext_tiny,
        'convnext_base': own_convnext_base,
        'convnext_large': own_convnext_large,
        'eva02_small': own_eva02_small,
        'eva02_base': own_eva02_base,
        'eva02_large': own_eva02_large
    },
    "imagenet_original_val": {
        'resnet50': own_resnet50,
        'own_resnet50': own_resnet50,
        'resnet101': own_resnet101,
        'resnet152': own_resnet152,
        'densenet121': own_densenet121,
        'densenet169': own_densenet169,
        'wide_resnet': own_wide_resnet,
        'mobilenet_v2': own_mobilenet_v2,
        'vit_l_16': own_vit_l_16,
        'vit_b_16': own_vit_b_16,
        'vit_b_32': own_vit_b_32,
        'vit_l_32': own_vit_l_32,
        'swin_b': own_swin_b,
        'beit_base': own_beit_base,
        'beit_large': own_beit_large,
        'beitv2_base': own_beitv2_base,
        'convnext_tiny': own_convnext_tiny,
        'convnext_base': own_convnext_base,
        'convnext_large': own_convnext_large,
        'eva02_small': own_eva02_small,
        'eva02_base': own_eva02_base,
        'eva02_large': own_eva02_large
    },
    "iwildcam": {
        'resnet50': own_resnet50,
        'vit_b_16': own_vit_b_16,
        'eva02_large': own_eva02_large
    }
}


