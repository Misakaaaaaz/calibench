"""
Model utility functions for loading and configuring models
"""
import os
import torch

from bench.utils import models_dict, dataset_num_classes


def get_model_normalization(model_name):
    """
    Get the correct normalization parameters for each model.

    Different pretrained models are trained with different normalization:
    - Standard ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - BEiT-style: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    - CLIP-style (EVA02): mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]

    Returns:
        tuple: (mean, std) for normalization
    """
    # BEiT-style normalization (mean=0.5, std=0.5)
    if model_name in ['beit_base', 'beit_large', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
        return ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # CLIP-style normalization (EVA02)
    elif model_name in ['eva02_base', 'eva02_large', 'eva02_small']:
        return ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

    # Standard ImageNet normalization (default)
    # ResNet, BEiTv2, Swin, ConvNext, DenseNet, MobileNet, WideResNet, etc.
    else:
        return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_model_input_size(model_name):
    """
    Get the correct input image size for each model.

    Most models use 224x224, but some require different sizes:
    - EVA02-Small: 336x336
    - EVA02-Base: 448x448
    - EVA02-Large: 448x448

    Returns:
        int: Input image size (height/width, assumes square images)
    """
    if model_name in ['eva02_small']:
        return 336
    elif model_name in ['eva02_base', 'eva02_large']:
        return 448
    else:
        return 224  # Default for most models


def create_model(args, model_name, dataset_name, device):
    """
    Helper function to create and load a model consistently across the codebase
    """
    # Get model class from models_dict
    model_fn = models_dict.get(dataset_name, {}).get(model_name)

    # Create appropriate model based on dataset
    if dataset_name in ['imagenet', 'imagenet_c', 'imagenet_sketch', 'imagenet_lt', 'imagenet_original_val', 'iwildcam']:
        # For ImageNet and iWildCam, decide whether to use pretrained or load underfitted weights
        if getattr(args, 'use_underfitted', False):
            # Create model without pretrained weights, then load underfitted weights
            model = model_fn(pretrained=False).to(device)

            # Load underfitted weights for ImageNet
            try:
                underfitted_epochs = getattr(args, 'underfitted_epochs', 5)
                #TODO: change path
                weight_path = f"/home/haolan/pretrained_weights/{dataset_name}_{model_name}_cross_entropy_epochs{underfitted_epochs}_seed{args.random_seed}.model"

                if os.path.exists(weight_path):
                    checkpoint = torch.load(weight_path)
                    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

                    if any('module' in key for key in state_dict.keys()):
                        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
                        model.load_state_dict(state_dict)
                    else:
                        model.load_state_dict(state_dict)

                    print(f"✓ Loaded underfitted {dataset_name} {model_name} ({underfitted_epochs} epochs)")
                else:
                    print(f"✗ Underfitted weights not found, using pretrained weights")
                    model = model_fn(pretrained=True).to(device)
            except Exception as e:
                print(f"✗ Error loading underfitted weights: {e}")
                model = model_fn(pretrained=True).to(device)
        else:
            # Use standard pretrained ImageNet weights
            model = model_fn(pretrained=True).to(device)

        # For iWildCam, replace final layer to match number of classes (206 instead of 1000)
        if dataset_name == 'iwildcam':
            num_classes = dataset_num_classes['iwildcam']
            # Handle different model architectures
            if hasattr(model, 'fc'):
                # ResNet, DenseNet, etc.
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, num_classes).to(device)
            elif hasattr(model, 'head'):
                # ViT, DeiT, etc.
                in_features = model.head.in_features
                model.head = torch.nn.Linear(in_features, num_classes).to(device)
            elif hasattr(model, 'heads'):
                # Some transformers use 'heads'
                if hasattr(model.heads, 'head'):
                    in_features = model.heads.head.in_features
                    model.heads.head = torch.nn.Linear(in_features, num_classes).to(device)
            elif hasattr(model, 'classifier'):
                # MobileNet, EfficientNet, etc.
                if isinstance(model.classifier, torch.nn.Linear):
                    in_features = model.classifier.in_features
                    model.classifier = torch.nn.Linear(in_features, num_classes).to(device)
                elif isinstance(model.classifier, torch.nn.Sequential):
                    # Last layer in sequential classifier
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = torch.nn.Linear(in_features, num_classes).to(device)
            print(f"✓ Replaced final layer for iWildCam: {num_classes} classes")
    elif dataset_name.startswith('cifar'):
        # For CIFAR datasets, create model with appropriate number of classes
        model = model_fn(num_classes=dataset_num_classes[dataset_name]).to(device)

        # Load pre-trained weights if available
        try:
            if getattr(args, 'use_underfitted', False):
                underfitted_epochs = getattr(args, 'underfitted_epochs', 5)
                weight_path = f"/home/haolan/pretrained_weights/{dataset_name}_{model_name}_{args.train_loss}_epochs{underfitted_epochs}_seed{args.random_seed}.model"
            else:
                weight_path = f"/hdd/haolan/pretrained_weights/{dataset_name}_{model_name}_{args.train_loss}.model"

            if os.path.exists(weight_path):
                checkpoint = torch.load(weight_path)
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

                if any('module' in key for key in state_dict.keys()):
                    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
                    model.load_state_dict(state_dict)
                else:
                    model.load_state_dict(state_dict)

                if getattr(args, 'use_underfitted', False):
                    print(f"✓ Loaded underfitted {dataset_name} {model_name} ({getattr(args, 'underfitted_epochs', 5)} epochs)")
                else:
                    print(f"✓ Loaded pretrained {dataset_name} {model_name}")
            else:
                print(f"✗ Weights not found at {weight_path}")
        except Exception as e:
            print(f"✗ Error loading weights: {e}")
    else:
        # For other datasets, simply create the model
        model = model_fn().to(device)

    return model
