import random
from types import SimpleNamespace

import numpy as np
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
from typing import Dict, Optional, Union
import os
import torch

from bench.utils.Datasets.imagenet_lt import get_imagenet_lt_data_loader
from bench.utils.cali.BBQ import BBQ
from bench.utils.cali.CTS import CTS
from bench.utils.cali.ETS import ETS
from bench.utils.cali.FC import FC
from bench.utils.cali.GC import GC
from bench.utils.cali.HB import HB
from bench.utils.cali.PTS import PTS
from bench.utils.cali.ProCal_DR import ProCal_DR
from bench.utils.cali.SMART import SMART_
from bench.utils.cali.TS import TS
from bench.utils.cali.Uncalibrated import Uncalibrated
from bench.utils.cali.VS import VS
from bench.utils.data_utils import *
from bench.utils.model_utils import *
from bench.utils.util import *


def generate_cali_result(args,dataset_name, data_loc,model_name, seed, valid_size,pool_valid_size, device,run_methods=["uncalibrated", "TS", "PTS", "CTS", "ETS", "SMART", "HB", "BBQ", "VS", "GC", "ProCal_DR", "FC", "Spline"]):
    print(f"Using device: {device}")
    validate_dataset_and_model(dataset_name,model_name)
    set_seed(seed)
    corruption_type = getattr(args, 'corruption_type', None) if dataset_name == 'imagenet_c' else None
    severity = getattr(args, 'severity', None) if dataset_name == 'imagenet_c' else None
    train_loss = args.train_loss if dataset_name.startswith('cifar') else None

    if logits_exist(dataset_name, model_name, seed, pool_valid_size, 'CE', corruption_type, severity, train_loss):
        print("Loading cached logits/features...")
        val_logits, val_labels, test_logits, test_labels, val_features, test_features = load_logits(dataset_name,
                                                                                                    model_name, seed,
                                                                                                    pool_valid_size, 'CE',
                                                                                                    corruption_type,
                                                                                                    severity,
                                                                                                    train_loss)
    else:
        print("Cache not found → computing logits/features...")
        model = create_model(args=args,model_name=model_name, dataset_name=dataset_name, device=device)
        model.eval()
        norm_mean, norm_std = get_model_normalization(model_name)
        input_size = get_model_input_size(model_name)
        val_loader, test_loader = get_imagenet_lt_data_loader(root=data_loc,batch_size=128, pin_memory=True,
                                                              random_seed=seed, mean=norm_mean, std=norm_std,
                                                              image_size=input_size, valid_size=pool_valid_size)
        # Compute logits and features
        val_logits = []
        val_labels = []
        val_features = []
        test_logits = []
        test_labels = []
        test_features = []
        print("Computing logits and features for calibration set...")
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                # Get features from model (all models should support return_features=True now)
                outputs, features = model(inputs, return_features=True)
                val_features.append(features.cpu().numpy())
                val_logits.append(outputs.cpu().numpy())
                val_labels.append(labels.numpy())
        print("Computing logits and features for test set...")
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs = inputs.to(device)
                # Get features from model (all models should support return_features=True now)
                outputs, features = model(inputs, return_features=True)
                test_features.append(features.cpu().numpy())
                test_logits.append(outputs.cpu().numpy())
                test_labels.append(labels.numpy())
        val_logits = np.vstack(val_logits)
        val_labels = np.hstack(val_labels)
        val_features = np.vstack(val_features)
        test_logits = np.vstack(test_logits)
        test_labels = np.hstack(test_labels)
        test_features = np.vstack(test_features)
        save_logits(val_logits, val_labels, test_logits, test_labels, val_features, test_features,
                    dataset_name, model_name, seed, pool_valid_size, 'CE', corruption_type, severity, train_loss)

    val_features,val_logits,val_labels=bootstrap_val_by_global_valid_size(val_features, val_logits,val_labels,valid_size,pool_valid_size)

    val_logits_tensor = torch.tensor(val_logits, dtype=torch.float32).to(device)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).to(device)
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
    test_logits_tensor = torch.tensor(test_logits, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)


    result_dir = os.path.join("results", f"{dataset_name}_{model_name}_seed{seed}_vs{valid_size}")
    os.makedirs(result_dir, exist_ok=True)
    overall_results_file = os.path.join(result_dir, f"calibration_results.json")
    if os.path.exists(overall_results_file):
        # Load existing results if available
        with open(overall_results_file, "r") as f:
            overall_results = json.load(f)
        print(f"Loaded existing results from {overall_results_file}")
    else:
        # Create new results dictionary if not available
        overall_results = {
            'dataset': dataset_name,
            'model': model_name,
            'seed': seed,
            'valid_size': valid_size,
            'overall': {}
        }

    if "uncalibrated" in run_methods:
        Uncalibrated(test_logits_tensor,test_labels_tensor,overall_results,device)

    if "TS" in run_methods:
        TS(val_logits_tensor, val_labels_tensor, test_logits_tensor, test_labels_tensor, overall_results, device)

    if "PTS" in run_methods:
        PTS(val_logits_tensor, val_labels_tensor, test_logits_tensor, test_labels_tensor, seed, overall_results, device)

    if "CTS" in run_methods:
        CTS(val_labels_tensor, val_logits_tensor, test_logits_tensor, test_labels_tensor, overall_results, device)

    if "ETS" in run_methods:
        ETS(val_logits, val_labels, test_logits, test_labels_tensor, overall_results,device)

    if "SMART" in run_methods:
        smart_loss = 'smooth_soft_ece'
        SMART_(smart_loss, dataset_name, model_name, seed, valid_size, corruption_type, severity, train_loss,
               val_logits, val_labels, test_logits, test_labels_tensor,device,  overall_results)

    if "HB" in run_methods:
        HB(val_logits_tensor, val_labels_tensor, test_logits_tensor, test_labels_tensor,device, overall_results)

    if "BBQ" in run_methods:
        BBQ(val_logits_tensor, val_labels_tensor, test_logits_tensor, test_labels_tensor, overall_results,device)

    if "VS" in run_methods:
        VS(val_logits_tensor, val_labels_tensor, test_logits_tensor, test_labels_tensor,device, overall_results)

    if "GC" in run_methods:
        GC(val_logits_tensor, val_labels_tensor, test_logits_tensor, test_labels_tensor,device, overall_results)

    if "ProCal_DR" in run_methods:
        ProCal_DR(val_features_tensor, test_features_tensor, val_logits_tensor, test_logits_tensor, val_labels_tensor,
                  test_labels_tensor,device, overall_results)

    if "FC" in run_methods:
        FC(args, model_name, dataset_name, device, val_features_tensor, val_logits_tensor, val_labels_tensor,
           test_features_tensor, test_labels_tensor, overall_results)

    with open(overall_results_file, "w") as f:
        json.dump(overall_results, f, indent=4)

    print(f"Saved updated results to {overall_results_file}")

if __name__ == "__main__":
    #Just show how to use it
    print("You can see how to use it in test_bench.py")









