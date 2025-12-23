import json

from bench.utils.cali.smart_calibrator import SMART
from bench.utils.data_utils import get_logit_paths
from bench.utils.util import get_all_metrics, store_method_results

from types import SimpleNamespace
import torch
import os

def SMART_(smart_loss,dataset_name,model_name,seed,valid_size,corruption_type,severity,train_loss,val_logits,val_labels,test_logits,test_labels_tensor,device,overall_results):
    print("\nTraining Sample logitsgap Aware Temperature Scaling...")
    # Use SMART loss function from argparse

    print("Training SMART with loss function:", smart_loss)
    smart = SMART(epochs=2000, dataset_name=dataset_name,
                  model_name=model_name, seed_value=seed,
                  valid_size=valid_size, loss_fn=smart_loss,
                  patience=200, min_delta=0.0001,
                  corruption_type=corruption_type, severity=severity,
                  train_loss=train_loss)

    # Try to load existing SMART model
    if not smart.load_model():
        # Train new model if loading failed
        smart.fit(val_logits, val_labels)

    smart_probs = smart.calibrate(test_logits)
    smart_probs_tensor = torch.tensor(smart_probs, dtype=torch.float32)

    # Compute and print metrics
    all_metrics = get_all_metrics(
        device,
        logits=smart_probs_tensor,  # SMART returns probabilities
        labels=test_labels_tensor,
    )

    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key=f'SMART_{smart_loss}',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn=smart_loss
    )

if __name__ == '__main__':
   pass

# ECE:  0.0097737745467905
# Accuracy:  0.23980644345283508
# AdaptiveECE:  0.009374484419822693
# ClasswiseECE:  0.0007760548032820225
# NLL:  4.506536483764648
# ECEDebiased:  0.015255164200498116
# ECESweep:  0.009325494212476327
# BrierLoss:  0.43617013096809387
# RBS:  0.933991551399231