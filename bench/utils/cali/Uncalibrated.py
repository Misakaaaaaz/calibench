
import torch
import os

from bench.utils.util import get_all_metrics, store_method_results


def Uncalibrated(test_logits_tensor,test_labels_tensor,overall_results,device):
    all_metrics = get_all_metrics(device,logits=test_logits_tensor, labels=test_labels_tensor, n_bins=15)
    # Store results
    store_method_results(
        overall_results=overall_results,
        method_key='uncalibrated',
        all_metrics=all_metrics,
        bins_list=[15],
        loss_fn='none'
    )

if __name__ == "__main__":
    pass