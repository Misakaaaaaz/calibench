from util import get_model_and_logits
from types import SimpleNamespace
import torch
import os
import numpy as np

def _to_tensor(x, is_label: bool = False):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.tensor(x)

    if is_label:
        return t.long()
    else:
        return t.float()

def main():
    args = SimpleNamespace(
        dataset_root='data',
        dataset='imagenet_sketch',
        model='resnet50'
    )

    for seed in range(1, 6):
        valid_size = 0.2
        val_logits, val_labels, test_logits, test_labels, val_features, test_features = get_model_and_logits(
            args,
            dataset_name='imagenet_sketch',
            model_name='resnet50',
            batch_size=128,
            valid_size=0.2,
            seed_value=seed
        )

        val_logits = _to_tensor(val_logits, is_label=False)
        val_features = _to_tensor(val_features, is_label=False)
        val_labels = _to_tensor(val_labels, is_label=True)
        test_logits = _to_tensor(test_logits, is_label=False)
        test_features = _to_tensor(test_features, is_label=False)
        test_labels = _to_tensor(test_labels, is_label=True)

        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
        save_path = os.path.join(
            OUTPUT_DIR, f"imagenet_sketch_resnet50_seed{seed}_vs{valid_size}.pt"
        )
        torch.save(
            (
                val_logits,
                val_features,
                val_labels,
                test_logits,
                test_features,
                test_labels,
            ),
            save_path,
        )

if __name__ == '__main__':
    main()

'''
from util import get_model_and_logits
from types import SimpleNamespace
import torch
import os
import numpy as np

def _to_tensor(x, is_label: bool = False):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.tensor(x)

    if is_label:
        return t.long()
    else:
        return t.float()

def main():
    args = SimpleNamespace(
        dataset_root='data',
        dataset='imagenet_sketch',
        model='resnet50',
        num_workers=8
    )

    val_logits, val_labels, test_logits, test_labels, val_features, test_features = get_model_and_logits(
        args,
        dataset_name='imagenet_sketch',
        model_name='resnet50',
        batch_size=128,
        valid_size=0.2,
        seed_value=1
    )

    val_logits = _to_tensor(val_logits, is_label=False)
    val_features = _to_tensor(val_features, is_label=False)
    val_labels = _to_tensor(val_labels, is_label=True)
    test_logits = _to_tensor(test_logits, is_label=False)
    test_features = _to_tensor(test_features, is_label=False)
    test_labels = _to_tensor(test_labels, is_label=True)

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    save_path = os.path.join(
        OUTPUT_DIR, f"imagenet_sketch_resnet50_seed.pt"
    )
    torch.save(
        (
            val_logits,
            val_features,
            val_labels,
            test_logits,
            test_features,
            test_labels,
        ),
        save_path,
    )

if __name__ == '__main__':
    main()
'''