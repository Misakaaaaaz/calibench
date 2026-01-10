'''
https://github.com/MohammadRoodbari/Image-Classification
'''

import torch
import torch.nn as nn
from transformers import BeitForImageClassification


cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

id2label = {i: label for i, label in enumerate(cifar10_classes)}
label2id = {label: i for i, label in enumerate(cifar10_classes)}



class BEiT_CIFAR_Wrapper(nn.Module):
    def __init__(self, model_id, num_classes=10, temp=1.0, id2label=None, label2id=None):
        super().__init__()
        self.temp = temp


        self.model = BeitForImageClassification.from_pretrained(
            model_id,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, return_features=False):
        # 1. 获取中间特征
        outputs = self.model.beit(pixel_values)
        sequence_output = outputs.last_hidden_state


        features = sequence_output.mean(dim=1)

        logits = self.model.classifier(features)

        logits = logits / self.temp

        if return_features:
            return logits, features
        return logits


class BEiT_Base_CIFAR(BEiT_CIFAR_Wrapper):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(
            'microsoft/beit-base-patch16-224',
            num_classes=num_classes,
            id2label=id2label,
            label2id=label2id,
            **kwargs
        )


class BEiT_Large_CIFAR(BEiT_CIFAR_Wrapper):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(
            'microsoft/beit-large-patch16-224',
            num_classes=num_classes,
            id2label=id2label,
            label2id=label2id,
            **kwargs
        )


class BEiTv2_Base_CIFAR(BEiT_CIFAR_Wrapper):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(
            'microsoft/beitv2-base-patch16-224',
            num_classes=num_classes,
            id2label=id2label,
            label2id=label2id,
            **kwargs
        )