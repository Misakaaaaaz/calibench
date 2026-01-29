import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required for EVA models. Install with: pip install timm>=0.9.0")

#TODO: need resize
class EVA02_CIFAR10_Base(nn.Module):
    def __init__(self, model_name='eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True, **kwargs):
        super(EVA02_CIFAR10_Base, self).__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            **kwargs
        )

    def forward(self, x, return_features=False):
        x = self.model.forward_features(x)
        features = x[:, 0] if x.ndim == 3 else x

        logits = self.model.forward_head(x)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.forward_head(x)


# 方便调用的子类
class EVA02_Base_CIFAR10(EVA02_CIFAR10_Base):
    def __init__(self, **kwargs):
        super().__init__(model_name='eva02_base_patch14_448.mim_in22k_ft_in1k', **kwargs)


class EVA02_Large_CIFAR10(EVA02_CIFAR10_Base):
    def __init__(self, **kwargs):
        super().__init__(model_name='eva02_large_patch14_448.mim_m38m_ft_in1k', **kwargs)


class EVA02_Small_CIFAR10(EVA02_CIFAR10_Base):
    def __init__(self, **kwargs):
        super().__init__(model_name='eva02_small_patch14_336.mim_in22k_ft_in1k', **kwargs)