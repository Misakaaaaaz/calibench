# BEIT (Bidirectional Encoder representation from Image Transformers) with feature output
import torch
import torch.nn as nn
try:
    import timm
except ImportError:
    raise ImportError("timm is required for BEIT models. Install with: pip install timm")


class BEiT_Base_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(BEiT_Base_ImageNet, self).__init__()
        self.model = timm.create_model('beit_base_patch16_224', **kwargs)

    def forward(self, x, return_features=False):
        tokens = self.model.forward_features(x)
        features = self.model.forward_head(tokens, pre_logits=True)

        logits = self.model.forward_head(tokens)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):

        return self.model.head(x)


class BEiT_Large_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(BEiT_Large_ImageNet, self).__init__()
        self.model = timm.create_model('beit_large_patch16_224', **kwargs)

    def forward(self, x, return_features=False):
        tokens = self.model.forward_features(x)
        features = self.model.forward_head(tokens, pre_logits=True)

        logits = self.model.forward_head(tokens)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.head(x)


class BEiTv2_Base_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(BEiTv2_Base_ImageNet, self).__init__()
        self.model = timm.create_model('beitv2_base_patch16_224', **kwargs)

    def forward(self, x, return_features=False):
        tokens = self.model.forward_features(x)
        features = self.model.forward_head(tokens, pre_logits=True)

        logits = self.model.forward_head(tokens)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.head(x)
