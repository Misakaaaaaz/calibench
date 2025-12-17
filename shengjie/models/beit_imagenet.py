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
        # Use timm's built-in forward_features for correct feature extraction
        x = self.model.forward_features(x)

        # Extract [CLS] token features before fc_norm
        features = x[:, 0]

        # Use forward_head for proper classification (includes fc_norm, head_drop, and head)
        logits = self.model.forward_head(x)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        # x should be the full token sequence from forward_features
        return self.model.forward_head(x)


class BEiT_Large_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(BEiT_Large_ImageNet, self).__init__()
        self.model = timm.create_model('beit_large_patch16_224', **kwargs)

    def forward(self, x, return_features=False):
        # Use timm's built-in forward_features for correct feature extraction
        x = self.model.forward_features(x)

        # Extract [CLS] token features before fc_norm
        features = x[:, 0]

        # Use forward_head for proper classification (includes fc_norm, head_drop, and head)
        logits = self.model.forward_head(x)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        # x should be the full token sequence from forward_features
        return self.model.forward_head(x)


class BEiTv2_Base_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(BEiTv2_Base_ImageNet, self).__init__()
        self.model = timm.create_model('beitv2_base_patch16_224', **kwargs)

    def forward(self, x, return_features=False):
        # Use timm's built-in forward_features for correct feature extraction
        x = self.model.forward_features(x)

        # Extract [CLS] token features before fc_norm
        features = x[:, 0]

        # Use forward_head for proper classification (includes fc_norm, head_drop, and head)
        logits = self.model.forward_head(x)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        # x should be the full token sequence from forward_features
        return self.model.forward_head(x)
