# Vision Transformer (ViT) with feature output
import torch
import torch.nn as nn
import torchvision

class ViT_B_16_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ViT_B_16_ImageNet, self).__init__()
        self.model = torchvision.models.vit_b_16(**kwargs)

    def forward(self, x, return_features=False):
        # ViT forward pass without the final classification head
        # Based on torchvision ViT implementation
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Extract [CLS] token representation as features
        features = x[:, 0]  # [CLS] token

        # Apply classification head
        logits = self.model.heads(features)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.heads(x)

class ViT_B_32_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ViT_B_32_ImageNet, self).__init__()
        self.model = torchvision.models.vit_b_32(**kwargs)

    def forward(self, x, return_features=False):
        x = self.model._process_input(x)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)
        features = x[:, 0]  # [CLS] token
        logits = self.model.heads(features)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.heads(x)

class ViT_L_16_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ViT_L_16_ImageNet, self).__init__()
        self.model = torchvision.models.vit_l_16(**kwargs)

    def forward(self, x, return_features=False):
        x = self.model._process_input(x)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)
        features = x[:, 0]  # [CLS] token
        logits = self.model.heads(features)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.heads(x)

class ViT_L_32_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ViT_L_32_ImageNet, self).__init__()
        self.model = torchvision.models.vit_l_32(**kwargs)

    def forward(self, x, return_features=False):
        x = self.model._process_input(x)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)
        features = x[:, 0]  # [CLS] token
        logits = self.model.heads(features)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.heads(x)