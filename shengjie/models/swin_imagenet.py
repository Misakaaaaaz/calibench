# Swin Transformer with feature output
import torch
import torch.nn as nn
import torchvision

class Swin_B_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(Swin_B_ImageNet, self).__init__()
        self.model = torchvision.models.swin_b(**kwargs)

    def forward(self, x, return_features=False):
        # Follow the correct Swin transformer pipeline:
        # features -> norm -> permute -> avgpool -> flatten -> head
        
        # Extract raw features from backbone
        raw_features = self.model.features(x)  # Shape: [B, H, W, C]
        
        # Apply layer normalization
        normed = self.model.norm(raw_features)  # Shape: [B, H, W, C]
        
        # Permute to change from BHWC to BCHW format
        permuted = self.model.permute(normed)  # Shape: [B, C, H, W]
        
        # Apply adaptive average pooling
        pooled = self.model.avgpool(permuted)  # Shape: [B, C, 1, 1]
        
        # Flatten to get feature vector
        features = self.model.flatten(pooled)  # Shape: [B, C]
        
        # Apply classification head
        logits = self.model.head(features)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.head(x)