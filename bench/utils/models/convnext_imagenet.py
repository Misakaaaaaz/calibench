# ConvNext with feature output
import torch
import torch.nn as nn
try:
    import timm
except ImportError:
    raise ImportError("timm is required for ConvNext models. Install with: pip install timm")


class ConvNext_Tiny_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ConvNext_Tiny_ImageNet, self).__init__()
        self.model = timm.create_model('convnext_tiny', **kwargs)

    def forward(self, x, return_features=False):
        # ConvNext architecture: stem -> stages -> norm -> head
        # Extract features before the classification head

        # Pass through stem and stages
        x = self.model.stem(x)
        x = self.model.stages(x)

        # Apply global pooling and norm (head processing)
        # ConvNext uses a sequential head that includes norm, global pooling, etc.
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'global_pool'):
            # Global pooling
            x = self.model.head.global_pool(x)

            # Normalization if it exists
            if hasattr(self.model.head, 'norm'):
                x = self.model.head.norm(x)

            # Flatten
            if hasattr(self.model.head, 'flatten'):
                features = self.model.head.flatten(x)
            else:
                features = x.flatten(1)

            # Apply dropout if exists
            if hasattr(self.model.head, 'drop'):
                features = self.model.head.drop(features)

            # Apply final FC layer
            if hasattr(self.model.head, 'fc'):
                logits = self.model.head.fc(features)
            else:
                logits = features
        else:
            # Fallback: use forward_features and forward_head
            features = self.model.forward_features(x)
            if hasattr(features, 'shape') and len(features.shape) > 2:
                features = features.mean(dim=[-2, -1])  # Global average pooling
            logits = self.model.head(features) if hasattr(self.model, 'head') else features

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        # Apply the final classifier layer
        if hasattr(self.model.head, 'fc'):
            return self.model.head.fc(x)
        elif hasattr(self.model, 'head'):
            return self.model.head(x)
        else:
            return x


class ConvNext_Base_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ConvNext_Base_ImageNet, self).__init__()
        self.model = timm.create_model('convnext_base', **kwargs)

    def forward(self, x, return_features=False):
        # Pass through stem and stages
        x = self.model.stem(x)
        x = self.model.stages(x)

        # Apply global pooling and norm
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'global_pool'):
            # Global pooling
            x = self.model.head.global_pool(x)

            # Normalization if it exists
            if hasattr(self.model.head, 'norm'):
                x = self.model.head.norm(x)

            # Flatten
            if hasattr(self.model.head, 'flatten'):
                features = self.model.head.flatten(x)
            else:
                features = x.flatten(1)

            # Apply dropout if exists
            if hasattr(self.model.head, 'drop'):
                features = self.model.head.drop(features)

            # Apply final FC layer
            if hasattr(self.model.head, 'fc'):
                logits = self.model.head.fc(features)
            else:
                logits = features
        else:
            # Fallback: use forward_features and forward_head
            features = self.model.forward_features(x)
            if hasattr(features, 'shape') and len(features.shape) > 2:
                features = features.mean(dim=[-2, -1])  # Global average pooling
            logits = self.model.head(features) if hasattr(self.model, 'head') else features

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        # Apply the final classifier layer
        if hasattr(self.model.head, 'fc'):
            return self.model.head.fc(x)
        elif hasattr(self.model, 'head'):
            return self.model.head(x)
        else:
            return x


class ConvNext_Large_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ConvNext_Large_ImageNet, self).__init__()
        self.model = timm.create_model('convnext_large', **kwargs)

    def forward(self, x, return_features=False):
        # Pass through stem and stages
        x = self.model.stem(x)
        x = self.model.stages(x)

        # Apply global pooling and norm
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'global_pool'):
            # Global pooling
            x = self.model.head.global_pool(x)

            # Normalization if it exists
            if hasattr(self.model.head, 'norm'):
                x = self.model.head.norm(x)

            # Flatten
            if hasattr(self.model.head, 'flatten'):
                features = self.model.head.flatten(x)
            else:
                features = x.flatten(1)

            # Apply dropout if exists
            if hasattr(self.model.head, 'drop'):
                features = self.model.head.drop(features)

            # Apply final FC layer
            if hasattr(self.model.head, 'fc'):
                logits = self.model.head.fc(features)
            else:
                logits = features
        else:
            # Fallback: use forward_features and forward_head
            features = self.model.forward_features(x)
            if hasattr(features, 'shape') and len(features.shape) > 2:
                features = features.mean(dim=[-2, -1])  # Global average pooling
            logits = self.model.head(features) if hasattr(self.model, 'head') else features

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        # Apply the final classifier layer
        if hasattr(self.model.head, 'fc'):
            return self.model.head.fc(x)
        elif hasattr(self.model, 'head'):
            return self.model.head(x)
        else:
            return x
