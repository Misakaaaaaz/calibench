# EVA (Enhanced Vision Transformer Architecture) for ImageNet classification
import torch
import torch.nn as nn
try:
    import timm
except ImportError:
    raise ImportError("timm is required for EVA models. Install with: pip install timm>=0.9.0")


class EVA02_Base_ImageNet(nn.Module):
    """
    EVA02-Base with ImageNet-1K pretrained classification head.

    IMPORTANT: Uses eva02_base_patch14_448.mim_in22k_ft_in1k which:
    - Has a pretrained 1000-class classification head (88.23% top-1 accuracy)
    - Requires 448x448 input images
    - Uses CLIP-style normalization: mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]

    The previous model (eva02_base_patch14_224.mim_in22k) was only a feature extractor
    with no classification head (num_classes=0).
    """
    def __init__(self, **kwargs):
        super(EVA02_Base_ImageNet, self).__init__()
        # Use the 448 version with ImageNet-1K fine-tuning
        # Note: EVA requires timm version >= 0.9.0
        try:
            # Force pretrained=True to get ImageNet-1K fine-tuned weights
            if 'pretrained' not in kwargs:
                kwargs['pretrained'] = True

            self.model = timm.create_model(
                'eva02_base_patch14_448.mim_in22k_ft_in1k',
                **kwargs
            )

            # Verify we got a classification head
            if self.model.num_classes != 1000:
                raise RuntimeError(
                    f"Expected 1000 classes but got {self.model.num_classes}. "
                    "Make sure you're using the correct model variant with .ft_in1k suffix."
                )

        except Exception as e:
            raise RuntimeError(
                f"Failed to create EVA-02 model. Please ensure timm>=0.9.0 is installed. "
                f"Error: {str(e)}"
            )

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, 3, 448, 448]
            return_features: If True, return both logits and features

        Returns:
            logits: Classification logits [batch_size, 1000]
            features: (optional) CLS token features [batch_size, hidden_dim]
        """
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
        """
        Apply classifier head to features.

        Args:
            x: Token sequence from forward_features [batch_size, num_tokens, hidden_dim]

        Returns:
            logits: Classification logits [batch_size, 1000]
        """
        return self.model.forward_head(x)


class EVA02_Large_ImageNet(nn.Module):
    """
    EVA02-Large with ImageNet-1K pretrained classification head.

    Uses eva02_large_patch14_448.mim_m38m_ft_in1k which:
    - Pretrained on Merged-38M dataset (IN-22K, CC12M, CC3M, COCO, ADE20K, Object365, OpenImages)
    - Fine-tuned on ImageNet-1K
    - Requires 448x448 input images
    - Uses CLIP-style normalization
    """
    def __init__(self, **kwargs):
        super(EVA02_Large_ImageNet, self).__init__()
        try:
            if 'pretrained' not in kwargs:
                kwargs['pretrained'] = True

            self.model = timm.create_model(
                'eva02_large_patch14_448.mim_m38m_ft_in1k',
                **kwargs
            )

            if self.model.num_classes != 1000:
                raise RuntimeError(
                    f"Expected 1000 classes but got {self.model.num_classes}."
                )

        except Exception as e:
            raise RuntimeError(
                f"Failed to create EVA-02 Large model. Error: {str(e)}"
            )

    def forward(self, x, return_features=False):
        x = self.model.forward_features(x)
        features = x[:, 0]
        logits = self.model.forward_head(x)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.forward_head(x)


class EVA02_Small_ImageNet(nn.Module):
    """
    EVA02-Small with ImageNet-1K pretrained classification head.

    Uses eva02_small_patch14_336.mim_in22k_ft_in1k which:
    - Requires 336x336 input images (not 224!)
    - Uses CLIP-style normalization
    """
    def __init__(self, **kwargs):
        super(EVA02_Small_ImageNet, self).__init__()
        try:
            if 'pretrained' not in kwargs:
                kwargs['pretrained'] = True

            self.model = timm.create_model(
                'eva02_small_patch14_336.mim_in22k_ft_in1k',
                **kwargs
            )

            if self.model.num_classes != 1000:
                raise RuntimeError(
                    f"Expected 1000 classes but got {self.model.num_classes}."
                )

        except Exception as e:
            raise RuntimeError(
                f"Failed to create EVA-02 Small model. Error: {str(e)}"
            )

    def forward(self, x, return_features=False):
        x = self.model.forward_features(x)
        features = x[:, 0]
        logits = self.model.forward_head(x)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.forward_head(x)
