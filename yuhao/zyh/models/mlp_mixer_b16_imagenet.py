import torch
import torch.nn as nn
import timm


class MLPMixer_B16_ImageNet(nn.Module):
    """
    MLP-Mixer-B/16 with ImageNet-1K pretrained head.
    Standard resolution is 224x224.
    """

    def __init__(self, **kwargs):
        super(MLPMixer_B16_ImageNet, self).__init__()
        try:

            if 'pretrained' not in kwargs:
                kwargs['pretrained'] = True

            self.model = timm.create_model(
                'mixer_b16_224.goog_in21k_ft_in1k',
                **kwargs
            )

        except Exception as e:
            raise RuntimeError(f"Failed to create MLP-Mixer model. Error: {str(e)}")

    def forward(self, x, return_features=False):

        tokens = self.model.forward_features(x)

        features = self.model.forward_head(tokens, pre_logits=True)

        logits = self.model.forward_head(tokens)

        if return_features:
            return logits, features
        return logits

    def classifier(self, x):

        return self.model.head(x)