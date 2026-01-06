# resnet with feature output
import torch
import torch.nn as nn
import torchvision

class ResNet_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet_ImageNet, self).__init__()
        self.model = torchvision.models.resnet50(**kwargs)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.fc(x)

class ResNet101_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet101_ImageNet, self).__init__()
        self.model = torchvision.models.resnet101(**kwargs)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.fc(x)

class ResNet152_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet152_ImageNet, self).__init__()
        self.model = torchvision.models.resnet152(**kwargs)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def classifier(self, x):
        return self.model.fc(x)
