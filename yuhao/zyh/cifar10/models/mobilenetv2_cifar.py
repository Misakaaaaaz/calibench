import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class MobileNet_V2_CIFAR10(nn.Module):
    def __init__(self, num_classes=10, pretrained=False, **kwargs):
        super(MobileNet_V2_CIFAR10, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=pretrained, **kwargs)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.feature_extractor = self.model.features
        in_features = self.model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
        self.model.classifier = self.classifier

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)  # [B, 1280]
        logits = self.classifier(features)  # [B, 10]

        if return_features:
            return logits, features
        return logits


'''
https://github.com/weiaicunzai/pytorch-cifar100
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        return residual

class MobileNetV2(nn.Module):
    def __init__(self, class_num=100):
        super().__init__()
        # 注意：这里的 Conv2d 建议根据 CIFAR 特点将 kernel 改为 3
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), # 改为 kernel=3 对 32x32 效果更好
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        # 最终分类层
        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x, return_features=False):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)


        x = F.adaptive_avg_pool2d(x, 1)

        features = x.view(x.size(0), -1) 

        x = self.conv2(x)
        logits = x.view(x.size(0), -1)

        if return_features:
            return logits, features
        return logits

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        return nn.Sequential(*layers)

def mobilenetv2(class_num=10):
    return MobileNetV2(class_num=class_num)
'''