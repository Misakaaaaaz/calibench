import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

# 基础组件
pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class MLPMixer_CIFAR(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, num_classes,
                 expansion_factor=4, expansion_factor_token=0.5, dropout=0., temp=1.0):
        super(MLPMixer_CIFAR, self).__init__()
        self.temp = temp

        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear


        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2) * channels, dim),
        )


        self.mixer_blocks = nn.ModuleList([
            nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(depth)
        ])


        self.norm = nn.LayerNorm(dim)
        self.reduce = Reduce('b n c -> b c', 'mean')
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x, return_features=False):
        # 输入形状: [B, 3, 32, 32]
        out = self.patch_embed(x)

        for block in self.mixer_blocks:
            out = block(out)

        out = self.norm(out)

        features = self.reduce(out)

        logits = self.fc(features) / self.temp

        if return_features:
            return logits, features
        return logits


# 具体的 MLP-Mixer-B-16 配置类
class MLPMixer_B16_CIFAR(MLPMixer_CIFAR):
    def __init__(self, num_classes=10, **kwargs):
        super(MLPMixer_B16_CIFAR, self).__init__(
            image_size=32,
            channels=3,
            patch_size=4,
            dim=512,
            depth=6,
            num_classes=num_classes,
            **kwargs
        )