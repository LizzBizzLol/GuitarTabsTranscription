"""Single-frame frequency-only models; output strings are ordered 6 to 1."""
from __future__ import annotations

import torch
from torch import nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, padding=1),
                         nn.BatchNorm1d(out_channels), nn.ReLU())


class _FrequencyEncoder(nn.Module):
    def __init__(self, n_mels, channels):
        super().__init__()
        if type(n_mels) is not int or n_mels < 8 or n_mels % 8:
            raise ValueError("n_mels must be a positive multiple of 8")
        if not isinstance(channels, (tuple, list)) or len(channels) != 3 or any(
                type(c) is not int or c <= 0 for c in channels):
            raise ValueError("channels must contain three positive integers")
        self.n_mels = n_mels
        self.channels = tuple(channels)
        a, b, c = channels
        self.encoder = nn.ModuleList([_block(1, a), _block(a, b), _block(b, c)])
        self.pools = nn.ModuleList([nn.MaxPool1d(2) for _ in channels])

    def _encode(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (1, self.n_mels) or x.shape[0] == 0:
            raise ValueError(f"Expected nonempty input [B, 1, {self.n_mels}]; context must be 1")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Input must contain finite floating-point values")
        skips = []
        for block, pool in zip(self.encoder, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        return x, skips

    def _head(self, features):
        return torch.stack([head(features) for head in self.heads], dim=1)


class TabCNNBaseline(nn.Module):
    """Sequential 1D Log-Mel reference, not a reproduction of original TabCNN."""

    def __init__(self, n_mels: int = 256):
        super().__init__()
        if type(n_mels) is not int or n_mels <= 0 or n_mels % 2:
            raise ValueError("n_mels must be a positive even integer")
        self.n_mels = n_mels
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32),
                          nn.ReLU(), nn.Dropout(0.25)),
            nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64),
                          nn.ReLU(), nn.Dropout(0.25)),
            nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.MaxPool1d(2),
                          nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25)),
        ])
        self.flatten = nn.Flatten(start_dim=1)
        self.heads = nn.ModuleList([nn.Linear(128 * (n_mels // 2), 26) for _ in range(6)])

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (1, self.n_mels) or x.shape[0] == 0:
            raise ValueError(f"Expected nonempty input [B, 1, {self.n_mels}]; context must be 1")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Input must contain finite floating-point values")
        for block in self.encoder:
            x = block(x)
        features = self.flatten(x)
        return torch.stack([head(features) for head in self.heads], dim=1)


class FullFrequencyUNet(_FrequencyEncoder):
    def __init__(self, n_mels: int = 256, channels=(32, 64, 128)):
        super().__init__(n_mels, channels)
        widths = list(reversed(self.channels))
        current = self.channels[-1]
        self.upsamples = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for width in widths:
            self.upsamples.append(nn.ConvTranspose1d(current, width, 2, stride=2))
            self.decoder.append(_block(2 * width, width))
            current = width
        self.flatten = nn.Flatten(start_dim=1)
        self.heads = nn.ModuleList([nn.Linear(self.channels[0] * n_mels, 26) for _ in range(6)])

    def forward(self, x):
        x, skips = self._encode(x)
        for upsample, block, skip in zip(self.upsamples, self.decoder, reversed(skips)):
            x = block(torch.cat((upsample(x), skip), dim=1))
        return self._head(self.flatten(x))


class TruncatedFrequencyUNet(_FrequencyEncoder):
    def __init__(self, n_mels: int = 256, channels=(32, 64, 128), pool_bins: int = 8):
        super().__init__(n_mels, channels)
        if type(pool_bins) is not int or pool_bins <= 0 or pool_bins > n_mels // 4:
            raise ValueError("pool_bins must be positive and no larger than the deepest skip width")
        self.pool_bins = pool_bins
        self.feature_pools = nn.ModuleList([nn.AdaptiveAvgPool1d(pool_bins) for _ in channels])
        self.flatten = nn.Flatten(start_dim=1)
        self.heads = nn.ModuleList([nn.Linear(sum(channels) * pool_bins, 26) for _ in range(6)])

    def forward(self, x):
        _, skips = self._encode(x)
        features = torch.cat([self.flatten(pool(skip)) for pool, skip in zip(self.feature_pools, skips)], dim=1)
        return self._head(features)


class AdditiveFrequencyEncoder(nn.Module):
    """Project pooled skips to a shared frequency grid and add, never concatenate."""

    def __init__(self, n_mels: int = 256):
        super().__init__()
        if type(n_mels) is not int or n_mels <= 0 or n_mels % 4:
            raise ValueError("n_mels must be a positive multiple of 4")
        self.n_mels = n_mels
        self.encoder = nn.ModuleList([_block(1, 64), _block(64, 128), _block(128, 256)])
        self.pools = nn.ModuleList([nn.MaxPool1d(2), nn.MaxPool1d(2)])
        self.proj1_to_3 = nn.Conv1d(64, 256, 1, stride=2)
        self.proj2_to_3 = nn.Conv1d(128, 256, 1, stride=1)
        self.flatten = nn.Flatten(start_dim=1)

        # Addition itself has no trainable parameters and keeps 256 channels.
        # Concatenating these same three projected maps would yield 768 channels,
        # tripling the flattened head input at the same frequency resolution.
        # At n_mels=256, six 26-class heads need 256*64*26*6 = 2,555,904
        # weights instead of 768*64*26*6 = 7,667,712 (both have 156 biases).
        # The projections are not free: they add 49,664 parameters including bias.
        # This is a comparison with matched projected-map concatenation, not with
        # the earlier U-Nets. This model totals 2,730,140 parameters at default
        # settings, MORE than either earlier model; preserving the full frequency
        # grid in the head remains costly. No averaging silently reduces it here.
        self.heads = nn.ModuleList([nn.Linear(256 * (n_mels // 4), 26) for _ in range(6)])

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (1, self.n_mels) or x.shape[0] == 0:
            raise ValueError(f"Expected nonempty input [B, 1, {self.n_mels}]; context must be 1")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Input must contain finite floating-point values")
        out_layer1_pooled = self.pools[0](self.encoder[0](x))
        out_layer2_pooled = self.pools[1](self.encoder[1](out_layer1_pooled))
        out_layer3 = self.encoder[2](out_layer2_pooled)
        fused = (out_layer3 + self.proj1_to_3(out_layer1_pooled)
                 + self.proj2_to_3(out_layer2_pooled))
        features = self.flatten(fused)
        return torch.stack([head(features) for head in self.heads], dim=1)
