"""Vendored U-Net definition from ../Urban3D-DeepRecon/src/unet.py.

Not imported by default; heavy deps (torch) may be required.
"""

from __future__ import annotations

import importlib


def _torch_modules():
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        F = importlib.import_module("torch.nn.functional")
        return torch, nn, F
    except Exception as e:  # pragma: no cover
        raise ImportError("torch is required for the vendored Urban3D U-Net") from e


class DoubleConv(object):
    def __init__(self, in_channels, out_channels):
        _torch, nn, _F = _torch_modules()
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(object):
    def __init__(self, in_channels=1, out_channels=1, features=None):
        torch, nn, F = _torch_modules()
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self._torch = torch
        self._nn = nn
        self._F = F
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = self._F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = self._torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
