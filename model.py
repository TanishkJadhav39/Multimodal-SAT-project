import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SiameseUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        # SHARED ENCODER
        # -------------------------------------------------

        self.enc1 = DoubleConv(1, 32)

        self.enc2 = DoubleConv(32, 64)

        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # -------------------------------------------------
        # FEATURE FUSION
        # -------------------------------------------------

        # 128 channels from before
        # + 128 channels from after
        # = 256 channels
        self.fusion = DoubleConv(256, 256)

        # -------------------------------------------------
        # DECODER
        # -------------------------------------------------

        # 64x64 -> 128x128
        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        # 128 from upsampling
        # + 64 from encoder features
        # = 192
        self.dec2 = DoubleConv(192, 128)

        # 128x128 -> 256x256
        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        # 64 from upsampling
        # + 32 from encoder features
        # = 96
        self.dec1 = DoubleConv(96, 64)

        # Final segmentation layer
        self.output = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def encode(self, x):

        # 256 x 256
        x1 = self.enc1(x)

        # 128 x 128
        x2 = self.enc2(
            self.pool(x1)
        )

        # 64 x 64
        x3 = self.enc3(
            self.pool(x2)
        )

        return x1, x2, x3

    def forward(self, before, after):

        # -------------------------------------------------
        # ENCODE BEFORE-FLOOD SAR
        # -------------------------------------------------

        a1, a2, a3 = self.encode(before)

        # -------------------------------------------------
        # ENCODE AFTER-FLOOD SAR
        # -------------------------------------------------

        b1, b2, b3 = self.encode(after)

        # -------------------------------------------------
        # FUSE DEEP FEATURES
        # -------------------------------------------------

        x = torch.cat(
            [a3, b3],
            dim=1
        )

        x = self.fusion(x)

        # -------------------------------------------------
        # DECODER LEVEL 1
        # -------------------------------------------------

        x = self.up2(x)

        # Combine with corresponding encoder features
        skip2 = a2 + b2

        x = torch.cat(
            [x, skip2],
            dim=1
        )

        x = self.dec2(x)

        # -------------------------------------------------
        # DECODER LEVEL 2
        # -------------------------------------------------

        x = self.up1(x)

        # Combine with high-resolution encoder features
        skip1 = a1 + b1

        x = torch.cat(
            [x, skip1],
            dim=1
        )

        x = self.dec1(x)

        # -------------------------------------------------
        # FLOOD PREDICTION
        # -------------------------------------------------

        output = self.output(x)

        return output