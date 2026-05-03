import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MaskPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ EXACT CHANNELS FROM CHECKPOINT
        self.enc1 = ConvBlock(6, 4)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(4, 8)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(8, 16)

        self.up2 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec2 = ConvBlock(16, 8)

        self.up1 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.dec1 = ConvBlock(8, 4)

        self.out_conv = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Decoder
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out