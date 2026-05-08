# model/networks.py

import logging
import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger("base")


# ============================================================
# INIT
# ============================================================

def weights_init_orthogonal(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net):
    net.apply(weights_init_orthogonal)


# ============================================================
# REQUIRED BLOCKS (DO NOT REMOVE)
# ============================================================

class Conv_bn_block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(*args, **kwargs)

        if "out_channels" in kwargs:
            out_ch = kwargs["out_channels"]
        else:
            out_ch = args[1]

        self._bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return torch.nn.functional.elu(
            self._bn(self._conv(x)),
            alpha=1.0
        )


class Res_block(nn.Module):
    def __init__(self, in_channels, cnum):
        super().__init__()

        self._conv1 = nn.Conv2d(in_channels, cnum, 3, 1, 2, dilation=2)
        self._conv2 = nn.Conv2d(cnum, cnum, 3, 1, 2, dilation=2)
        self._conv3 = nn.Conv2d(cnum, in_channels, 3, 1, 2, dilation=2)

        self._bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        xin = x

        x = torch.nn.functional.elu(self._conv1(x))
        x = torch.nn.functional.elu(self._conv2(x))
        x = self._conv3(x)

        x = xin + x
        x = torch.nn.functional.elu(self._bn(x))

        return x


# ============================================================
# GENERATOR
# ============================================================

def define_G(opt):

    from .restormer import Restormer
    from .cold_diffusion import ColdDiffusion

    backbone = Restormer(
        in_channels=13,
        out_channels=3,
        dim=64
    )

    netG = ColdDiffusion(
        backbone,
        T=50
    )

    if opt["phase"] == "train":
        init_weights(netG)

    return netG