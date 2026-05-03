# model/networks.py

import functools
import logging

import torch
import torch.nn as nn

from torch.nn import init

logger = logging.getLogger("base")


# ============================================================
# INIT FUNCTIONS
# ============================================================

def weights_init_normal(m, std=0.02):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        init.normal_(m.weight.data, 0.0, std)

        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find("Linear") != -1:

        init.normal_(m.weight.data, 0.0, std)

        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find("BatchNorm2d") != -1:

        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):

    classname = m.__class__.__name__

    if classname.find("Conv2d") != -1:

        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

        m.weight.data *= scale

        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find("Linear") != -1:

        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

        m.weight.data *= scale

        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find("BatchNorm2d") != -1:

        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


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


def init_weights(net, init_type="kaiming", scale=1, std=0.02):

    logger.info(f"Initialization method [{init_type}]")

    if init_type == "normal":

        net.apply(
            functools.partial(
                weights_init_normal,
                std=std
            )
        )

    elif init_type == "kaiming":

        net.apply(
            functools.partial(
                weights_init_kaiming,
                scale=scale
            )
        )

    elif init_type == "orthogonal":

        net.apply(weights_init_orthogonal)

    else:

        raise NotImplementedError(
            f"{init_type} not implemented"
        )


# ============================================================
# CONV BLOCK
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


# ============================================================
# RES BLOCK
# ============================================================

class Res_block(nn.Module):

    def __init__(self, in_channels, cnum):
        super().__init__()

        self._conv1 = nn.Conv2d(
            in_channels,
            cnum,
            3,
            1,
            2,
            dilation=2
        )

        self._conv2 = nn.Conv2d(
            cnum,
            cnum,
            3,
            1,
            2,
            dilation=2
        )

        self._conv3 = nn.Conv2d(
            cnum,
            in_channels,
            3,
            1,
            2,
            dilation=2
        )

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
# DEFINE GENERATOR
# ============================================================

def define_G(opt):

    model_opt = opt["model"]

    which = model_opt["which_model_G"].lower()

    image_size = model_opt["diffusion"]["image_size"]

    if not isinstance(image_size, (tuple, list)):
        image_size = [image_size, image_size]

    unet_cfg = model_opt["unet"]

    if (
        "norm_groups" not in unet_cfg
        or unet_cfg["norm_groups"] is None
    ):
        unet_cfg["norm_groups"] = 32

    # ============================================================
    # COLD DIFFUSION
    # ============================================================

    if which == "cold":

        from .sr3_modules import unet as sr3_unet
        from .cold_diffusion import ColdDiffusion

        backbone = sr3_unet.UNet(

            in_channel=unet_cfg["in_channel"],

            out_channel=unet_cfg["out_channel"],

            norm_groups=unet_cfg["norm_groups"],

            inner_channel=unet_cfg["inner_channel"],

            channel_mults=unet_cfg["channel_multiplier"],

            attn_res=unet_cfg["attn_res"],

            res_blocks=unet_cfg["res_blocks"],

            dropout=unet_cfg["dropout"],

            image_size=image_size,
        )

        T = int(
            model_opt["beta_schedule"]["train"].get(
                "n_timestep",
                200
            )
        )

        netG = ColdDiffusion(
            backbone,
            T=T
        )

    else:

        raise ValueError(
            f"Unsupported model type: {which}"
        )

    # ============================================================
    # INIT
    # ============================================================

    if opt["phase"] == "train":

        init_weights(
            netG,
            init_type="orthogonal"
        )

    # ============================================================
    # GPU
    # ============================================================

    if (
        opt.get("gpu_ids")
        and opt.get("distributed", False)
    ):

        assert torch.cuda.is_available()

        netG = nn.DataParallel(netG)

    return netG