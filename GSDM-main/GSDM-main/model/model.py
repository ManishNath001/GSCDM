# model/model.py

import logging
from collections import OrderedDict
import os

import torch
import torch.nn as nn

import model.networks as networks
from .base_model import BaseModel
from .networks import Conv_bn_block, Res_block

logger = logging.getLogger("base")


# ============================================================
# SPM NETWORK
# ============================================================

class SPM(torch.nn.Module):
    def __init__(self, in_channels, cum, get_feature_map=False):
        super().__init__()

        self.cnum = cum
        self.get_feature_map = get_feature_map

        self.res_block = Res_block(in_channels, self.cnum)

        self._conv1_1 = Conv_bn_block(
            in_channels=in_channels,
            out_channels=self.cnum,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2
        )

        self._conv1_2 = Conv_bn_block(
            in_channels=self.cnum,
            out_channels=self.cnum,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2
        )

        self._pool1 = torch.nn.Conv2d(
            self.cnum,
            2 * self.cnum,
            3,
            2,
            1
        )

        self._conv2_1 = Conv_bn_block(
            2 * self.cnum,
            2 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._conv2_2 = Conv_bn_block(
            2 * self.cnum,
            2 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._pool2 = torch.nn.Conv2d(
            2 * self.cnum,
            4 * self.cnum,
            3,
            2,
            1
        )

        self._conv3_1 = Conv_bn_block(
            4 * self.cnum,
            4 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._conv3_2 = Conv_bn_block(
            4 * self.cnum,
            4 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._pool3 = torch.nn.Conv2d(
            4 * self.cnum,
            8 * self.cnum,
            3,
            2,
            1
        )

        self._conv4_1 = Conv_bn_block(
            8 * self.cnum,
            8 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._conv4_2 = Conv_bn_block(
            8 * self.cnum,
            8 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._deconv1 = torch.nn.ConvTranspose2d(
            8 * self.cnum,
            4 * self.cnum,
            2,
            2
        )

        self._bn1 = torch.nn.BatchNorm2d(4 * self.cnum)

        self._conv5_1 = Conv_bn_block(
            4 * self.cnum,
            4 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._conv5_2 = Conv_bn_block(
            4 * self.cnum,
            4 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._deconv2 = torch.nn.ConvTranspose2d(
            4 * self.cnum,
            2 * self.cnum,
            2,
            2
        )

        self._bn2 = torch.nn.BatchNorm2d(2 * self.cnum)

        self._conv6_1 = Conv_bn_block(
            2 * self.cnum,
            2 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._conv6_2 = Conv_bn_block(
            2 * self.cnum,
            2 * self.cnum,
            3,
            1,
            2,
            2
        )

        self._deconv3 = torch.nn.ConvTranspose2d(
            2 * self.cnum,
            self.cnum,
            2,
            2
        )

        self._bn3 = torch.nn.BatchNorm2d(self.cnum)

        self._conv7 = torch.nn.Conv2d(
            self.cnum,
            1,
            3,
            1,
            1
        )

    def forward(self, x):

        x = self.res_block(x)

        x = self._conv1_1(x)
        x = self._conv1_2(x)

        f1 = x

        x = torch.nn.functional.elu(self._pool1(x))

        x = self._conv2_1(x)
        x = self._conv2_2(x)

        f2 = x

        x = torch.nn.functional.elu(self._pool2(x))

        x = self._conv3_1(x)
        x = self._conv3_2(x)

        f3 = x

        x = torch.nn.functional.elu(self._pool3(x))

        x = self._conv4_1(x)
        x = self._conv4_2(x)

        x = self._deconv1(x)
        x = x + f3

        x = torch.nn.functional.elu(self._bn1(x))

        x = self._conv5_1(x)
        x = self._conv5_2(x)

        x = self._deconv2(x)
        x = x + f2

        x = torch.nn.functional.elu(self._bn2(x))

        x = self._conv6_1(x)
        x = self._conv6_2(x)

        x = self._deconv3(x)
        x = x + f1

        x = torch.nn.functional.elu(self._bn3(x))

        return torch.tanh(self._conv7(x))


# ============================================================
# DDPM WRAPPER
# ============================================================

class DDPM(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.opt = opt

        self.netG = networks.define_G(opt)

        self.log_dict = OrderedDict()

        if opt["phase"] == "train":

            self.netG.train()

            self.optG = torch.optim.Adam(
                self.netG.parameters(),
                lr=float(opt["train"]["optimizer"]["lr"])
            )

        else:
            self.netG.eval()

    def feed_data(self, data):
        self.data = data

    def optimize_parameters(self):

        self.optG.zero_grad()

        loss = self.netG(self.data)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.netG.parameters(),
            1.0
        )

        self.optG.step()

        self.log_dict["l_pix"] = float(loss.item())

    def save_network(self, epoch, iter_step):

        save_dir = self.opt["path"]["log"]

        os.makedirs(save_dir, exist_ok=True)

        gen_path = os.path.join(
            save_dir,
            f"I{iter_step}_E{epoch}_gen.pth"
        )

        opt_path = os.path.join(
            save_dir,
            f"I{iter_step}_E{epoch}_opt.pth"
        )

        net = (
            self.netG.module
            if isinstance(self.netG, nn.DataParallel)
            else self.netG
        )

        torch.save(net.state_dict(), gen_path)

        torch.save(
            {
                "epoch": epoch,
                "iter": iter_step,
                "optimizer": self.optG.state_dict()
            },
            opt_path
        )

        print(f"Saved: {gen_path}")