# model/restormer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# WINDOW ATTENTION (MEMORY SAFE)
# ============================================================

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        # padding
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))

        B, C, Hp, Wp = x.shape

        # split windows
        x = x.reshape(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(-1, ws * ws, C)

        # qkv
        x_reshaped = x.permute(0, 2, 1).reshape(-1, C, ws, ws)
        qkv = self.to_qkv(x_reshaped)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)

        attn = (q @ k) / (C ** 0.5)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(-1, C, ws, ws)

        out = self.proj(out)

        # merge windows
        out = out.reshape(B, Hp // ws, Wp // ws, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.reshape(B, C, Hp, Wp)

        return out[:, :, :H, :W]


# ============================================================
# BLOCK
# ============================================================

class RestormerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # attention
        x_ = x.permute(0, 2, 3, 1)
        x_ = self.norm1(x_)
        x_ = x_.permute(0, 3, 1, 2)

        x = x + self.attn(x_)

        # feedforward
        x_ = x.permute(0, 2, 3, 1)
        x_ = self.norm2(x_)
        x_ = x_.permute(0, 3, 1, 2)

        x = x + self.ffn(x_)

        return x


# ============================================================
# MAIN MODEL
# ============================================================

class Restormer(nn.Module):
    def __init__(self, in_channels=13, out_channels=3, dim=32):
        super().__init__()

        self.embed = nn.Conv2d(in_channels, dim, 3, padding=1)

        self.block1 = RestormerBlock(dim)
        self.block2 = RestormerBlock(dim)

        self.output = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x, t=None):
        x = self.embed(x)

        x = self.block1(x)
        x = self.block2(x)

        return self.output(x)