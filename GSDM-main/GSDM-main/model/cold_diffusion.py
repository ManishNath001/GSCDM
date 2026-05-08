import torch
import torch.nn as nn
import torch.nn.functional as F


class ColdDiffusion(nn.Module):
    def __init__(self, net, T=50):
        super().__init__()

        self.unet = net
        self.T = T

        self.l1 = nn.L1Loss()

    def _alpha(self, k):
        return (k.float() / self.T).view(-1, 1, 1, 1)

    def sobel(self, x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    def downsample(self, x):
        return F.interpolate(
            x,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False
        )

    def forward(self, batch):

        corrupted = batch["corrupted"]
        gt = batch["gt"]
        structure = batch["structure"]
        mask = batch["mask"]

        # =====================================================
        # STRICT BINARY MASK
        # =====================================================
        mask = (mask > 0.5).float()

        B = corrupted.size(0)
        device = corrupted.device

        # =====================================================
        # RANDOM TIMESTEP
        # =====================================================
        k = torch.randint(
            1,
            self.T + 1,
            (B,),
            device=device
        )

        alpha = self._alpha(k)

        # =====================================================
        # DETERMINISTIC DIFFUSION
        # =====================================================
        x_t = corrupted + mask * (
            (gt - corrupted) * alpha
        )

        # =====================================================
        # MODEL INPUT
        # =====================================================
        inp = torch.cat([
            corrupted,
            structure,
            mask,
            x_t
        ], dim=1)

        time = k.float() / self.T

        # =====================================================
        # RESIDUAL PREDICTION
        # =====================================================
        residual = self.unet(inp, time)

        residual = torch.nan_to_num(
            residual
        ).clamp(-1, 1)

        pred = corrupted + residual

        # =====================================================
        # ANCHORED RECONSTRUCTION
        # =====================================================
        restored = (
            corrupted * (1 - mask)
            + pred * mask
        )

        # =====================================================
        # LOSSES
        # =====================================================

        # masked loss
        loss_masked = self.l1(
            restored * mask,
            gt * mask
        )

        # global loss
        loss_global = self.l1(
            restored,
            gt
        )

        # edge loss
        gx_p, gy_p = self.sobel(restored)
        gx_gt, gy_gt = self.sobel(gt)

        edge_loss = (
            self.l1(gx_p, gx_gt)
            + self.l1(gy_p, gy_gt)
        )

        # multiscale loss
        loss_ms = self.l1(
            self.downsample(restored),
            self.downsample(gt)
        )

        # =====================================================
        # FINAL LOSS
        # =====================================================
        loss = (
            3.0 * loss_masked +
            0.5 * loss_global +
            0.5 * edge_loss +
            0.5 * loss_ms
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN detected")
            return None

        return loss

    @torch.no_grad()
    def super_resolution(self, rm_in):

        corrupted = rm_in[:, 0:3]
        structure = rm_in[:, 3:7]
        mask = rm_in[:, 7:10]

        mask = (mask > 0.5).float()

        B = corrupted.size(0)
        device = corrupted.device

        current = corrupted.clone()

        # =====================================================
        # REVERSE DIFFUSION
        # =====================================================
        for k in reversed(range(1, self.T + 1)):

            time = torch.full(
                (B,),
                k / self.T,
                device=device
            )

            inp = torch.cat([
                corrupted,
                structure,
                mask,
                current
            ], dim=1)

            # ==============================================
            # RESIDUAL PREDICTION
            # ==============================================
            residual = self.unet(inp, time)

            residual = torch.nan_to_num(
                residual
            ).clamp(-1, 1)

            pred = current + residual

            # ==============================================
            # HARD MASK CONSTRAINT
            # ==============================================
            current = (
                corrupted * (1 - mask)
                + pred * mask
            )

        return current.clamp(-1, 1)