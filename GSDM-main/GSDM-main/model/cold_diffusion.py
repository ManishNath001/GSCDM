import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT
        ).features[:16]

        self.vgg = vgg.eval()

        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y + 1) / 2

        return F.l1_loss(
            self.vgg(x),
            self.vgg(y)
        )


class ColdDiffusion(nn.Module):
    def __init__(self, unet, T=100):
        super().__init__()

        self.unet = unet
        self.T = T

        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptualLoss()

    def _alpha(self, k):
        return (k.float() / self.T).view(-1, 1, 1, 1)

    def forward(self, batch):
        corrupted = batch["corrupted"]
        gt = batch["gt"]
        structure = batch["structure"]
        mask = batch["mask"]

        B = corrupted.size(0)
        device = corrupted.device

        k = torch.randint(
            1, self.T + 1, (B,), device=device
        )

        alpha = self._alpha(k)

        # noisy clean estimate
        noisy_clean = (
            gt * (1 - alpha) +
            corrupted * alpha
        )

        inp = torch.cat([
            corrupted,
            structure,
            mask,
            noisy_clean
        ], dim=1)

        time = k.float() / self.T

        pred_clean = self.unet(inp, time)

        pred_clean = torch.clamp(pred_clean, -1.5, 1.5)

        restored = pred_clean.clamp(-1, 1)

        image_loss = self.l1(restored, gt)

        with torch.amp.autocast("cuda", enabled=False):
            perceptual_loss = self.perc(
                restored.float(),
                gt.float()
            )

        # edge loss
        edge_x_loss = self.l1(
            torch.abs(restored[:, :, :, 1:] - restored[:, :, :, :-1]),
            torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        )

        edge_y_loss = self.l1(
            torch.abs(restored[:, :, 1:, :] - restored[:, :, :-1, :]),
            torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
        )

        edge_loss = torch.clamp(edge_x_loss + edge_y_loss, 0, 10)

        loss = (
            1.0 * image_loss +
            0.6 * perceptual_loss +
            0.5 * edge_loss
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN/Inf loss detected")
            return None

        return loss

    @torch.no_grad()
    def super_resolution(self, rm_in):
        corrupted = rm_in[:, 0:3]
        structure = rm_in[:, 3:7]
        mask = rm_in[:, 7:10]

        B = corrupted.size(0)
        device = corrupted.device

        current = corrupted.clone()

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

            pred_clean = self.unet(inp, time)

            pred_clean = torch.nan_to_num(
                pred_clean,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0
            )

            pred_clean = torch.clamp(pred_clean, -1.5, 1.5)

            # 🔥 CRITICAL FIX: ANCHORED RECONSTRUCTION
            alpha = k / self.T

            current = (
                (1 - alpha) * pred_clean +
                alpha * corrupted
            )

        restored = current

        restored = torch.nan_to_num(
            restored,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        return restored.clamp(-1, 1)