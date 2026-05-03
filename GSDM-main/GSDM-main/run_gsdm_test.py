#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import argparse
import yaml
import logging

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.cuda.amp import autocast

import util
from model.model import DDPM, SPM

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

IMG_EXT = {'jpg','JPG','jpeg','JPEG','png','PNG','bmp','BMP','ppm','PPM'}

def list_images(path):
    return [f for f in sorted(os.listdir(path)) if f.split('.')[-1] in IMG_EXT]

def to01_np(t):
    x = t.detach().cpu().float()
    x = (x + 1.0) / 2.0
    if x.dim() == 4: x = x[0]
    if x.dim() == 3: x = x.permute(1,2,0).contiguous()
    x = x.clamp_(0,1).numpy()
    if x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    return x

def compute_psnr_ssim(pred01, gt01):
    if not HAS_SKIMAGE:
        mse = np.mean((gt01 - pred01) ** 2)
        psnr = 100.0 if mse == 0 else 10 * math.log10(1.0 / max(mse, 1e-12))
        return float(psnr), float('nan')
    psnr = sk_psnr(gt01, pred01, data_range=1.0)
    try:
        ssim = sk_ssim(gt01, pred01, data_range=1.0, channel_axis=-1)
    except TypeError:
        ssim = sk_ssim(gt01, pred01, data_range=1.0, multichannel=True)
    return float(psnr), float(ssim)

def compute_accuracy(pred01, gt01, thresholds=(0.05,0.10)):
    abs_err = np.abs(pred01 - gt01).mean(axis=2)
    return [float(np.mean(abs_err < th)) for th in thresholds]


class TestCorruptedDataset(Dataset):
    def __init__(self, corrupted_dir, gt_dir, resolution, max_images=1000):
        self.corrupted_dir = corrupted_dir
        self.gt_dir = gt_dir
        self.has_gt = gt_dir is not None and os.path.isdir(gt_dir)
        names = list_images(corrupted_dir)
        self.names = names[:max_images] if max_images and max_images > 0 else names
        self.tr = transforms.Compose([transforms.Resize(resolution), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        c_img = Image.open(os.path.join(self.corrupted_dir, name)).convert('RGB')
        c_t = self.tr(c_img) * 2 - 1  # [-1,1]
        sample = {"name": name, "corrupted": c_t}

        if self.has_gt:
            gt_img = Image.open(os.path.join(self.gt_dir, name)).convert('RGB')
            gt_t = self.tr(gt_img) * 2 - 1
            sample["gt"] = gt_t

        return sample


def load_yaml(cfg_path, phase="val"):
    with open(cfg_path, "r") as f:
        opt = yaml.safe_load(f)
    opt["phase"] = phase
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list)
    opt["distributed"] = len(opt["gpu_ids"]) > 1
    return opt


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser("Run GSDM (SPM + RM cold diffusion) on test set")
    parser.add_argument("--config", type=str, default="config/train_rm.yaml")
    parser.add_argument("--resume", type=str, required=True, help="checkpoint stem WITHOUT _gen.pth (e.g. .../I125000_E199)")
    parser.add_argument("--corrupted_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--acc_thresholds", type=str, default="0.05,0.10")
    args = parser.parse_args()

    opt = load_yaml(args.config, phase="val")
    opt.setdefault("path", {})
    opt["path"]["resume_state"] = args.resume

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    logger = logging.getLogger("run_gsdm_test")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)

    # RM
    my_model = DDPM(opt)
    my_model.load_network(logger=logger)

    # SPM
    spm = SPM(in_channels=3, cum=opt["SPM_cum"])
    state_dict = torch.load(opt["path"]["SPM_pretrain"], map_location="cpu")
    spm.load_state_dict(state_dict)
    spm = spm.to(device).eval()
    for p in spm.parameters():
        p.requires_grad = False

    res = opt["train_dataset"]["resolution"]  # [64,288]
    gt_dir = args.gt_dir if args.gt_dir else None

    ds = TestCorruptedDataset(args.corrupted_dir, gt_dir, res, max_images=args.max_images)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    thresholds = tuple(float(x) for x in args.acc_thresholds.split(","))

    rows = []
    pbar = tqdm(loader, total=len(loader), desc="GSDM inference")

    with torch.no_grad():
        for batch in pbar:
            corrupted = batch["corrupted"]  # CPU [-1,1]
            names = batch["name"]
            has_gt = "gt" in batch
            gt = batch.get("gt", None)

            # SPM on GPU
            with autocast(enabled=True, dtype=torch.float16):
                spm_in = ((corrupted + 1) / 2).to(device)
                sp = spm(spm_in)  # [B,1,H,W] in [-1,1]
            structure3 = sp.to(corrupted.dtype).repeat_interleave(3, dim=1).cpu()  # [B,3,H,W]

            # RM input: 6ch
            rm_in = torch.cat((corrupted, structure3), dim=1)  # [B,6,H,W]
            my_model.feed_data(rm_in)
            pred = my_model.test()  # [B,3,H,W] in [-1,1]

            bsz = pred.size(0)
            for i in range(bsz):
                name = names[i]
                pred_img = util.tensor2img(pred[i])  # uint8 HWC
                util.save_img(pred_img, os.path.join(args.out_dir, name))

                if has_gt:
                    pred01 = to01_np(pred[i])
                    gt01 = to01_np(gt[i])
                    psnr, ssim = compute_psnr_ssim(pred01, gt01)
                    accs = compute_accuracy(pred01, gt01, thresholds)
                    row = {"name": name, "psnr": psnr, "ssim": ssim}
                    for j, th in enumerate(thresholds):
                        row[f"acc<{th:.2f}"] = accs[j]
                    rows.append(row)

    if gt_dir and rows:
        csv_path = os.path.join(args.out_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        psnr_mean = np.nanmean([r["psnr"] for r in rows])
        ssim_mean = np.nanmean([r["ssim"] for r in rows])
        print(f"\nAverages on {len(rows)} images -> PSNR: {psnr_mean:.3f} dB, SSIM: {ssim_mean:.4f}")
        for th in thresholds:
            avg_acc = np.nanmean([r[f"acc<{th:.2f}"] for r in rows])
            print(f"Accuracy (<{th:.2f}): {avg_acc:.4f}")
        print(f"Per-image metrics saved to: {csv_path}")

    print(f"Done. Restored images saved to: {args.out_dir}")


if __name__ == "__main__":
    main()