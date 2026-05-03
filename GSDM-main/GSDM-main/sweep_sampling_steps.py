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


class TestCorruptedDataset(Dataset):
    def __init__(self, corrupted_dir, gt_dir, resolution, max_images=1000):
        self.corrupted_dir = corrupted_dir
        self.gt_dir = gt_dir
        self.has_gt = gt_dir is not None and os.path.isdir(gt_dir)

        names = list_images(corrupted_dir)
        self.names = names[:max_images] if max_images and max_images > 0 else names

        self.tr = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        c_img = Image.open(os.path.join(self.corrupted_dir, name)).convert('RGB')
        c_t = self.tr(c_img) * 2 - 1  # [-1,1]
        sample = {"name": name, "corrupted": c_t}

        if self.has_gt:
            gt_img = Image.open(os.path.join(self.gt_dir, name)).convert('RGB')
            gt_t = self.tr(gt_img) * 2 - 1  # [-1,1]
            sample["gt"] = gt_t

        return sample


def load_yaml(cfg_path, phase="val"):
    with open(cfg_path, "r") as f:
        opt = yaml.safe_load(f)
    opt["phase"] = phase
    gpu_list = ",".join(str(x) for x in opt.get("gpu_ids", []))
    if gpu_list:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list)
    opt["distributed"] = len(opt.get("gpu_ids", [])) > 1
    return opt


def unwrap_netG(ddpm_model: DDPM):
    netG = ddpm_model.netG
    return netG.module if isinstance(netG, torch.nn.DataParallel) else netG


def to01_np(t: torch.Tensor):
    x = t.detach().cpu().float()
    x = (x + 1.0) / 2.0
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 3:
        x = x.permute(1, 2, 0).contiguous()
    x = x.clamp_(0, 1).numpy()
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
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


def compute_accuracy(pred01, gt01, thresholds=(0.05, 0.10)):
    abs_err = np.abs(pred01 - gt01).mean(axis=2)  # HxW
    return [float(np.mean(abs_err < th)) for th in thresholds]


@torch.no_grad()
def eval_one_setting(netG, spm, loader, device, sampling_steps, thresholds=(0.05, 0.10)):
    psnrs, ssims = [], []
    acc05, acc10 = [], []

    for batch in tqdm(loader, desc=f"steps={sampling_steps}", leave=False):
        corrupted = batch["corrupted"].to(device, non_blocking=True)  # [-1,1]
        gt = batch.get("gt", None)
        if gt is None:
            raise RuntimeError("GT not found. Provide --gt_dir for metric sweep.")
        gt = gt.to(device, non_blocking=True)

        # ---- SPM in FP32 (stable) ----
        spm_in = ((corrupted + 1) / 2).float()  # [0,1] float32
        sp = spm(spm_in)  # [B,1,H,W] in [-1,1]
        structure3 = sp.repeat_interleave(3, dim=1).to(dtype=corrupted.dtype)  # [B,3,H,W]

        rm_in = torch.cat([corrupted, structure3], dim=1)  # [B,6,H,W]

        # ---- RM inference ----
        try:
            pred = netG.super_resolution(rm_in, sampling_steps=sampling_steps)
        except TypeError:
            pred = netG.super_resolution(rm_in)  # fallback if signature is old
        pred = pred.clamp(-1, 1)

        # metrics per item
        bsz = pred.size(0)
        for i in range(bsz):
            pred01 = to01_np(pred[i])
            gt01 = to01_np(gt[i])

            psnr, ssim = compute_psnr_ssim(pred01, gt01)
            a05, a10 = compute_accuracy(pred01, gt01, thresholds=thresholds)

            psnrs.append(psnr)
            ssims.append(ssim)
            acc05.append(a05)
            acc10.append(a10)

    return {
        "sampling_steps": int(sampling_steps),
        "psnr": float(np.nanmean(psnrs)),
        "ssim": float(np.nanmean(ssims)),
        "acc<0.05": float(np.nanmean(acc05)),
        "acc<0.10": float(np.nanmean(acc10)),
    }


def main():
    parser = argparse.ArgumentParser("Sweep cold diffusion sampling_steps and report metrics")
    parser.add_argument("--config", type=str, default="config/train_rm.yaml")
    parser.add_argument("--resume", type=str, required=True,
                        help="checkpoint stem WITHOUT _gen.pth (e.g. .../I50000_E80)")
    parser.add_argument("--corrupted_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=200,
                        help="Use a smaller value (e.g., 200) to pick best steps quickly; then rerun with 1000.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--steps", type=str, default="1,5,10,25,50",
                        help="comma-separated sampling steps to test")
    parser.add_argument("--out_csv", type=str, default="sampling_sweep.csv")
    args = parser.parse_args()

    opt = load_yaml(args.config, phase="val")
    opt.setdefault("path", {})
    opt["path"]["resume_state"] = args.resume

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger("sweep")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s sweep: %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)

    # ---- RM ----
    ddpm = DDPM(opt)
    ddpm.load_network(logger=logger)
    netG = unwrap_netG(ddpm).to(device).eval()

    logger.info(f"netG: {type(netG)} | T={getattr(netG, 'T', 'NA')}")

    # ---- SPM ----
    spm = SPM(in_channels=3, cum=opt["SPM_cum"])
    state_dict = torch.load(opt["path"]["SPM_pretrain"], map_location="cpu")
    spm.load_state_dict(state_dict)
    spm = spm.to(device).eval()
    for p in spm.parameters():
        p.requires_grad = False

    # ---- Data ----
    res = opt["train_dataset"]["resolution"]
    ds = TestCorruptedDataset(args.corrupted_dir, args.gt_dir, res, max_images=args.max_images)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    steps_list = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    results = []

    for s in steps_list:
        r = eval_one_setting(netG, spm, loader, device, sampling_steps=s, thresholds=(0.05, 0.10))
        results.append(r)
        logger.info(f"steps={r['sampling_steps']:>3d} | PSNR={r['psnr']:.3f} | SSIM={r['ssim']:.4f} | "
                    f"Acc<0.05={r['acc<0.05']:.4f} | Acc<0.10={r['acc<0.10']:.4f}")

    # Sort by PSNR desc
    results = sorted(results, key=lambda x: x["psnr"], reverse=True)

    # Print nice table
    print("\n=== Sampling Steps Sweep (sorted by PSNR) ===")
    print(f"{'steps':>6s} | {'PSNR':>8s} | {'SSIM':>8s} | {'Acc<0.05':>10s} | {'Acc<0.10':>10s}")
    print("-" * 55)
    for r in results:
        print(f"{r['sampling_steps']:6d} | {r['psnr']:8.3f} | {r['ssim']:8.4f} | {r['acc<0.05']:10.4f} | {r['acc<0.10']:10.4f}")

    # Save CSV
    out_csv = args.out_csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sampling_steps", "psnr", "ssim", "acc<0.05", "acc<0.10"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nSaved CSV: {out_csv}")


if __name__ == "__main__":
    main()