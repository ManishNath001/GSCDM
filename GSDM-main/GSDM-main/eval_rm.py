import os
import argparse
import yaml
import csv
import math
import numpy as np
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast

import util
from model.model import DDPM, SPM

# Optional LPIPS
try:
    import lpips  # pip install lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

# Optional skimage for PSNR/SSIM
try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

from torchvision import transforms
from PIL import Image


def load_yaml(cfg_path, phase="val"):
    with open(cfg_path, "r") as f:
        opt = yaml.safe_load(f)
    opt["phase"] = phase
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    opt["distributed"] = len(opt["gpu_ids"]) > 1
    return opt


def to01_np(t):
    """[-1,1] torch [B,C,H,W] or [C,H,W] -> [0,1] numpy HWC"""
    if isinstance(t, torch.Tensor):
        x = t.detach().cpu().float()
        x = (x + 1.0) / 2.0
        if x.dim() == 4:  # BCHW -> first element; caller should handle per-item
            x = x[0]
        if x.dim() == 3:
            x = x.permute(1, 2, 0).contiguous()  # HWC
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.clamp_(0, 1).numpy()
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        return x
    elif isinstance(t, np.ndarray):
        # assume already HWC in [0,1]
        return t
    else:
        raise TypeError(f"Unsupported type for to01_np: {type(t)}")


def compute_psnr_ssim(pred01, gt01):
    """Both inputs HWC, float32 in [0,1]. Returns (psnr, ssim)."""
    if HAS_SKIMAGE:
        psnr = sk_psnr(gt01, pred01, data_range=1.0)
        # Support both new and old skimage signatures
        try:
            ssim = sk_ssim(gt01, pred01, data_range=1.0, channel_axis=-1)
        except TypeError:
            ssim = sk_ssim(gt01, pred01, data_range=1.0, multichannel=True)
    else:
        # Lightweight PSNR fallback: PSNR = 10 log10(1 / MSE)
        mse = np.mean((gt01 - pred01) ** 2)
        psnr = 100.0 if mse == 0 else 10 * math.log10(1.0 / max(mse, 1e-12))
        ssim = float("nan")
    return float(psnr), float(ssim)


def compute_lpips(lpips_net, pred01, gt01, device="cuda"):
    """pred01, gt01: HWC [0,1] -> convert to torch NCHW [-1,1]"""
    if not HAS_LPIPS or lpips_net is None:
        return float("nan")

    def hwc01_to_nchw11(x):
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()  # 1xCxHxW
        t = t * 2 - 1
        return t.to(device)

    with torch.no_grad():
        d = lpips_net(hwc01_to_nchw11(pred01), hwc01_to_nchw11(gt01))
    return float(d.item())


# ---------------------- Accuracy helpers ----------------------

def load_mask(mask_path, resolution_hw):
    """
    Load a grayscale mask, resize to (H,W), binarize at 0.5.
    Returns HxW boolean where True indicates the (corrupted) region to evaluate.
    """
    if not mask_path or not os.path.isfile(mask_path):
        return None
    m = Image.open(mask_path).convert('L')
    tr = transforms.Compose([transforms.Resize(resolution_hw), transforms.ToTensor()])  # 1xHxW in [0,1]
    m = tr(m)[0].numpy()
    return (m >= 0.5)


def pixel_accuracy(pred01, gt01, thresh=0.05, mask=None):
    """
    Fraction of pixels with |pred-gt|_per-pixel (mean over channels) <= thresh.
    pred01, gt01: HxWx3 in [0,1]
    mask: optional HxW boolean; if provided, evaluate only where mask==True
    """
    err = np.abs(pred01 - gt01).mean(axis=2)  # HxW
    if mask is not None:
        if mask.sum() == 0:
            return float('nan')
        return float((err[mask] <= thresh).mean())
    return float((err <= thresh).mean())


# --------------------------------------------------------------

def build_eval_loader(opt, num_samples=None, mask_dir=None):
    """
    Uses the same folder paths from train_dataset section to build an eval loader.
    If you want a dedicated eval set, point train_dataset.{corrupted_dir,gt_dir}
    to your eval folders before running this script, or adapt here.
    """
    ds = util.rm_train_dataset(
        corrupted_dir=opt["train_dataset"]["corrupted_dir"],
        gt_dir=opt["train_dataset"]["gt_dir"],
        data_shape=opt["train_dataset"]["resolution"],
        train=False
    )

    # if mask_dir is provided, we’ll attach mask paths by monkey-patching __getitem__
    if mask_dir and os.path.isdir(mask_dir):
        orig_getitem = ds.__getitem__

        def getitem_with_mask(idx):
            sample = orig_getitem(idx)
            # ds.img_list contains filenames; util.rm_train_dataset preserves names
            name = ds.img_list[idx]
            sample["mask_path"] = os.path.join(mask_dir, name)
            return sample

        ds.__getitem__ = getitem_with_mask  # type: ignore
    else:
        # no mask_dir -> provide empty path
        orig_getitem = ds.__getitem__

        def getitem_no_mask(idx):
            sample = orig_getitem(idx)
            sample["mask_path"] = ""
            return sample

        ds.__getitem__ = getitem_no_mask  # type: ignore

    if num_samples is not None and num_samples > 0:
        # Take a deterministic subset of evenly spaced indices
        n = len(ds)
        step = max(1, n // num_samples)
        idxs = list(range(0, n, step))[:num_samples]
        from torch.utils.data import Subset
        ds = Subset(ds, idxs)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )
    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_rm.yaml")
    parser.add_argument("--num_samples", type=int, default=200, help="evaluate on N evenly spaced samples")
    parser.add_argument("--save_dir", type=str, default="", help="where to save CSV and optional previews")
    parser.add_argument("--save_previews", action="store_true", help="save side-by-side PNGs")
    parser.add_argument("--mask_dir", type=str, default="", help="optional dir with binary masks (same filenames)")
    parser.add_argument("--acc_thresholds", type=str, default="0.05,0.10",
                        help="comma-separated thresholds on [0,1] for accuracy (e.g., '0.05,0.10')")
    args = parser.parse_args()

    # parse thresholds
    acc_thresholds = [float(x) for x in args.acc_thresholds.split(",") if x.strip()]

    opt = load_yaml(args.config, phase="val")
    device = torch.device("cuda")

    # Output paths
    if args.save_dir:
        out_dir = args.save_dir
    else:
        out_dir = os.path.join(opt["path"]["log"], opt["train_dataset"]["dataset_name"], "rm_eval")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")

    # Build model
    my_model = DDPM(opt)
    my_model.load_network(logger=None)
    my_model.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")

    # SPM (inference-only)
    spm = SPM(in_channels=3, cum=opt["SPM_cum"])
    state_dict = torch.load(opt["path"]["SPM_pretrain"], map_location="cpu")
    spm.load_state_dict(state_dict)
    spm = spm.to(device).eval()
    for p in spm.parameters():
        p.requires_grad = False

    # LPIPS (optional)
    lpips_net = None
    if HAS_LPIPS:
        lpips_net = lpips.LPIPS(net='alex').to(device).eval()

    # Data
    loader = build_eval_loader(opt, num_samples=args.num_samples, mask_dir=args.mask_dir if args.mask_dir else None)
    res_hw = tuple(opt["train_dataset"]["resolution"])  # (H, W)

    # Eval loop
    rows = []
    pbar = tqdm(loader, desc="Evaluating")
    for i, batch in enumerate(pbar):
        img_corrupted = batch["corrupted"]  # [-1,1], BCHW
        img_gt       = batch["gt"]          # [-1,1]
        mask_path    = batch.get("mask_path", [""])[0] if isinstance(batch.get("mask_path", ""), list) else batch.get("mask_path", "")

        # SPM structure (AMP for speed/memory)
        with torch.inference_mode():
            with autocast(enabled=True, dtype=torch.float16):
                spm_in = ((img_corrupted + 1) / 2).to(device, non_blocking=True)
                spm_out = spm(spm_in)  # [B,1,H,W]
        structure_3c = spm_out.to(img_corrupted.dtype).repeat_interleave(3, dim=1).cpu()

        # Model inference (concatenate like training)
        input_data = torch.cat((img_corrupted, structure_3c), dim=1)
        with torch.no_grad():
            my_model.feed_data(input_data)
            pred = my_model.test()  # assume returns BCHW in [-1,1]

        # Convert to numpy [0,1] HWC
        pred01 = to01_np(pred)
        gt01   = to01_np(img_gt)

        # Metrics
        psnr, ssim = compute_psnr_ssim(pred01, gt01)
        lp = compute_lpips(lpips_net, pred01, gt01, device=device) if lpips_net else float("nan")

        # Accuracy (global)
        acc_global = {f"acc@{t:.2f}": pixel_accuracy(pred01, gt01, thresh=t, mask=None) for t in acc_thresholds}

        # Accuracy (masked, optional)
        m = load_mask(mask_path, res_hw) if mask_path else None
        if m is not None:
            acc_masked = {f"acc_mask@{t:.2f}": pixel_accuracy(pred01, gt01, thresh=t, mask=m) for t in acc_thresholds}
        else:
            acc_masked = {f"acc_mask@{t:.2f}": float('nan') for t in acc_thresholds}

        row = {"index": i, "psnr": psnr, "ssim": ssim, "lpips": lp, **acc_global, **acc_masked}
        rows.append(row)

        # progress bar
        pbar.set_postfix(psnr=f"{psnr:.2f}", ssim=f"{ssim:.4f}", **{k: f"{v:.3f}" for k, v in acc_global.items()})

        # Optional preview PNGs
        if args.save_previews:
            # Grid: corrupted | structure (3c) | pred | gt
            row_t = torch.cat(
                (img_corrupted.squeeze(0), structure_3c.squeeze(0), pred.cpu().squeeze(0), img_gt.squeeze(0)),
                dim=2
            )
            preview_np = util.tensor2img(row_t)  # uint8 RGB HWC
            util.save_img(preview_np, os.path.join(out_dir, f"preview_{str(i).zfill(5)}.png"))

    # Write CSV + print averages
    # Determine column order
    base_cols = ["index", "psnr", "ssim", "lpips"]
    acc_cols  = [f"acc@{t:.2f}" for t in acc_thresholds]
    accm_cols = [f"acc_mask@{t:.2f}" for t in acc_thresholds]
    cols = base_cols + acc_cols + accm_cols

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, float('nan')) for k in cols})

    def _avg(key):
        vals = [r.get(key, float('nan')) for r in rows if not np.isnan(r.get(key, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    print("\n=== Averages ===")
    for k in cols:
        if k == "index":
            continue
        print(f"{k}: {_avg(k):.6f}")

    print(f"\nPer-image metrics saved to: {csv_path}")
    if args.save_previews:
        print(f"Previews saved to: {out_dir}")


if __name__ == "__main__":
    main()

