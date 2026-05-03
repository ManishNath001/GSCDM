import os
import torch
import argparse
import yaml
from tqdm import tqdm
import util

from model.model import DDPM, SPM
from model.mask_predictor import MaskPredictor


def run_spm(spm, corrupted):
    with torch.no_grad():
        spm_in = (corrupted + 1) / 2
        sp = spm(spm_in)
    return sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sampling_steps", type=int, default=1)
    args = parser.parse_args()

    opt = yaml.safe_load(open(args.config))

    device = torch.device("cuda")

    # RM model
    model = DDPM(opt)
    model.load_network(path=args.model_path)
    model.netG.eval()

    print(f"Loaded RM model from {args.model_path}")
    print(f"ColdDiffusion T: {model.netG.T}")
    print(f"Sampling steps: {args.sampling_steps}")

    # SPM
    spm = SPM(in_channels=3, cum=opt["SPM_cum"])
    spm.load_state_dict(torch.load(opt["path"]["SPM_pretrain"]))
    spm = spm.to(device).eval()

    # MPM
    mpm = MaskPredictor(in_ch=6, base_ch=4)
    mpm.load_state_dict(torch.load(opt["path"]["MPM_pretrain"]))
    mpm = mpm.to(device).eval()

    # Dataset
    dataset = util.rm_test_dataset(
        input_dir=opt["val_dataset"]["input_dir"],
        gt_dir=opt["val_dataset"]["output_dir"],
        data_shape=opt["val_dataset"]["resolution"],
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    save_dir = opt["path"]["results"]
    os.makedirs(save_dir, exist_ok=True)

    psnr_total = 0
    ssim_total = 0
    count = 0

    for data in tqdm(loader, desc="Inference"):
        corrupted = data["corrupted"].to(device)
        gt = data["gt"].to(device)

        # SPM
        sp = run_spm(spm, corrupted)
        structure3 = sp.repeat(1, 3, 1, 1)

        # MPM
        mpm_input = torch.cat([corrupted, structure3], dim=1)
        mask = torch.sigmoid(mpm(mpm_input))
        mask3 = mask.repeat(1, 3, 1, 1)

        # RM input
        rm_input = torch.cat([corrupted, structure3, mask3], dim=1)

        # Inference
        pred = model.netG.super_resolution(
            rm_input,
            sampling_steps=args.sampling_steps
        )

        pred = pred.clamp(-1, 1)

        # Metrics
        pred_np = util.tensor2img(pred)
        gt_np = util.tensor2img(gt)

        psnr = util.calculate_psnr(pred_np, gt_np)
        ssim = util.calculate_ssim(pred_np, gt_np)

        psnr_total += psnr
        ssim_total += ssim
        count += 1

    print("\n==== FINAL RESULTS ====")
    print(f"PSNR: {psnr_total / count:.3f}")
    print(f"SSIM: {ssim_total / count:.4f}")


if __name__ == "__main__":
    main()