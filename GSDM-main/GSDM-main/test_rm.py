import os
import argparse
import torch
import yaml
import numpy as np

from tqdm import tqdm

import torchvision.utils as vutils

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import util
import model.networks as networks


def load_checkpoint(model, ckpt_path):

    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu"
    )

    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    new_state_dict = {}

    for k, v in state_dict.items():

        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(
        new_state_dict,
        strict=False
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--gt_dir", required=True)

    parser.add_argument("--structure_dir", required=True)
    parser.add_argument("--mask_dir", required=True)

    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open(args.config, "r") as f:
        opt = yaml.safe_load(f)

    opt["phase"] = "test"

    model = networks.define_G(opt)

    load_checkpoint(model, args.ckpt)

    model = model.to(device)

    model.eval()

    dataset = util.rm_train_dataset(
        corrupted_dir=args.input_dir,
        gt_dir=args.gt_dir,
        structure_dir=args.structure_dir,
        mask_dir=args.mask_dir,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    os.makedirs(args.output_dir, exist_ok=True)

    psnr_list = []
    ssim_list = []

    for i, data in enumerate(tqdm(loader)):

        corrupted = data["corrupted"].to(device)
        structure = data["structure"].to(device)
        mask = data["mask"].to(device)

        gt = data["gt"]

        residual = torch.zeros_like(corrupted)

        rm_input = torch.cat([
            corrupted,
            structure,
            mask,
            residual
        ], dim=1)

        with torch.no_grad():

            output = model.super_resolution(rm_input)

        output = (output.clamp(-1, 1) + 1) / 2

        save_path = os.path.join(
            args.output_dir,
            f"{i:05d}.png"
        )

        vutils.save_image(output, save_path)

        out_img = output.squeeze().cpu().numpy().transpose(1, 2, 0)

        gt_img = (
            gt.squeeze().cpu().numpy().transpose(1, 2, 0) + 1
        ) / 2

        psnr_list.append(
            psnr(
                gt_img,
                out_img,
                data_range=1.0
            )
        )

        ssim_list.append(
            ssim(
                gt_img,
                out_img,
                channel_axis=2,
                data_range=1.0
            )
        )

    print("\n================ METRICS ================")

    print(f"Average PSNR: {np.mean(psnr_list):.4f}")

    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

    print("=========================================")


if __name__ == "__main__":
    main()