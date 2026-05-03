import os
import torch
import argparse
import logging
from tqdm import tqdm
import yaml
import util

from model.mask_predictor import MaskPredictor
from model.model import SPM
from torch.amp import autocast, GradScaler


def load_yaml(args):
    with open(args.config, "r") as f:
        opt = yaml.safe_load(f)

    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    return opt


def run_spm(spm, corrupted):
    with torch.no_grad():
        spm_in = (corrupted + 1) / 2
        sp = spm(spm_in)   # NO .cpu()
    return sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_mpm.yaml")
    args = parser.parse_args()

    opt = load_yaml(args)

    logger = logging.getLogger("MPM")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    dataset = util.mpm_train_dataset(
        corrupted_dir=opt["train_dataset"]["corrupted_dir"],
        gt_dir=opt["train_dataset"]["gt_dir"],
        mask_dir=opt["train_dataset"]["mask_dir"],
        data_shape=opt["train_dataset"]["resolution"],  # ✅ FIXED
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt["train_dataset"]["batch_size"],
        shuffle=True,
        num_workers=opt["train_dataset"]["num_workers"],
    )

    logger.info("MPM Dataset loaded")

    device = torch.device("cuda")

    mpm = MaskPredictor(in_ch=6, base_ch=4).to(device)

    # ✅ FIXED LOSS
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(mpm.parameters(), lr=opt["train"]["lr"])
    scaler = GradScaler("cuda")

    # SPM on CPU
    spm = SPM(in_channels=3, cum=opt["SPM_cum"])
    spm.load_state_dict(torch.load(opt["path"]["SPM_pretrain"], map_location="cpu"))
    spm = spm.to(device).eval()

    logger.info("SPM loaded and frozen")

    save_dir = opt["path"]["checkpoint"]
    os.makedirs(save_dir, exist_ok=True)

    step = 0
    max_iter = opt["train"]["n_iter"]

    while step < max_iter:
        for batch in tqdm(loader):
            step += 1
            if step > max_iter:
                break

            corrupted = batch["corrupted"].to(device)
            gt_mask = batch["mask"].to(device)

            # SPM
            sp = run_spm(spm, corrupted)
            structure3 = sp.repeat(1, 3, 1, 1)

            # MPM input
            mpm_input = torch.cat([corrupted, structure3], dim=1)

            with autocast("cuda"):
                pred_mask = mpm(mpm_input)
                loss = criterion(pred_mask, gt_mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % opt["train"]["print_freq"] == 0:
                logger.info(f"Step {step}: Loss {loss.item():.4f}")

            if step % opt["train"]["save_freq"] == 0:
                save_path = os.path.join(save_dir, f"mpm_{step}.pth")
                torch.save(mpm.state_dict(), save_path)
                logger.info(f"Saved: {save_path}")


if __name__ == "__main__":
    main()