import os
import argparse
import yaml
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

import util

from model.model import DDPM


def save_checkpoint(model, opt, epoch, i):

    save_dir = opt['path']['log']

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"I{i}_E{epoch}_gen.pth"
    )

    torch.save({
        "model": model.netG.state_dict(),
        "iter": i,
        "epoch": epoch
    }, save_path)

    print(f"Saved: {save_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        required=True
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = util.rm_train_dataset(
        corrupted_dir=opt['train_dataset']['corrupted_dir'],
        gt_dir=opt['train_dataset']['gt_dir'],
        structure_dir=opt['train_dataset']['structure_dir'],
        mask_dir=opt['train_dataset']['mask_dir']
    )

    loader = DataLoader(
        dataset,
        batch_size=opt['train_dataset']['batch_size'],
        shuffle=opt['train_dataset']['use_shuffle'],
        num_workers=opt['train_dataset']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    model = DDPM(opt)

    model.netG = model.netG.to(device)

    optimizer = torch.optim.AdamW(
        model.netG.parameters(),
        lr=float(opt['train']['optimizer']['lr']),
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt['train']['n_iter']
    )

    n_iter = opt['train']['n_iter']

    print_freq = opt['train']['print_freq']

    save_freq = opt['train']['save_checkpoint_freq']

    scaler = torch.amp.GradScaler("cuda")

    i = 0

    for epoch in range(100000):

        for data in tqdm(loader):

            for k in data:
                if torch.is_tensor(data[k]):
                    data[k] = data[k].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):

                loss = model.netG(data)

                if loss is None:
                    optimizer.zero_grad(set_to_none=True)
                    continue

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.netG.parameters(),
                0.5
            )

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()

            if i % print_freq == 0:

                print(
                    f"Iter {i} | "
                    f"Loss: {loss.item():.6f}"
                )

            if i % save_freq == 0:

                save_checkpoint(
                    model,
                    opt,
                    epoch,
                    i
                )

            i += 1

            if i >= n_iter:

                print("Training completed")

                return


if __name__ == "__main__":
    main()