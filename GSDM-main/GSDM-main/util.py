import os
import torch

from torch.utils.data import Dataset

from PIL import Image

import torchvision.transforms as transforms


class rm_train_dataset(Dataset):

    def __init__(
        self,
        corrupted_dir,
        gt_dir,
        structure_dir,
        mask_dir
    ):

        self.corrupted_dir = corrupted_dir
        self.gt_dir = gt_dir
        self.structure_dir = structure_dir
        self.mask_dir = mask_dir

        self.corrupted_list = sorted([
            f for f in os.listdir(corrupted_dir)
            if f.lower().endswith((
                ".png",
                ".jpg",
                ".jpeg"
            ))
        ])

        self.length = len(self.corrupted_list)

        self.transform = transforms.Compose([
            transforms.Resize((64, 288)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

        print(f"Dataset loaded | Using {self.length} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        corrupted_name = self.corrupted_list[idx]

        base = os.path.splitext(corrupted_name)[0]

        corrupted_path = os.path.join(
            self.corrupted_dir,
            corrupted_name
        )

        gt_path = os.path.join(
            self.gt_dir,
            corrupted_name
        )

        structure_path = os.path.join(
            self.structure_dir,
            base + ".pt"
        )

        mask_path = os.path.join(
            self.mask_dir,
            base + ".pt"
        )

        corrupted = self.transform(
            Image.open(corrupted_path).convert("RGB")
        )

        gt = self.transform(
            Image.open(gt_path).convert("RGB")
        )

        structure = torch.load(
            structure_path,
            map_location="cpu"
        ).float()

        mask = torch.load(
            mask_path,
            map_location="cpu"
        ).float()

        # ================= SHAPE FIXES =================

        if structure.dim() == 4:
            structure = structure.squeeze(0)

        if mask.dim() == 4:
            mask = mask.squeeze(0)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # ================= STRUCTURE NORMALIZATION =================

        structure = structure.float()

        if structure.abs().max() > 1:
            structure = structure / (
                structure.abs().max() + 1e-8
            )

        structure = structure.clamp(-1, 1)

        # ================= MASK =================

        mask = (mask > 0.5).float()

        return {
            "corrupted": corrupted,
            "gt": gt,
            "structure": structure,
            "mask": mask
        }