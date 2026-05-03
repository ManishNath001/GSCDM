import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image

import torchvision.transforms as T


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

corrupted_dir = "/fab3/mtech/2024/manish.nath/dataset/ht_test/Corrupted_Image"

gt_dir = "/fab3/mtech/2024/manish.nath/dataset/ht_test/Intact_Image"

save_structure_dir = "./precomputed_test/structure"
save_mask_dir = "./precomputed_test/mask"

os.makedirs(save_structure_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)

transform = T.Compose([
    T.Resize((64, 288)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

files = sorted([
    f for f in os.listdir(corrupted_dir)
    if f.lower().endswith((
        ".png",
        ".jpg",
        ".jpeg"
    ))
])

print(f"Total test images: {len(files)}")


def build_structure(gray):

    sobelx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sobely = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    lap = cv2.Laplacian(
        gray,
        cv2.CV_32F
    )

    canny = cv2.Canny(
        (gray * 255).astype(np.uint8),
        80,
        160
    ).astype(np.float32) / 255.0

    structure = np.stack([
        sobelx,
        sobely,
        lap,
        canny
    ], axis=0)

    structure = structure / (
        np.max(np.abs(structure)) + 1e-8
    )

    return torch.from_numpy(structure).float()


def build_mask(corrupted, gt):

    gray_c = corrupted.mean(
        dim=1,
        keepdim=True
    )

    gray_g = gt.mean(
        dim=1,
        keepdim=True
    )

    pix_diff = torch.abs(
        gray_c - gray_g
    )

    edge_cx = torch.abs(
        gray_c[:, :, :, 1:] -
        gray_c[:, :, :, :-1]
    )

    edge_gx = torch.abs(
        gray_g[:, :, :, 1:] -
        gray_g[:, :, :, :-1]
    )

    edge_dx = F.pad(
        torch.abs(edge_cx - edge_gx),
        (0,1,0,0)
    )

    edge_cy = torch.abs(
        gray_c[:, :, 1:, :] -
        gray_c[:, :, :-1, :]
    )

    edge_gy = torch.abs(
        gray_g[:, :, 1:, :] -
        gray_g[:, :, :-1, :]
    )

    edge_dy = F.pad(
        torch.abs(edge_cy - edge_gy),
        (0,0,0,1)
    )

    diff = pix_diff + edge_dx + edge_dy

    diff = F.avg_pool2d(
        diff,
        kernel_size=5,
        stride=1,
        padding=2
    )

    diff = diff / (
        diff.max() + 1e-8
    )

    mask = (diff > 0.08).float()

    mask = mask.repeat(1,3,1,1)

    return mask


for fname in tqdm(files):

    try:

        c_path = os.path.join(
            corrupted_dir,
            fname
        )

        g_path = os.path.join(
            gt_dir,
            fname
        )

        corrupted_img = Image.open(
            c_path
        ).convert("RGB")

        gt_img = Image.open(
            g_path
        ).convert("RGB")

        corrupted = transform(
            corrupted_img
        ).unsqueeze(0)

        gt = transform(
            gt_img
        ).unsqueeze(0)

        gray = np.array(
            corrupted_img.resize((288, 64)).convert("L"),
            dtype=np.float32
        ) / 255.0

        structure = build_structure(gray)

        mask = build_mask(
            corrupted,
            gt
        )

        name = os.path.splitext(fname)[0] + ".pt"

        torch.save(
            structure,
            os.path.join(
                save_structure_dir,
                name
            )
        )

        torch.save(
            mask.squeeze(0),
            os.path.join(
                save_mask_dir,
                name
            )
        )

    except Exception as e:
        print(f"Skipping {fname}: {e}")

print("===================================")
print("Test preprocessing completed")
print("===================================")