#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster corrosion images by visual similarity (feature-based unsupervised grouping)
and estimate the ratio of images per corrosion type.

Steps:
1. Load all images from a directory.
2. Extract features using a pretrained CNN (ResNet50, default).
3. Cluster features with KMeans (or automatic K selection).
4. Save per-cluster example thumbnails and ratios.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# -------------------- Dataset --------------------
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None, max_images=None):
        self.image_dir = image_dir
        self.paths = [os.path.join(image_dir, f)
                      for f in os.listdir(image_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if max_images:
            self.paths = self.paths[:max_images]
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        return x, str(p), img  # include PIL for thumbnails


# -------------------- Feature Extraction --------------------
def extract_features(model, loader, device):
    model.eval()
    features, names, thumbs = [], [], []
    with torch.no_grad():
        for xb, nameb, thumbb in tqdm(loader, desc="Extracting features"):
            xb = torch.stack(xb).to(device, non_blocking=True)  # stack manually due to collate_fn
            feats = model(xb).cpu()
            features.append(feats)
            names.extend(nameb)
            thumbs.extend(thumbb)
    feats = torch.cat(features, dim=0)
    return feats.numpy(), names, thumbs


# -------------------- KMeans Clustering --------------------
def auto_kmeans(features, k_min=3, k_max=8):
    best_k = k_min
    best_score = -1
    best_model = None
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        print(f"k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, kmeans
    print(f"✅ Optimal K = {best_k} (silhouette={best_score:.4f})")
    return best_model, best_k


# -------------------- Save Cluster Samples --------------------
def save_cluster_examples(names, labels, thumbs, out_dir, n_samples=10):
    os.makedirs(out_dir, exist_ok=True)
    unique_labels = sorted(set(labels))
    cluster_counts = {lbl: 0 for lbl in unique_labels}

    for lbl in unique_labels:
        cluster_dir = os.path.join(out_dir, f"cluster_{lbl:02d}")
        os.makedirs(cluster_dir, exist_ok=True)
        indices = np.where(labels == lbl)[0]
        cluster_counts[lbl] = len(indices)
        for i in indices[:n_samples]:
            name = os.path.basename(names[i])
            thumbs[i].save(os.path.join(cluster_dir, name))

    total = sum(cluster_counts.values())
    print("\nCluster ratios:")
    for lbl, count in cluster_counts.items():
        print(f"  Cluster {lbl}: {count} images ({count/total:.2%})")
    return cluster_counts


# -------------------- Visualization --------------------
def plot_cluster_distribution(cluster_counts, out_dir):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4))
    keys = [f"C{c}" for c in cluster_counts.keys()]
    vals = [v for v in cluster_counts.values()]
    sns.barplot(x=keys, y=vals, palette="viridis")
    plt.title("Cluster Distribution (Corrosion Types)")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_distribution.png"))
    plt.close()


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Corrosion Clustering via ResNet features")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to all corrosion images")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--auto_k", nargs=2, type=int, default=[3, 8], help="Min and max K for auto-selection")
    parser.add_argument("--model", type=str, default="resnet50", help="Feature extractor backbone")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset + Loader
    ds = ImageFolderDataset(args.image_dir, transform, max_images=args.max_images)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: tuple(zip(*batch))  # ✅ prevents PIL Image batching error
    )

    # Feature extractor
    if args.model.lower() == "resnet50":
        backbone = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(backbone.children())[:-1]))  # remove FC layer
        model = torch.nn.Sequential(model, torch.nn.Flatten())  # output 2048-d vector
    else:
        raise ValueError("Only resnet50 supported in this version.")
    model = model.to(device).eval()

    print(f"Extracting features from {len(ds)} images...")
    feats, names, thumbs = extract_features(model, loader, device)

    # Auto KMeans
    kmeans, best_k = auto_kmeans(feats, k_min=args.auto_k[0], k_max=args.auto_k[1])
    labels = kmeans.predict(feats)

    # Save per-cluster samples + plot
    cluster_counts = save_cluster_examples(names, labels, thumbs, args.out_dir)
    plot_cluster_distribution(cluster_counts, args.out_dir)

    # Save numeric results
    out_csv = os.path.join(args.out_dir, "cluster_ratios.csv")
    total = sum(cluster_counts.values())
    with open(out_csv, "w") as f:
        f.write("Cluster,Count,Ratio\n")
        for lbl, count in cluster_counts.items():
            f.write(f"{lbl},{count},{count/total:.4f}\n")

    print(f"\n✅ Clustering complete. Results saved in: {args.out_dir}")


if __name__ == "__main__":
    main()

