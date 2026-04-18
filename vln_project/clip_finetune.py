#!/usr/bin/env python3
"""
clip_finetune.py
================
Offline training script. Run ONCE before launching ROS2.
Fine-tunes CLIP ViT-B/32 on your 5-class environment dataset.

Dataset layout (matches your sample_images folder exactly):
  sample_images/
    cafe_table/              <- put your photos here
    double_cabinet/
    first_2015_trash_can/
    single_cabinet/
    table/

Usage:
  python clip_finetune.py \
    --data_dir ~/vln_project/dataset/sample_images \
    --output_dir checkpoints/ \
    --epochs 30

Output:
  checkpoints/clip_finetuned.pt   (model weights)
  checkpoints/meta.json           (class list + embed dim)

To add new classes later:
  1. Add folder to sample_images/
  2. Add class name to CLASSES list below
  3. Re-run this script
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

try:
    import open_clip
    USE_OPEN_CLIP = True
except ImportError:
    import clip
    USE_OPEN_CLIP = False

# ── Edit this list when you add new objects ──────────────────────────────────
CLASSES = [
    "cafe_table",
    "double_cabinet",
    "first_2015_trash_can",
    "single_cabinet",
    "table",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class EnvDataset(Dataset):
    def __init__(self, data_dir: str, preprocess, classes: list):
        self.preprocess = preprocess
        self.c2i        = {c: i for i, c in enumerate(classes)}
        self.samples    = []

        for cls in classes:
            cls_dir = Path(data_dir) / cls
            if not cls_dir.exists():
                print(f"  [WARN] folder missing: {cls_dir}")
                continue
            found = 0
            for p in cls_dir.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(p), self.c2i[cls]))
                    found += 1
            print(f"  {cls:30s}  {found} images")

        if not self.samples:
            raise ValueError(f"No images found under {data_dir}")
        print(f"  Total: {len(self.samples)} images across {len(classes)} classes\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.preprocess(Image.open(path).convert("RGB"))
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# Model — CLIP encoder + MLP classification head
# ─────────────────────────────────────────────────────────────────────────────
class CLIPClassifier(nn.Module):
    """
    CLIP image encoder (frozen initially) + trainable MLP head.

    Architecture:
      CLIP ViT-B/32  →  512-dim L2-normalised embedding
        →  LayerNorm
        →  Linear(512 → 512)  →  GELU  →  Dropout(0.3)
        →  Linear(512 → 256)  →  GELU  →  Dropout(0.3)
        →  Linear(256 → num_classes)
    """
    def __init__(self, clip_model, embed_dim: int, num_classes: int):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.clip_model.encode_image(x).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return self.head(feats)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    print(f"\n{'='*55}")
    print(f"  CLIP Fine-tuning  |  device={DEVICE}")
    print(f"  Classes: {CLASSES}")
    print(f"{'='*55}\n")

    # Load CLIP backbone
    if USE_OPEN_CLIP:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        embed_dim = 512
    else:
        clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        embed_dim = 512
    clip_model = clip_model.to(DEVICE)

    # Dataset + split
    dataset = EnvDataset(args.data_dir, preprocess, CLASSES)
    val_n   = max(1, int(len(dataset) * 0.2))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(
        dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=2)

    # Model
    model = CLIPClassifier(clip_model, embed_dim, len(CLASSES)).to(DEVICE)

    # Phase 1: freeze CLIP, train head only
    for p in model.clip_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    phase2_started = False

    for epoch in range(1, args.epochs + 1):

        # Phase 2: unfreeze CLIP top layers at halfway point
        if epoch == (args.epochs // 2 + 1) and not phase2_started:
            print(f"\n  [Phase 2] Unfreezing CLIP encoder (lr x0.1)...")
            for p in model.clip_model.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW([
                {"params": model.clip_model.parameters(), "lr": args.lr * 0.1},
                {"params": model.head.parameters(),       "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1)
            phase2_started = True

        # Train
        model.train()
        tot_loss = correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item() * len(labels)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += len(labels)
        scheduler.step()

        # Validate
        model.eval()
        val_c = val_t = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                val_c += (model(imgs).argmax(1) == labels).sum().item()
                val_t += len(labels)
        val_acc = val_c / val_t

        marker = "  *" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={tot_loss/total:.4f}  "
              f"train={correct/total:.3f}  "
              f"val={val_acc:.3f}{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "clip_finetuned.pt"))

    # Save metadata
    meta = {
        "classes":      CLASSES,
        "embed_dim":    embed_dim,
        "best_val_acc": round(best_val_acc, 4),
        "use_open_clip": USE_OPEN_CLIP,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Best val accuracy : {best_val_acc:.3f}")
    print(f"  Saved to          : {args.output_dir}/")
    print(f"    - clip_finetuned.pt")
    print(f"    - meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True,
                   help="Path to sample_images/ folder")
    p.add_argument("--output_dir", default="checkpoints",
                   help="Where to save weights and meta.json")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    train(p.parse_args())
