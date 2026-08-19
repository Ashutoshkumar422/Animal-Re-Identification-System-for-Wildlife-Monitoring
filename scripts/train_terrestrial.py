# train_terrestrial.py — train leak-free CLIP+MFA on the merged
# "Terrestrial Mammal Dataset" (Xu et al. 2026 protocol reproduction).
#
# This run uses:
#   - the multi-scale image branch (global [CLS] + local patch pooling),
#   - the BNNeck fix (triplet loss on the pre-BN feature),
#   - partial backbone fine-tuning: the last UNFREEZE_LAST_N_BLOCKS CLIP
#     visual blocks are unfrozen and trained at LR * BACKBONE_LR_MULT.
#
# The dataset (data/terrestrial_dataset.py) already returns NEUTRAL, leak-free
# prompts. After training, evaluate with scripts/eval_terrestrial.py.
#
# Run:
#   python scripts/train_terrestrial.py
#   python scripts/train_terrestrial.py --batch-size 32     # if 16GB VRAM OOMs at 64

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import argparse

import torch
import clip
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP,
    LAMBDA_IA, CHECKPOINT_DIR, BACKBONE_LR_MULT, UNFREEZE_LAST_N_BLOCKS,
)
from data.terrestrial_dataset import load_terrestrial
from models.mfa     import MetaFeatureAdapter
from models.losses  import ReIDLoss, CrossModalContrastiveLoss
from utils.visualizer import TrainingVisualizer


# Run name carries the block count so each unfreeze depth saves a distinct
# checkpoint (e.g. terrestrial_mfa_ft4_epoch60.pth) — no overwrites.
RUN_NAME = f"terrestrial_mfa_ft{UNFREEZE_LAST_N_BLOCKS}"


def train(batch_size: int):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    # load_terrestrial is deterministic (TM_SPLIT_SEED) — the train/test split
    # built here is identical to the one eval_terrestrial.py will reconstruct.
    train_ds, _val_ds, _test_ds = load_terrestrial(verbose=True)
    loader = DataLoader(train_ds, batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    num_classes = train_ds.num_classes
    print(f"Classes (train entities): {num_classes} | "
          f"Train images: {len(train_ds)} | Batch size: {batch_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MetaFeatureAdapter(num_classes=num_classes).to(device)

    # Two parameter groups: fresh adapters at LR, unfrozen CLIP blocks at a
    # lower LR so fine-tuning does not wash out the pretrained features.
    adapter_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("clip_model.")
         else adapter_params).append(p)
    trainable = adapter_params + backbone_params

    n_ad = sum(p.numel() for p in adapter_params)
    n_bb = sum(p.numel() for p in backbone_params)
    print(f"Trainable: {n_ad:,} adapter + {n_bb:,} backbone "
          f"({UNFREEZE_LAST_N_BLOCKS} CLIP blocks) = {n_ad + n_bb:,} total")

    groups = [{"params": adapter_params, "lr": LR}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": LR * BACKBONE_LR_MULT})
    optimizer = Adam(groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    reid_loss   = ReIDLoss(num_classes=num_classes).to(device)
    cross_modal = CrossModalContrastiveLoss().to(device)

    viz = TrainingVisualizer(num_epochs=NUM_EPOCHS, run_name=RUN_NAME)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total, n_batches = 0.0, 0
        info = {}

        for images, prompts, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            tokens = clip.tokenize(list(prompts), truncate=True).to(device)

            optimizer.zero_grad()
            logits, feat, i_meta, text_emb, _ = model(images, tokens, return_all=True)

            # BNNeck: ID loss on post-BN logits, triplet on the PRE-BN feature.
            loss_reid, info = reid_loss(logits, i_meta, labels)
            loss_ia         = cross_modal(text_emb, feat)
            loss            = loss_reid + LAMBDA_IA * loss_ia

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()

            total     += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total / max(n_batches, 1)
        lrs = scheduler.get_last_lr()

        if epoch % 10 == 0 or epoch == 1:
            lr_str = " / ".join(f"{x:.2e}" for x in lrs)
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr_str}")

        viz.update(epoch, loss=avg_loss, lr=lrs[0],
                   loss_id=info.get("loss_id"),
                   loss_tri=info.get("loss_tri"),
                   loss_meta=info.get("loss_meta"))
        viz.save()

        if epoch % 20 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_epoch{epoch}.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "num_classes": num_classes,
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

    final_path = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final weights → {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="lower to 32 if 16GB VRAM OOMs with unfrozen blocks")
    args = parser.parse_args()
    train(args.batch_size)
