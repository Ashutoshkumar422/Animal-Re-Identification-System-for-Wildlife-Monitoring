# train_leopard.py — Train MFA on the LeopardID 2022 dataset (disjoint protocol)
#
# Run:
#   python scripts/train_leopard.py
#
# This is a standalone counterpart to train.py — it does not depend on
# DATASET_NAME being "leopard", so you can keep ATRW as the default in config.py.

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
import clip
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP,
    LAMBDA_IA, CHECKPOINT_DIR, LEOPARD_ROOT,
)
from data.leopard_dataset import load_leopard
from models.mfa     import MetaFeatureAdapter
from models.losses  import ReIDLoss, CrossModalContrastiveLoss
from utils.visualizer import TrainingVisualizer


RUN_NAME = "leopard_mfa"


def train():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_set, _, _ = load_leopard(root=LEOPARD_ROOT)
    loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4, pin_memory=True)

    num_classes = train_set.num_classes
    print(f"Classes: {num_classes} | Train samples: {len(train_set)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MetaFeatureAdapter(num_classes=num_classes).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── Losses ────────────────────────────────────────────────────────────────
    reid_loss   = ReIDLoss(num_classes=num_classes).to(device)
    cross_modal = CrossModalContrastiveLoss().to(device)

    # ── Visualization ─────────────────────────────────────────────────────────
    viz = TrainingVisualizer(num_epochs=NUM_EPOCHS, run_name=RUN_NAME)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total, n_batches = 0.0, 0
        info = {}

        for images, prompts, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            tokens = clip.tokenize(list(prompts), truncate=True).to(device)

            optimizer.zero_grad()

            logits, feat, text_emb, _ = model(images, tokens, return_all=True)

            loss_reid, info = reid_loss(logits, feat, labels)
            loss_ia         = cross_modal(text_emb, feat)
            loss            = loss_reid + LAMBDA_IA * loss_ia

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()

            total     += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        viz.update(epoch, loss=avg_loss, lr=scheduler.get_last_lr()[0],
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
    train()
