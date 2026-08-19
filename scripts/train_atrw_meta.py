# train_atrw_meta.py — Option A for ATRW.
#
# ATRW has no ground-truth viewpoint annotations, so face direction defaults
# to "front" for every image. The only real signals injected here are:
#   - circadian (day/night from brightness)
#   - temperature (warm/cool/cold from RGB)
# Effectively a weaker version of Option A — useful as a control for whether
# *any* per-image variation in the prompt helps on ATRW, even without viewpoint.
#
# Prereq:
#   python scripts/extract_metadata.py --dataset atrw       # writes data/atrw/metadata_auto.json
#
# Run:
#   python scripts/train_atrw_meta.py

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP,
    LAMBDA_IA, CHECKPOINT_DIR, ATRW_ROOT,
)
from data.atrw_dataset import load_atrw
from models.mfa     import MetaFeatureAdapter
from models.losses  import ReIDLoss, CrossModalContrastiveLoss
from utils.visualizer import TrainingVisualizer


RUN_NAME    = "atrw_mfa_meta"
# Keep BATCH_SIZE = config default (64) so (e) is directly comparable to (d)
# from train_atrw_neutral.py. ATRW's small size (1090 train) also makes BS=128
# only ~9 batches/epoch — likely under-trains the contrastive loss.
META_PATH   = os.path.join(ATRW_ROOT, "metadata_auto.json")

NEUTRAL_FALLBACK = ("cool", "front", "day")


def _build_prompt(meta: dict) -> str:
    temp = meta.get("temperature", NEUTRAL_FALLBACK[0])
    orient = meta.get("orientation", NEUTRAL_FALLBACK[1])
    circ = meta.get("circadian", NEUTRAL_FALLBACK[2])
    return (
        f"A photo of a tiger in {temp} temperature, "
        f"with face direction {orient}, captured during the {circ}."
    )


class _MetaPromptDataset(Dataset):
    """
    Wraps the base ATRW dataset. Looks up per-image metadata in the cache keyed
    by the sample's filename and substitutes a real-metadata prompt.
    """
    def __init__(self, base_ds, meta_cache):
        self.base = base_ds
        self.meta = meta_cache
        self.num_classes = base_ds.num_classes
        # ATRW dataset's samples store the FULL relative path (e.g.
        # "./data/atrw/train/003107.jpg") in s[0], but the metadata cache is
        # keyed by bare filename ("003107.jpg"). Normalize via basename so the
        # lookup actually hits.
        self._sample_keys = [os.path.basename(s[0]) for s in base_ds.samples]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _orig_prompt, label, sp = self.base[idx]
        key = self._sample_keys[idx]
        meta = self.meta.get(key, {})
        return img, _build_prompt(meta), label, sp


def train():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if not os.path.isfile(META_PATH):
        raise FileNotFoundError(
            f"{META_PATH} not found — run `python scripts/extract_metadata.py --dataset atrw` first."
        )
    with open(META_PATH) as f:
        meta_cache = json.load(f)
    print(f"Loaded metadata cache: {len(meta_cache)} entries from {META_PATH}")

    base_train, _, _ = load_atrw(root=ATRW_ROOT)
    train_set = _MetaPromptDataset(base_train, meta_cache)
    coverage = sum(1 for k in train_set._sample_keys if k in meta_cache)
    print(f"Metadata coverage on train set: {coverage}/{len(train_set)} samples "
          f"({100*coverage/len(train_set):.1f}%)")
    if coverage == 0:
        raise RuntimeError(
            "Metadata cache lookup is failing for every sample — keys don't match. "
            "Inspect a few sample keys vs cache keys to find the mismatch."
        )

    loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4, pin_memory=True)

    num_classes = train_set.num_classes
    print(f"Classes: {num_classes} | Train samples: {len(train_set)} | BS: {BATCH_SIZE}")

    model = MetaFeatureAdapter(num_classes=num_classes).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
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
