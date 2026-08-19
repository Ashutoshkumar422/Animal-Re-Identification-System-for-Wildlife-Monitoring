# eval_atrw_ablation.py — Same prompt-leakage ablation as eval_leopard_ablation.py,
# applied to ATRW under both reported protocols (full 107-id and top-47 closed-set).
#
# Both protocols are non-disjoint at the identity level (test ids overlap with
# train ids). The leak mechanism still applies: gallery and query images of the
# same test identity always share a prompt because the prompt embeds the label.
#
# Run:
#   python scripts/eval_atrw_ablation.py --checkpoint mfa_epoch60.pth

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader

from config import DEVICE, CHECKPOINT_DIR, ATRW_ROOT
from data.atrw_dataset import (
    load_atrw, ATRWDataset, _parse_train_csv, _make_splits,
)
from models.mfa import MetaFeatureAdapter
from eval.metrics import compute_map_cmc1, compute_rank_k


# Same template the ATRW pipeline uses at train time. Defined locally so this
# script doesn't depend on whether data/atrw_dataset.py exports it as a constant.
PROMPT_ATRW = (
    "A photo of a tiger individual {label} in cool temperature, "
    "with face direction front, captured during the day."
)
NEUTRAL_PROMPT = (
    "A photo of a tiger in cool temperature, "
    "with face direction front, captured during the day."
)


def _extract(model, loader, device, prompt_fn):
    """prompt_fn(label_int)->str, or None for visual-only encoder."""
    feats, labels = [], []
    with torch.no_grad():
        for images, _, lbls, _ in loader:
            images = images.to(device)
            if prompt_fn is None:
                f = model.encode_image_raw(images).float()
            else:
                prompts = [prompt_fn(int(l)) for l in lbls]
                tokens  = clip.tokenize(prompts, truncate=True).to(device)
                _, f    = model(images, tokens)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu().numpy())
            labels.extend(lbls.tolist())
    return np.vstack(feats), np.array(labels)


def _eval_variant(model, g_loader, q_loader, device, prompt_fn):
    g_f, gl = _extract(model, g_loader, device, prompt_fn)
    q_f, ql = _extract(model, q_loader, device, prompt_fn)
    mAP, cmc1 = compute_map_cmc1(q_f, ql, g_f, gl)
    rank5     = compute_rank_k(q_f, ql, g_f, gl, k=5)
    return mAP, cmc1, rank5


def _run_protocol(model, g_loader, q_loader, device, header):
    variants = [
        ("(a) CLIP visual-only baseline",        None),
        ("(b) MFA, label-in-prompt (LEAK)",      lambda l: PROMPT_ATRW.format(label=l)),
        ("(c) MFA, constant label = 0",          lambda l: PROMPT_ATRW.format(label=0)),
        ("(d) MFA, no individual reference",     lambda l: NEUTRAL_PROMPT),
    ]

    rows = []
    for name, fn in variants:
        print(f"  Evaluating {name} ...")
        mAP, cmc1, r5 = _eval_variant(model, g_loader, q_loader, device, fn)
        rows.append((name, mAP, cmc1, r5))

    print(f"\n=== {header} ===")
    print(f"{'Variant':<40} {'mAP':>8} {'Rank-1':>8} {'Rank-5':>8}")
    print("-" * 68)
    for name, mAP, cmc1, r5 in rows:
        print(f"{name:<40} {mAP:>7.2f}% {cmc1:>7.2f}% {r5:>7.2f}%")

    leak_vs_const   = rows[1][1] - rows[2][1]
    leak_vs_neutral = rows[1][1] - rows[3][1]
    real_vs_visual  = rows[3][1] - rows[0][1]
    print(f"  Leak magnitude (b vs c, constant label):  {leak_vs_const:+.2f}% mAP")
    print(f"  Leak magnitude (b vs d, no-id prompt):    {leak_vs_neutral:+.2f}% mAP")
    print(f"  Honest MFA gain over visual-only (d − a): {real_vs_visual:+.2f}% mAP")


def run(checkpoint: str):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # ── Load full 107-id splits and build the top-47 subset ──────────────────
    _, gallery_set_full, query_set_full = load_atrw(root=ATRW_ROOT)

    id_to_files = _parse_train_csv(os.path.join(ATRW_ROOT, "reid_list_train.csv"))
    _, gallery_s, query_s = _make_splits(id_to_files)
    gallery_counts = Counter(lbl for _, lbl in gallery_s)
    top47 = {lbl for lbl, _ in gallery_counts.most_common(47)}
    gallery_47 = [(f, l) for f, l in gallery_s if l in top47]
    query_47   = [(f, l) for f, l in query_s   if l in top47]
    img_dir    = os.path.join(ATRW_ROOT, "train")
    gallery_set_47 = ATRWDataset(gallery_47, img_dir, split="gallery")
    query_set_47   = ATRWDataset(query_47,   img_dir, split="query")

    # ── Loaders ──────────────────────────────────────────────────────────────
    g_full = DataLoader(gallery_set_full, batch_size=64, shuffle=False, num_workers=4)
    q_full = DataLoader(query_set_full,   batch_size=64, shuffle=False, num_workers=4)
    g_47   = DataLoader(gallery_set_47,   batch_size=64, shuffle=False, num_workers=4)
    q_47   = DataLoader(query_set_47,     batch_size=64, shuffle=False, num_workers=4)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    ckpt      = torch.load(ckpt_path, map_location=device)
    n_train   = ckpt.get("num_classes", 107)
    model = MetaFeatureAdapter(num_classes=n_train).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    print(f"Loaded: {ckpt_path}  (trained on {n_train} identities)")
    print(f"Full 107-id: gallery={len(gallery_set_full)}  query={len(query_set_full)}")
    print(f"Top-47:      gallery={len(gallery_set_47)}    query={len(query_set_47)}\n")

    print("[Protocol 1] Full 107-identity split")
    _run_protocol(model, g_full, q_full, device,
                  "ATRW Protocol 1 — Full 107-id (non-disjoint)")
    print()
    print("[Protocol 2] Top-47 most-populated identities (closed-set)")
    _run_protocol(model, g_47, q_47, device,
                  "ATRW Protocol 2 — Top-47 closed-set (non-disjoint)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="mfa_epoch60.pth")
    args = parser.parse_args()
    run(args.checkpoint)
