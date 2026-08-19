# eval_leopard.py — Evaluate MFA on LeopardID 2022 (disjoint-identity protocol)
#
# Reports mAP, CMC-1 (Rank-1), Rank-5 for both:
#   (a) CLIP visual-only baseline   — uses model.encode_image_raw
#   (b) CLIP + MFA (full pipeline)  — uses the trained adapters + GCA fusion
#
# Mirrors the table format in Xu et al. 2026 so results are directly comparable.
# Test identities are disjoint from training identities (open-set re-ID style).
#
# Run:
#   python scripts/eval_leopard.py --checkpoint leopard_mfa_epoch60.pth

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader

from config import DEVICE, CHECKPOINT_DIR, LEOPARD_ROOT
from data.leopard_dataset import load_leopard
from models.mfa import MetaFeatureAdapter
from eval.metrics import (
    extract_features, compute_map_cmc1, compute_rank_k, plot_tsne,
)
from utils.visualizer import EvalVisualizer


RUN_NAME = "leopard_mfa"


def _visual_only_features(model, loader, device):
    """Encode batch images with the frozen CLIP backbone only (no MFA head)."""
    feats, labels = [], []
    with torch.no_grad():
        for images, _, lbls, _ in loader:
            f = model.encode_image_raw(images.to(device))
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
            labels.extend(lbls.tolist())
    return np.vstack(feats), np.array(labels)


def run_evaluation(checkpoint: str = "leopard_mfa_epoch60.pth"):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # ── Load disjoint-identity test split ────────────────────────────────────
    _, gallery_set, query_set = load_leopard(root=LEOPARD_ROOT)
    gallery_loader = DataLoader(gallery_set, batch_size=64,
                                shuffle=False, num_workers=4)
    query_loader   = DataLoader(query_set, batch_size=64,
                                shuffle=False, num_workers=4)

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    ckpt      = torch.load(ckpt_path, map_location=device)
    # The classifier head was sized for the *training* identities. At eval time
    # the gallery/query identities are disjoint, but the classifier's output
    # logits are not used for retrieval — only the BN bottleneck `feat` is.
    train_num_classes = ckpt.get("num_classes")
    if train_num_classes is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} has no 'num_classes' key. "
            "Re-train using train_leopard.py which saves it."
        )
    model = MetaFeatureAdapter(num_classes=train_num_classes).to(device)
    model.load_state_dict(
        ckpt["model_state"] if "model_state" in ckpt else ckpt
    )
    model.eval()
    print(f"Loaded: {ckpt_path}  (trained on {train_num_classes} identities)")

    # ── (a) CLIP visual-only baseline ────────────────────────────────────────
    print("\n[baseline] Extracting CLIP visual-only features...")
    g_v, gl_v = _visual_only_features(model, gallery_loader, device)
    q_v, ql_v = _visual_only_features(model, query_loader,   device)
    map_v,  cmc1_v  = compute_map_cmc1(q_v, ql_v, g_v, gl_v)
    rank5_v         = compute_rank_k(q_v, ql_v, g_v, gl_v, k=5)

    # ── (b) MFA full pipeline ────────────────────────────────────────────────
    print("[mfa] Extracting MFA-fused features...")
    g_f, gl_f, _ = extract_features(model, gallery_loader, device, use_metadata=True)
    q_f, ql_f, _ = extract_features(model, query_loader,   device, use_metadata=True)
    map_m, cmc1_m  = compute_map_cmc1(q_f, ql_f, g_f, gl_f)
    rank5_m        = compute_rank_k(q_f, ql_f, g_f, gl_f, k=5)

    # ── Plots ────────────────────────────────────────────────────────────────
    eviz = EvalVisualizer(run_name=RUN_NAME)
    eviz.plot_cmc_curve(q_f, ql_f, g_f, gl_f)
    eviz.plot_distance_distribution(q_f, ql_f, g_f, gl_f)
    eviz.save()

    print("\nGenerating t-SNE plot...")
    plot_tsne(g_f, gl_f, "leopard", save_name=f"tsne_leopard_{RUN_NAME}.png")

    # ── Results table (matches Xu et al. 2026 column layout) ─────────────────
    print("\n=== LeopardID 2022 — disjoint-identity test ===")
    print(f"Gallery: {len(gl_v)} imgs | Query: {len(ql_v)} imgs | "
          f"Test identities: {len(set(gl_v))}")
    print()
    print(f"{'Method':<28} {'mAP':>8} {'Rank-1':>8} {'Rank-5':>8}")
    print("-" * 56)
    print(f"{'CLIP visual-only (baseline)':<28} "
          f"{map_v:>7.2f}% {cmc1_v:>7.2f}% {rank5_v:>7.2f}%")
    print(f"{'CLIP + MFA (ours)':<28} "
          f"{map_m:>7.2f}% {cmc1_m:>7.2f}% {rank5_m:>7.2f}%")
    print()
    print(f"Δ vs baseline: "
          f"mAP {map_m - map_v:+.2f}%  "
          f"Rank-1 {cmc1_m - cmc1_v:+.2f}%  "
          f"Rank-5 {rank5_m - rank5_v:+.2f}%")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="leopard_mfa_epoch60.pth",
                        help="filename inside CHECKPOINT_DIR")
    args = parser.parse_args()
    run_evaluation(checkpoint=args.checkpoint)
