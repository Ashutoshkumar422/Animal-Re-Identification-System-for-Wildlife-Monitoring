# eval_terrestrial.py — closed-set evaluation on the merged
# "Terrestrial Mammal Dataset", reproducing Xu et al. 2026's protocol.
#
# Protocol (Xu et al. Section 3.2): closed-set, Query = Gallery. Every test
# image is ranked against all OTHER test images (leave-one-out — self is
# excluded from its own gallery). Metrics: mAP, Rank-1/5/10.
#
# Produces three evaluations:
#   - Overall      (all test images)   -> our row for the journal's Table 4
#   - Tiger subset (test tigers only)  -> our row for Table 5 (tiger)
#   - Leopard subset                   -> our row for Table 5 (leopard)
#
# Each block reports three variants:
#   (a)    CLIP visual-only baseline
#   (d)    MFA with the neutral, leak-free prompt
#   (d+RR) (d) + k-reciprocal re-ranking  <- headline number for the paper
#
# Run:
#   python scripts/eval_terrestrial.py --checkpoint terrestrial_mfa_epoch60.pth

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader

from config import DEVICE, CHECKPOINT_DIR, RESULTS_DIR
from data.terrestrial_dataset import load_terrestrial
from models.mfa import MetaFeatureAdapter
from eval.re_ranking import re_ranking


# Xu et al. 2026 (ARNet) published numbers, for an at-a-glance comparison.
# Table 4 is the merged "all categories" set; Table 5 is the species subsets.
XU_ARNET = {
    "overall": {"mAP": 58.24, "R1": 86.50, "R5": 93.00, "R10": None},
    "tiger":   {"mAP": 77.59, "R1": 98.72, "R5": 99.43, "R10": None},
    "leopard": {"mAP": 29.92, "R1": 68.30, "R5": 83.40, "R10": None},
}


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract(model, loader, device, use_prompt):
    """
    use_prompt=True  -> MFA with the dataset's neutral (leak-free) prompt.
    use_prompt=False -> CLIP visual-only baseline (no text branch).
    Returns (features [N,D], labels [N], species [N]).
    """
    feats, labels, species = [], [], []
    with torch.no_grad():
        for images, prompts, lbls, sp in loader:
            images = images.to(device)
            if use_prompt:
                tokens = clip.tokenize(list(prompts), truncate=True).to(device)
                _, f = model(images, tokens)
            else:
                f = model.encode_image_raw(images)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
            labels.extend(int(l) for l in lbls)
            species.extend(list(sp))
    return np.vstack(feats), np.array(labels), np.array(species)


# ── Closed-set metric (Query = Gallery, self excluded) ────────────────────────

def compute_closed_set_metrics(labels, feats=None, dist=None, ranks=(1, 5, 10)):
    """
    Query = Gallery closed-set retrieval. Each row is a query; it is ranked
    against every other row (its own position is masked out).

    Pass either `feats` (cosine distance is used) or a precomputed `dist`
    matrix (e.g. from re-ranking). Returns (mAP, {k: CMC-rank-k}, n_valid).
    """
    if dist is None:
        dist = 1.0 - feats @ feats.T
    dist = np.array(dist, dtype=np.float32, copy=True)
    np.fill_diagonal(dist, np.inf)              # leave-one-out: exclude self

    aps, rank_hits, valid = [], {k: 0 for k in ranks}, 0
    for i in range(len(labels)):
        order   = np.argsort(dist[i])
        matched = (labels[order] == labels[i])
        n_rel   = int(matched.sum())
        if n_rel == 0:                          # no other image of this identity
            continue
        valid += 1

        # Average precision: mean precision at each true-match rank.
        hit_pos    = np.where(matched)[0]                       # 0-indexed
        precisions = np.arange(1, n_rel + 1) / (hit_pos + 1.0)
        aps.append(precisions.mean())

        # CMC rank-k.
        for k in ranks:
            if matched[:k].any():
                rank_hits[k] += 1

    mAP = float(np.mean(aps)) * 100.0 if aps else 0.0
    cmc = {k: rank_hits[k] / valid * 100.0 for k in ranks} if valid else \
          {k: 0.0 for k in ranks}
    return mAP, cmc, valid


# ── Reporting ─────────────────────────────────────────────────────────────────

def _eval_block(name, key, feats_v, feats_p, labels):
    """Evaluate one (overall / species) block and print a comparison table."""
    mAP_v, cmc_v, n = compute_closed_set_metrics(labels, feats=feats_v)
    mAP_p, cmc_p, _ = compute_closed_set_metrics(labels, feats=feats_p)
    rr_dist         = re_ranking(feats_p)
    mAP_r, cmc_r, _ = compute_closed_set_metrics(labels, dist=rr_dist)
    xu = XU_ARNET[key]

    def _row(tag, mAP, cmc):
        print(f"{tag:<36}{mAP:>8.2f} {cmc[1]:>8.2f} {cmc[5]:>8.2f} {cmc[10]:>8.2f}")

    def _fmt(x):
        return f"{x:>8.2f}" if x is not None else f"{'—':>8}"

    print(f"\n=== {name}  ({n} query/gallery images) ===")
    print(f"{'Variant':<36}{'mAP':>8} {'Rank-1':>8} {'Rank-5':>8} {'Rank-10':>8}")
    print("-" * 76)
    _row("(a) CLIP visual-only",            mAP_v, cmc_v)
    _row("(d) MFA neutral (leak-free)",     mAP_p, cmc_p)
    _row("(d+RR) + k-reciprocal re-rank *", mAP_r, cmc_r)
    print(f"{'    reference: Xu et al. ARNet':<36}"
          f"{_fmt(xu['mAP'])} {_fmt(xu['R1'])} {_fmt(xu['R5'])} {_fmt(xu['R10'])}")
    if xu["mAP"] is not None:
        print(f"    Δ headline (d+RR) vs ARNet:  mAP {mAP_r - xu['mAP']:+.2f}"
              f"   R1 {cmc_r[1] - xu['R1']:+.2f}"
              f"   R5 {cmc_r[5] - xu['R5']:+.2f}")

    return {
        "n_images": int(n),
        "visual_only":        {"mAP": mAP_v, "R1": cmc_v[1], "R5": cmc_v[5], "R10": cmc_v[10]},
        "mfa_neutral":        {"mAP": mAP_p, "R1": cmc_p[1], "R5": cmc_p[5], "R10": cmc_p[10]},
        "mfa_neutral_rerank": {"mAP": mAP_r, "R1": cmc_r[1], "R5": cmc_r[5], "R10": cmc_r[10]},
        "xu_arnet": xu,
    }


def run(checkpoint: str, batch_size: int):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # ── Reconstruct the (seeded, identical) test split ───────────────────────
    _, _, test_ds = load_terrestrial(verbose=True)
    loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        n_train, state = ckpt.get("num_classes", 160), ckpt["model_state"]
    else:
        n_train, state = 160, ckpt           # bare state_dict (e.g. *_final.pth)
    model = MetaFeatureAdapter(num_classes=n_train).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {ckpt_path}  (trained on {n_train} entities)")

    # ── Extract features once (visual-only + neutral-prompt) ─────────────────
    feats_v, labels, species = _extract(model, loader, device, use_prompt=False)
    feats_p, _,      _       = _extract(model, loader, device, use_prompt=True)

    is_tiger   = species == "tiger"
    is_leopard = species == "leopard"

    results = {}
    results["overall"] = _eval_block(
        "Overall — journal Table 4 (merged test set)",
        "overall", feats_v, feats_p, labels)
    results["tiger"] = _eval_block(
        "Tiger subset — journal Table 5",
        "tiger", feats_v[is_tiger], feats_p[is_tiger], labels[is_tiger])
    results["leopard"] = _eval_block(
        "Leopard subset — journal Table 5",
        "leopard", feats_v[is_leopard], feats_p[is_leopard], labels[is_leopard])

    print("\n* = headline number for the paper (leak-free MFA + standard "
          "k-reciprocal re-ranking). Rank-10 shown for completeness; the "
          "journal Tables 4-5 report only mAP/R1/R5.")

    out_path = os.path.join(RESULTS_DIR, "terrestrial_eval.json")
    with open(out_path, "w") as f:
        json.dump({"checkpoint": checkpoint, "results": results}, f, indent=2)
    print(f"\nSaved numbers → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="terrestrial_mfa_epoch60.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run(args.checkpoint, args.batch_size)
