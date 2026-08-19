# eval_atrw_meta.py — Evaluate any ATRW checkpoint with three prompt variants
# under both reported protocols:
#   (a) Visual-only baseline
#   (d) Neutral prompt (Option B baseline)
#   (e) Auto-metadata prompt (per-image circadian/temp; orientation always "front")
#
# Run:
#   python scripts/eval_atrw_meta.py --checkpoint atrw_mfa_neutral_epoch60.pth
#   python scripts/eval_atrw_meta.py --checkpoint atrw_mfa_meta_epoch60.pth

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
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


META_PATH      = os.path.join(ATRW_ROOT, "metadata_auto.json")
NEUTRAL_PROMPT = (
    "A photo of a tiger in cool temperature, "
    "with face direction front, captured during the day."
)


def _build_meta_prompt(meta: dict) -> str:
    temp   = meta.get("temperature", "cool")
    orient = meta.get("orientation", "front")
    circ   = meta.get("circadian",   "day")
    return (
        f"A photo of a tiger in {temp} temperature, "
        f"with face direction {orient}, captured during the {circ}."
    )


def _extract(model, loader, device, sample_keys, meta_cache, mode: str):
    feats, labels = [], []
    cursor = 0
    with torch.no_grad():
        for images, _, lbls, _ in loader:
            images = images.to(device)
            if mode == "visual":
                f = model.encode_image_raw(images).float()
            else:
                if mode == "neutral":
                    prompts = [NEUTRAL_PROMPT] * len(lbls)
                else:  # 'meta'
                    keys = sample_keys[cursor : cursor + len(lbls)]
                    prompts = [_build_meta_prompt(meta_cache.get(k, {})) for k in keys]
                tokens = clip.tokenize(prompts, truncate=True).to(device)
                _, f   = model(images, tokens)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu().numpy())
            labels.extend(lbls.tolist())
            cursor += len(lbls)
    return np.vstack(feats), np.array(labels)


def _run_protocol(model, g_loader, q_loader, device,
                  g_keys, q_keys, meta_cache, header):
    rows = []
    for name, mode in [
        ("(a) CLIP visual-only baseline",  "visual"),
        ("(d) MFA + neutral prompt",       "neutral"),
        ("(e) MFA + auto-metadata prompt", "meta"),
    ]:
        print(f"  Evaluating {name} ...")
        g_f, gl = _extract(model, g_loader, device, g_keys, meta_cache, mode)
        q_f, ql = _extract(model, q_loader, device, q_keys, meta_cache, mode)
        mAP, cmc1 = compute_map_cmc1(q_f, ql, g_f, gl)
        rank5     = compute_rank_k(q_f, ql, g_f, gl, k=5)
        rows.append((name, mAP, cmc1, rank5))

    print(f"\n=== {header} ===")
    print(f"{'Variant':<40} {'mAP':>8} {'Rank-1':>8} {'Rank-5':>8}")
    print("-" * 68)
    for name, mAP, cmc1, r5 in rows:
        print(f"{name:<40} {mAP:>7.2f}% {cmc1:>7.2f}% {r5:>7.2f}%")
    delta = rows[2][1] - rows[1][1]
    print(f"  Δ from auto-metadata over neutral prompt: {delta:+.2f}% mAP")


def run(checkpoint: str):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(META_PATH):
        raise FileNotFoundError(
            f"{META_PATH} not found — run `python scripts/extract_metadata.py --dataset atrw` first."
        )
    with open(META_PATH) as f:
        meta_cache = json.load(f)
    print(f"Metadata cache: {len(meta_cache)} entries")

    # ATRW dataset stores full relative paths in samples[i][0]; the metadata
    # cache is keyed by bare filename. Normalize via basename so the lookup hits.
    # Full 107-id splits (shuffle=False so cursor-based key lookup matches sample order).
    _, gallery_full, query_full = load_atrw(root=ATRW_ROOT)
    g_keys_full = [os.path.basename(s[0]) for s in gallery_full.samples]
    q_keys_full = [os.path.basename(s[0]) for s in query_full.samples]
    g_full = DataLoader(gallery_full, batch_size=64, shuffle=False, num_workers=4)
    q_full = DataLoader(query_full,   batch_size=64, shuffle=False, num_workers=4)

    # Top-47 closed-set (same identity-filtering as eval47.py).
    id_to_files = _parse_train_csv(os.path.join(ATRW_ROOT, "reid_list_train.csv"))
    _, gallery_s, query_s = _make_splits(id_to_files)
    counts   = Counter(lbl for _, lbl in gallery_s)
    top47    = {lbl for lbl, _ in counts.most_common(47)}
    gal_47   = [(f, l) for f, l in gallery_s if l in top47]
    qry_47   = [(f, l) for f, l in query_s   if l in top47]
    img_dir  = os.path.join(ATRW_ROOT, "train")
    gallery_47 = ATRWDataset(gal_47, img_dir, split="gallery")
    query_47   = ATRWDataset(qry_47, img_dir, split="query")
    g_keys_47  = [os.path.basename(s[0]) for s in gallery_47.samples]
    q_keys_47  = [os.path.basename(s[0]) for s in query_47.samples]
    g_47 = DataLoader(gallery_47, batch_size=64, shuffle=False, num_workers=4)
    q_47 = DataLoader(query_47,   batch_size=64, shuffle=False, num_workers=4)

    # Sanity check: cache lookup must hit for at least 99% of samples.
    g_hit = sum(1 for k in g_keys_full if k in meta_cache)
    if g_hit < 0.99 * len(g_keys_full):
        raise RuntimeError(
            f"Cache coverage is only {g_hit}/{len(g_keys_full)} on gallery — "
            "key format mismatch. Run diag_atrw_meta_cache.py to debug."
        )
    print(f"Cache coverage OK: gallery {g_hit}/{len(g_keys_full)}")

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, checkpoint), map_location=device)
    n_train = ckpt.get("num_classes", 107)
    model = MetaFeatureAdapter(num_classes=n_train).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    print(f"Loaded: {checkpoint}  (trained on {n_train} identities)\n")

    print("[Protocol 1] Full 107-identity split")
    _run_protocol(model, g_full, q_full, device, g_keys_full, q_keys_full,
                  meta_cache, "ATRW Protocol 1 — Full 107-id (non-disjoint)")
    print()
    print("[Protocol 2] Top-47 closed-set")
    _run_protocol(model, g_47, q_47, device, g_keys_47, q_keys_47,
                  meta_cache, "ATRW Protocol 2 — Top-47 closed-set (non-disjoint)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="filename inside CHECKPOINT_DIR")
    args = parser.parse_args()
    run(args.checkpoint)
