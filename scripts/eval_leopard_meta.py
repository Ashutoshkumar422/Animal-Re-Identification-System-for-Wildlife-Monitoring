# eval_leopard_meta.py — Evaluate any leopard checkpoint with three prompt variants:
#   (a) Visual-only baseline
#   (d) Neutral prompt (Option B baseline — no metadata, no individual id)
#   (e) Auto-metadata prompt (Option A — per-image circadian/temp/viewpoint)
#
# By running this against multiple checkpoints you can decompose the gain:
#   - on neutral-trained ckpt  → does eval-time metadata help even untrained?
#   - on meta-trained ckpt     → does the full A-trained-A-evaluated pipeline win?
#
# Run:
#   python scripts/eval_leopard_meta.py --checkpoint leopard_mfa_neutral_epoch60.pth
#   python scripts/eval_leopard_meta.py --checkpoint leopard_mfa_meta_epoch60.pth

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

from config import DEVICE, CHECKPOINT_DIR, LEOPARD_ROOT
from data.leopard_dataset import load_leopard
from models.mfa import MetaFeatureAdapter
from eval.metrics import compute_map_cmc1, compute_rank_k


META_PATH      = os.path.join(LEOPARD_ROOT, "metadata_auto.json")
NEUTRAL_PROMPT = (
    "A photo of a leopard in cool temperature, "
    "with face direction front, captured during the day."
)


def _build_meta_prompt(meta: dict) -> str:
    temp   = meta.get("temperature", "cool")
    orient = meta.get("orientation", "front")
    circ   = meta.get("circadian",   "day")
    return (
        f"A photo of a leopard in {temp} temperature, "
        f"with face direction {orient}, captured during the {circ}."
    )


def _extract(model, loader, device, sample_keys, meta_cache, mode: str):
    """
    mode = 'visual' | 'neutral' | 'meta'
      visual  → no text branch
      neutral → constant NEUTRAL_PROMPT for every sample
      meta    → per-sample auto-metadata prompt (lookup by sample key)
    """
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


def _eval(model, g_loader, q_loader, device, g_keys, q_keys, meta, mode):
    g_f, gl = _extract(model, g_loader, device, g_keys, meta, mode)
    q_f, ql = _extract(model, q_loader, device, q_keys, meta, mode)
    mAP, cmc1 = compute_map_cmc1(q_f, ql, g_f, gl)
    rank5     = compute_rank_k(q_f, ql, g_f, gl, k=5)
    return mAP, cmc1, rank5


def run(checkpoint: str):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(META_PATH):
        raise FileNotFoundError(
            f"{META_PATH} not found — run `python scripts/extract_metadata.py --dataset leopard` first."
        )
    with open(META_PATH) as f:
        meta_cache = json.load(f)
    print(f"Metadata cache: {len(meta_cache)} entries")

    _, gallery_set, query_set = load_leopard(root=LEOPARD_ROOT)

    # IMPORTANT: shuffle=False so cursor-based key lookup stays aligned with samples.
    g_loader = DataLoader(gallery_set, batch_size=64, shuffle=False, num_workers=4)
    q_loader = DataLoader(query_set,   batch_size=64, shuffle=False, num_workers=4)
    g_keys   = [s[0] for s in gallery_set.samples]
    q_keys   = [s[0] for s in query_set.samples]
    g_cov    = sum(1 for k in g_keys if k in meta_cache) / len(g_keys)
    q_cov    = sum(1 for k in q_keys if k in meta_cache) / len(q_keys)
    print(f"Coverage — gallery: {g_cov*100:.1f}%   query: {q_cov*100:.1f}%")

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, checkpoint), map_location=device)
    n_train = ckpt.get("num_classes")
    if n_train is None:
        raise ValueError(f"Checkpoint missing 'num_classes' key.")
    model = MetaFeatureAdapter(num_classes=n_train).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    print(f"Loaded: {checkpoint}  (trained on {n_train} identities)\n")

    rows = []
    for name, mode in [
        ("(a) CLIP visual-only baseline",  "visual"),
        ("(d) MFA + neutral prompt",       "neutral"),
        ("(e) MFA + auto-metadata prompt", "meta"),
    ]:
        print(f"Evaluating {name} ...")
        mAP, cmc1, r5 = _eval(model, g_loader, q_loader, device,
                              g_keys, q_keys, meta_cache, mode)
        rows.append((name, mAP, cmc1, r5))

    print("\n=== Leopard — neutral vs auto-metadata prompts at eval ===")
    print(f"{'Variant':<40} {'mAP':>8} {'Rank-1':>8} {'Rank-5':>8}")
    print("-" * 68)
    for name, mAP, cmc1, r5 in rows:
        print(f"{name:<40} {mAP:>7.2f}% {cmc1:>7.2f}% {r5:>7.2f}%")

    delta_mAP = rows[2][1] - rows[1][1]
    print(f"\nΔ from auto-metadata over neutral prompt: {delta_mAP:+.2f}% mAP")
    print("Positive = real metadata fusion adds signal beyond the neutral prompt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="filename inside CHECKPOINT_DIR")
    args = parser.parse_args()
    run(args.checkpoint)
