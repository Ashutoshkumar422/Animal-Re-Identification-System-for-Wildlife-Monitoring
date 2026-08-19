# eval_leopard_ablation.py — Quantify label-in-prompt leakage at eval time.
#
# Re-runs the same disjoint-identity test split through the same checkpoint
# under four prompt-construction variants:
#
#   (a) Visual-only baseline (no MFA, no text branch)
#   (b) MFA with label-in-prompt — what eval_leopard.py currently does. The
#       per-sample prompt embeds the ground-truth identity label, so gallery
#       and query images of the same test identity always share a prompt.
#   (c) MFA with constant label=0 — every sample gets the same prompt with a
#       fixed individual id. Removes per-identity leakage but keeps the
#       "individual N" template the model was trained with.
#   (d) MFA with no individual reference — every sample gets the same prompt
#       that mentions only species and (dummy) metadata. Cleanest leak-free run.
#
# If (b) is much higher than (c) and (d), the headline 72% mAP from
# eval_leopard.py was driven by the prompt acting as an identity oracle, not by
# genuine visual feature improvement from the adapters.
#
# Run:
#   python scripts/eval_leopard_ablation.py --checkpoint leopard_mfa_epoch60.pth

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
from data.leopard_dataset import load_leopard, PROMPT_LEOPARD
from models.mfa import MetaFeatureAdapter
from eval.metrics import compute_map_cmc1, compute_rank_k


NEUTRAL_PROMPT = (
    "A photo of a leopard in cool temperature, "
    "with face direction front, captured during the day."
)


def _extract(model, loader, device, prompt_fn):
    """
    prompt_fn(label_int) -> str  — used to build the per-sample prompt
    prompt_fn = None             — visual-only (no text branch, no adapters)
    """
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


def run(checkpoint: str):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    _, gallery_set, query_set = load_leopard(root=LEOPARD_ROOT)
    g_loader = DataLoader(gallery_set, batch_size=64, shuffle=False, num_workers=4)
    q_loader = DataLoader(query_set,   batch_size=64, shuffle=False, num_workers=4)

    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    ckpt      = torch.load(ckpt_path, map_location=device)
    n_train   = ckpt.get("num_classes")
    if n_train is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} has no 'num_classes' key. "
            "Re-train using train_leopard.py."
        )
    model = MetaFeatureAdapter(num_classes=n_train).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    print(f"Loaded: {ckpt_path}  (trained on {n_train} identities)")
    print(f"Test:  gallery={len(gallery_set)}  query={len(query_set)}\n")

    variants = [
        ("(a) CLIP visual-only baseline",        None),
        ("(b) MFA, label-in-prompt (LEAK)",      lambda l: PROMPT_LEOPARD.format(label=l)),
        ("(c) MFA, constant label = 0",          lambda l: PROMPT_LEOPARD.format(label=0)),
        ("(d) MFA, no individual reference",     lambda l: NEUTRAL_PROMPT),
    ]

    rows = []
    for name, fn in variants:
        print(f"Evaluating {name} ...")
        mAP, cmc1, r5 = _eval_variant(model, g_loader, q_loader, device, fn)
        rows.append((name, mAP, cmc1, r5))

    print("\n=== Ablation — prompt construction at eval time ===")
    print(f"{'Variant':<40} {'mAP':>8} {'Rank-1':>8} {'Rank-5':>8}")
    print("-" * 68)
    for name, mAP, cmc1, r5 in rows:
        print(f"{name:<40} {mAP:>7.2f}% {cmc1:>7.2f}% {r5:>7.2f}%")

    leak_vs_const   = rows[1][1] - rows[2][1]
    leak_vs_neutral = rows[1][1] - rows[3][1]
    print()
    print(f"Leak magnitude on mAP (b vs c, constant label):  {leak_vs_const:+.2f}%")
    print(f"Leak magnitude on mAP (b vs d, no-id prompt):    {leak_vs_neutral:+.2f}%")
    print()
    print("Interpretation:")
    print("  - If (b) ≫ (c),(d) and (c),(d) ≈ (a):  the 72% MFA headline was "
          "driven entirely by ground-truth identity leaking through the prompt.")
    print("  - If (c),(d) > (a) by a meaningful margin:  the visual adapter is "
          "doing real work even without identity-conditioned prompts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="leopard_mfa_epoch60.pth")
    args = parser.parse_args()
    run(args.checkpoint)
