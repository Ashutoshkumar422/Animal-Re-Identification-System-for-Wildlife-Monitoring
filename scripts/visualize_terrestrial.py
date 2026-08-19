# visualize_terrestrial.py — paper figures for the merged Terrestrial Mammal study.
#
# Produces three PNGs in results/:
#   fig_dataset_samples.png — tiger / leopard sample crops
#   fig_retrieval.png       — qualitative retrieval: query + top-5, green/red borders
#   fig_tsne.png            — t-SNE of the test embeddings (by species + per-identity)
#
# Needs the merged dataset + a trained checkpoint (default: terrestrial_mfa_ft4).
# Run on the GPU machine:
#   python scripts/visualize_terrestrial.py --checkpoint terrestrial_mfa_ft4_epoch60.pth

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from config import DEVICE, CHECKPOINT_DIR, RESULTS_DIR
from data.terrestrial_dataset import load_terrestrial
from models.mfa import MetaFeatureAdapter

SEED = 42
DISP = 170   # display thumbnail size (px)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_raw(path, bbox, size=DISP):
    """Load an image, crop by bbox if present, resize to a square thumbnail."""
    img = Image.open(path).convert("RGB")
    if bbox is not None:
        x, y, w, h = bbox
        x, y   = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
        if x2 > x and y2 > y:
            img = img.crop((x, y, x2, y2))
    return img.resize((size, size))


@torch.no_grad()
def _extract(model, test_ds, device):
    """Post-BN MFA features for every test image (order matches test_ds.samples)."""
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    feats, labels, species = [], [], []
    for images, prompts, lbls, sp in loader:
        images = images.to(device)
        tokens = clip.tokenize(list(prompts), truncate=True).to(device)
        _, f = model(images, tokens)
        feats.append(F.normalize(f.float(), dim=-1).cpu().numpy())
        labels.extend(int(l) for l in lbls)
        species.extend(list(sp))
    return np.vstack(feats), np.array(labels), np.array(species)


def _style(ax, color, lw=3):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(lw)


# ── Figure 1 — dataset samples ────────────────────────────────────────────────

def fig_dataset_samples(test_ds, path):
    rng = random.Random(SEED)
    by_sp = {"tiger": [], "leopard": []}
    for s in test_ds.samples:
        by_sp[s[3]].append(s)

    n_col = 6
    fig, axes = plt.subplots(2, n_col, figsize=(n_col * 1.9, 4.3))
    for r, sp in enumerate(["tiger", "leopard"]):
        picks = rng.sample(by_sp[sp], min(n_col, len(by_sp[sp])))
        for c in range(n_col):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(picks):
                p, bbox, _, _ = picks[c]
                ax.imshow(_load_raw(p, bbox))
            if c == 0:
                ax.set_ylabel(sp.capitalize(), fontsize=11, fontweight="bold")
    fig.suptitle("Terrestrial Mammal Dataset — sample test identities", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 2 — qualitative retrieval ──────────────────────────────────────────

def fig_retrieval(test_ds, feats, labels, species, path):
    rng = random.Random(SEED)
    samples = test_ds.samples
    dist = 1.0 - feats @ feats.T
    np.fill_diagonal(dist, np.inf)

    rows = []
    for sp in ["tiger", "leopard"]:
        cand = [i for i in range(len(labels))
                if species[i] == sp and int((labels == labels[i]).sum()) > 1]
        rng.shuffle(cand)
        rows.extend(cand[:2])

    n_top = 5
    fig, axes = plt.subplots(len(rows), n_top + 1,
                             figsize=((n_top + 1) * 1.85, len(rows) * 1.95))
    for r, qi in enumerate(rows):
        order = np.argsort(dist[qi])[:n_top]
        p, b, _, sp = samples[qi]
        ax = axes[r, 0]
        ax.imshow(_load_raw(p, b))
        _style(ax, "black")
        ax.set_ylabel(sp.capitalize(), fontsize=10, fontweight="bold")
        if r == 0:
            ax.set_title("Query", fontsize=10)
        for c, gi in enumerate(order):
            gp, gb, _, _ = samples[gi]
            ax = axes[r, c + 1]
            ax.imshow(_load_raw(gp, gb))
            color = "#2ca02c" if labels[gi] == labels[qi] else "#d62728"
            _style(ax, color)
            if r == 0:
                ax.set_title(f"Rank {c + 1}", fontsize=10)
    fig.suptitle("Qualitative retrieval — green: correct identity, red: incorrect",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 3 — t-SNE ──────────────────────────────────────────────────────────

def fig_tsne(feats, labels, species, path):
    rng = random.Random(SEED)
    emb = TSNE(n_components=2, random_state=SEED,
               perplexity=30, init="pca").fit_transform(feats)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for sp, col in [("tiger", "#1f77b4"), ("leopard", "#ff7f0e")]:
        m = species == sp
        axes[0].scatter(emb[m, 0], emb[m, 1], s=10, c=col, alpha=0.6, label=sp)
    axes[0].legend(loc="best")
    axes[0].set_title("Test embeddings coloured by species")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    uniq = list(np.unique(labels))
    rng.shuffle(uniq)
    pick = uniq[:12]
    axes[1].scatter(emb[:, 0], emb[:, 1], s=8, c="lightgrey", alpha=0.5)
    cmap = plt.get_cmap("tab20")
    for i, uid in enumerate(pick):
        m = labels == uid
        axes[1].scatter(emb[m, 0], emb[m, 1], s=24, color=cmap(i),
                        edgecolors="k", linewidths=0.3)
    axes[1].set_title("12 sample identities — tight clusters indicate strong re-ID")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(checkpoint):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    train_ds, _val, test_ds = load_terrestrial(verbose=False)

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, checkpoint), map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        n_train, state = ckpt.get("num_classes", 160), ckpt["model_state"]
    else:
        n_train, state = train_ds.num_classes, ckpt
    model = MetaFeatureAdapter(num_classes=n_train).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {checkpoint}")

    feats, labels, species = _extract(model, test_ds, device)

    for name, fn in [
        ("fig_dataset_samples.png", lambda p: fig_dataset_samples(test_ds, p)),
        ("fig_retrieval.png",       lambda p: fig_retrieval(test_ds, feats, labels, species, p)),
        ("fig_tsne.png",            lambda p: fig_tsne(feats, labels, species, p)),
    ]:
        try:
            fn(os.path.join(RESULTS_DIR, name))
        except Exception as e:
            print(f"  WARNING: {name} failed — {e}")

    print(f"\nFigures written to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="terrestrial_mfa_ft4_epoch60.pth")
    args = parser.parse_args()
    main(args.checkpoint)
