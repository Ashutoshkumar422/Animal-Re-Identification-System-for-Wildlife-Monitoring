# utils/visualizer.py
# Drop-in runtime visualization for training, evaluation, and validation.
# Saves all plots to results/plots/. Integrates via 3 lines in train.py / evaluate.py.

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
           "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. TrainingVisualizer  — attach to train.py
# ─────────────────────────────────────────────────────────────────────────────

class TrainingVisualizer:
    """
    Tracks per-epoch loss, LR, and optional validation metrics.
    Call .update() at end of each epoch. Call .save() at end of training.

    Usage in train.py:
        from utils.visualizer import TrainingVisualizer
        viz = TrainingVisualizer(num_epochs=60, run_name="atrw_mfa")

        for epoch in range(num_epochs):
            ...
            viz.update(epoch, loss=loss_val, lr=current_lr,
                       loss_id=info["loss_id"], loss_tri=info["loss_tri"])

        viz.save()
    """

    def __init__(self, num_epochs: int, run_name: str = "run"):
        self.num_epochs = num_epochs
        self.run_name   = run_name
        self.history    = {
            "epoch": [], "loss": [], "lr": [],
            "loss_id": [], "loss_tri": [], "loss_meta": [],
            "val_mAP": [], "val_cmc1": [],
        }

    def update(self, epoch: int, loss: float, lr: float,
               loss_id: float = None, loss_tri: float = None,
               loss_meta: float = None,
               val_mAP: float = None, val_cmc1: float = None):
        self.history["epoch"].append(epoch + 1)
        self.history["loss"].append(loss)
        self.history["lr"].append(lr)
        self.history["loss_id"].append(loss_id)
        self.history["loss_tri"].append(loss_tri)
        self.history["loss_meta"].append(loss_meta)
        self.history["val_mAP"].append(val_mAP)
        self.history["val_cmc1"].append(val_cmc1)
        # Live save every 5 epochs so plots are viewable mid-training
        if (epoch + 1) % 5 == 0:
            self._plot()

    def save(self):
        self._plot()
        # Also dump raw history as JSON for later analysis
        hist_path = os.path.join(PLOT_DIR, f"{self.run_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Visualizer] Training plots saved → {PLOT_DIR}/")

    def _plot(self):
        epochs = self.history["epoch"]
        if not epochs:
            return

        has_components = any(v is not None
                             for v in self.history["loss_id"])
        has_val = any(v is not None for v in self.history["val_mAP"])

        rows = 2 + int(has_components) + int(has_val)
        fig, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows),
                                 facecolor="#F5F5F5")
        fig.suptitle(f"Training — {self.run_name}", fontsize=14,
                     fontweight="bold", y=0.98)
        if rows == 1:
            axes = [axes]

        ax_idx = 0

        # ── Total loss ──
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, self.history["loss"],
                color=_COLORS[0], lw=2, label="Total loss")
        ax.fill_between(epochs, self.history["loss"],
                        alpha=0.15, color=_COLORS[0])
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Total Training Loss"); ax.legend(); ax.grid(alpha=0.3)

        # ── LR schedule ──
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, self.history["lr"],
                color=_COLORS[1], lw=2, label="Learning rate")
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR (log)")
        ax.set_yscale("log"); ax.set_title("Learning Rate Schedule")
        ax.legend(); ax.grid(alpha=0.3)

        # ── Loss components ──
        if has_components:
            ax = axes[ax_idx]; ax_idx += 1
            for key, label, color in [
                ("loss_id",   "ID loss",   _COLORS[2]),
                ("loss_tri",  "Triplet",   _COLORS[3]),
                ("loss_meta", "Meta (iA)", _COLORS[4]),
            ]:
                vals = [v for v in self.history[key] if v is not None]
                ep   = epochs[:len(vals)]
                if vals:
                    ax.plot(ep, vals, color=color, lw=2, label=label)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title("Loss Components"); ax.legend(); ax.grid(alpha=0.3)

        # ── Validation metrics ──
        if has_val:
            ax = axes[ax_idx]; ax_idx += 1
            map_vals  = [v for v in self.history["val_mAP"]  if v is not None]
            cmc_vals  = [v for v in self.history["val_cmc1"] if v is not None]
            ep        = epochs[:len(map_vals)]
            if map_vals:
                ax.plot(ep, map_vals,  color=_COLORS[5], lw=2, label="mAP (%)")
            if cmc_vals:
                ax.plot(ep, cmc_vals,  color=_COLORS[6], lw=2, label="CMC-1 (%)")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Metric (%)")
            ax.set_title("Validation mAP & CMC-1 vs Epoch")
            ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(PLOT_DIR, f"{self.run_name}_training.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. EvalVisualizer — attach to evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

class EvalVisualizer:
    """
    Visualizes evaluation results: CMC curve, distance distribution,
    and rank-1 retrieval samples.

    Usage in evaluate.py:
        from utils.visualizer import EvalVisualizer
        eviz = EvalVisualizer(run_name="atrw_mfa")

        # After compute_map_cmc1():
        eviz.plot_cmc_curve(q_feats, q_labels, g_feats, g_labels)
        eviz.plot_distance_distribution(q_feats, q_labels, g_feats, g_labels)
        eviz.save()
    """

    def __init__(self, run_name: str = "eval"):
        self.run_name = run_name

    def plot_cmc_curve(self, q_feats, q_labels, g_feats, g_labels,
                       max_rank: int = 20):
        sim      = q_feats @ g_feats.T
        dist     = 1 - sim
        n_query  = len(q_labels)
        hits     = np.zeros(max_rank)

        for i in range(n_query):
            sorted_idx = np.argsort(dist[i])
            sorted_lbl = g_labels[sorted_idx]
            matches    = (sorted_lbl == q_labels[i])
            for r in range(min(max_rank, len(matches))):
                if matches[:r + 1].any():
                    hits[r] += 1

        cmc = hits / n_query * 100
        ranks = np.arange(1, max_rank + 1)

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F5F5F5")
        ax.plot(ranks, cmc, color=_COLORS[0], lw=2.5, marker="o",
                markersize=4, label="MFA model")
        ax.fill_between(ranks, cmc, alpha=0.15, color=_COLORS[0])
        ax.axhline(cmc[0], color=_COLORS[2], lw=1.2,
                   linestyle="--", label=f"CMC-1 = {cmc[0]:.1f}%")
        ax.set_xlabel("Rank"); ax.set_ylabel("Identification Rate (%)")
        ax.set_title(f"CMC Curve — {self.run_name}")
        ax.set_ylim(0, 105); ax.set_xticks(ranks)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"{self.run_name}_cmc_curve.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer] CMC curve → {out}")
        return cmc

    def plot_distance_distribution(self, q_feats, q_labels,
                                   g_feats, g_labels, bins: int = 60):
        sim  = q_feats @ g_feats.T
        dist = (1 - sim).flatten()

        q_rep = np.repeat(q_labels[:, None], g_feats.shape[0], axis=1).flatten()
        g_rep = np.tile(g_labels[None, :],   (q_feats.shape[0], 1)).flatten()
        same  = dist[q_rep == g_rep]
        diff  = dist[q_rep != g_rep]

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F5F5F5")
        ax.hist(same, bins=bins, alpha=0.65, color=_COLORS[3],
                label="Same identity", density=True)
        ax.hist(diff, bins=bins, alpha=0.65, color=_COLORS[2],
                label="Different identity", density=True)
        ax.set_xlabel("Cosine Distance"); ax.set_ylabel("Density")
        ax.set_title(f"Intra vs Inter-class Distance — {self.run_name}")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"{self.run_name}_dist_distribution.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer] Distance distribution → {out}")

    def save(self):
        print(f"[Visualizer] Eval plots saved → {PLOT_DIR}/")


