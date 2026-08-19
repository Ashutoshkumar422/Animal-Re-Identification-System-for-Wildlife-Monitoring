# eval/metrics.py — mAP, CMC-1, feature extraction, t-SNE visualization

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import clip
import os
from config import RESULTS_DIR


@torch.no_grad()
def extract_features(model, loader, device, use_metadata=True):
    """
    Extract normalized embeddings for all samples in loader.

    Returns:
        features : np.ndarray (N, D)
        labels   : np.ndarray (N,)
        species  : list of str (N,)
    """
    model.eval()
    all_feats, all_labels, all_species = [], [], []

    for batch in tqdm(loader, desc="Extracting features", leave=False):
        images, prompts, labels, sp = batch
        images = images.to(device)

        if use_metadata:
            tokens = clip.tokenize(list(prompts), truncate=True).to(device)
            _, feat = model(images, tokens)
        else:
            # Baseline: visual only
            _, feat = model(images, return_feat=True)

        all_feats.append(F.normalize(feat, dim=-1).cpu())
        all_labels.extend(labels.tolist())
        all_species.extend(sp)

    return (
        torch.cat(all_feats).numpy(),
        np.array(all_labels),
        all_species,
    )


def compute_map_cmc1(query_feats, query_labels,
                     gallery_feats, gallery_labels):
    """
    Compute mAP and CMC-1 (Rank-1 accuracy).

    Uses cosine similarity (features must be L2-normalized).
    """
    # Cosine distance matrix (Q, G)
    sim  = query_feats @ gallery_feats.T          # (Q, G)
    dist = 1 - sim                                # lower = more similar

    ap_list, rank1_list = [], []
    for i in range(len(query_labels)):
        matches     = (gallery_labels == query_labels[i]).astype(int)
        sorted_idx  = np.argsort(dist[i])
        sorted_m    = matches[sorted_idx]

        # mAP: average precision over all ranks
        if matches.sum() > 0:
            ap = average_precision_score(sorted_m,
                                         -dist[i][sorted_idx])
        else:
            ap = 0.0
        ap_list.append(ap)

        # CMC-1: is the top-1 retrieved image the correct identity?
        rank1_list.append(int(sorted_m[0] == 1))

    mAP  = float(np.mean(ap_list))  * 100
    cmc1 = float(np.mean(rank1_list)) * 100
    return mAP, cmc1

def compute_rank_k(q_feats, q_labels, g_feats, g_labels, k=5):
    """CMC at rank-k"""
    sim  = q_feats @ g_feats.T
    dist = 1 - sim
    hits = 0
    for i in range(len(q_labels)):
        sorted_lbl = g_labels[np.argsort(dist[i])]
        if q_labels[i] in sorted_lbl[:k]:
            hits += 1
    return hits / len(q_labels) * 100


def evaluate_species(model, species, device, use_metadata=True):
    """
    Full intra-species evaluation: extract features → compute mAP + CMC-1.
    Now loads the species config directly.
    """
    from torch.utils.data import DataLoader
    from data.dataset import load_metawild

    # Load only the requested species config
    _, gallery_set, query_set = load_metawild(species_filter=species)

    gallery_loader = DataLoader(gallery_set, batch_size=128, num_workers=4)
    query_loader   = DataLoader(query_set,   batch_size=128, num_workers=4)

    g_feats, g_labels, _ = extract_features(model, gallery_loader,
                                             device, use_metadata)
    q_feats, q_labels, _ = extract_features(model, query_loader,
                                             device, use_metadata)

    mAP, cmc1 = compute_map_cmc1(q_feats, q_labels, g_feats, g_labels)
    return mAP, cmc1


def plot_tsne(features, labels, species_name, save_name="tsne.png"):
    """
    t-SNE visualization of learned embeddings (like Figure 5 in the paper).
    Each color = one individual identity.
    """
    print(f"Running t-SNE for {species_name}...")
    tsne  = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb2d = tsne.fit_transform(features)

    unique_ids = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_ids))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, uid in enumerate(unique_ids):
        mask = labels == uid
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   c=[cmap(i)], s=15, alpha=0.7, label=str(uid))

    ax.set_title(f"t-SNE — {species_name} embeddings", fontsize=14)
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    ax.legend(loc="best", fontsize=6, ncol=3, markerscale=1.5)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved t-SNE plot → {path}")
