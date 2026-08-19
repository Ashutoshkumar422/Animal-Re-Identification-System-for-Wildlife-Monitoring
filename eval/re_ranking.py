# eval/re_ranking.py
# k-reciprocal re-ranking (Zhong et al., "Re-ranking Person Re-identification
# with k-reciprocal Encoding", CVPR 2017).
#
# A standard, method-agnostic test-time technique: it refines a distance
# matrix using the k-reciprocal neighbour structure of the gallery and
# routinely lifts mAP by several points. ARNet (Xu et al. 2026) uses plain
# Euclidean distance with no re-ranking.
#
# This is the closed-set variant (Query == Gallery): one L2-normalized
# feature matrix in, one refined N x N distance matrix out (lower = closer).

import numpy as np


def re_ranking(feats, k1=20, k2=6, lambda_value=0.3):
    """
    feats        : (N, D) L2-normalized features; query set == gallery set.
    k1, k2       : neighbourhood sizes for k-reciprocal encoding / expansion.
    lambda_value : blend between Jaccard distance and original distance.
    Returns an (N, N) re-ranked distance matrix.
    """
    feats = np.asarray(feats, dtype=np.float32)
    N = feats.shape[0]
    k1 = min(k1, N - 1)
    k2 = min(k2, N)

    # Original distance: squared Euclidean of L2-normalized feats = 2 - 2*cos.
    original_dist = 2.0 - 2.0 * (feats @ feats.T)
    original_dist = np.clip(original_dist, 0.0, None).astype(np.float32)
    # Column-normalize, then transpose (standard pre-processing).
    original_dist = original_dist / (np.max(original_dist, axis=0) + 1e-12)
    original_dist = original_dist.T

    V = np.zeros((N, N), dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    for i in range(N):
        # k-reciprocal neighbours of i.
        fwd   = initial_rank[i, :k1 + 1]
        bwd   = initial_rank[fwd, :k1 + 1]
        recip = fwd[np.any(bwd == i, axis=1)]

        # Local expansion of the reciprocal set with a smaller neighbourhood.
        expansion = recip
        half = int(round(k1 / 2)) + 1
        for cand in recip:
            c_fwd   = initial_rank[cand, :half]
            c_bwd   = initial_rank[c_fwd, :half]
            c_recip = c_fwd[np.any(c_bwd == cand, axis=1)]
            if len(c_recip) and \
               len(np.intersect1d(c_recip, recip)) > 2.0 / 3.0 * len(c_recip):
                expansion = np.append(expansion, c_recip)
        expansion = np.unique(expansion)

        w = np.exp(-original_dist[i, expansion])
        V[i, expansion] = w / np.sum(w)

    # k2 local query expansion on the V encoding.
    if k2 != 1:
        V_qe = np.zeros_like(V)
        for i in range(N):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe

    # Jaccard distance via an inverted index.
    inv_index = [np.where(V[:, i] != 0)[0] for i in range(N)]
    jaccard   = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        tmp = np.zeros(N, dtype=np.float32)
        nz  = np.where(V[i] != 0)[0]
        for j, col in enumerate(nz):
            imgs = inv_index[col]
            tmp[imgs] += np.minimum(V[i, col], V[imgs, col])
        jaccard[i] = 1.0 - tmp / (2.0 - tmp)

    final_dist = jaccard * (1.0 - lambda_value) + original_dist * lambda_value
    return final_dist.astype(np.float32)
