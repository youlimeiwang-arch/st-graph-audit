"""M3: a few mutual-kNN edges between slices in a feature space.

Intra-slice spatial kNN stays as STAIG built it. Coordinates are not aligned.
New edges are contrastive positives and GNN neighbors. Drop probability is low
so they are kept more often than noisy intra-slice edges.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def add_cross_slice_edges(
    adata,
    feat_key: str = "feat",
    k: int = 2,
    pca_dim: int = 64,
    batch_key: str = "batch",
    min_sim: float = 0.0,
    drop_floor: float = 0.05,
) -> dict:
    if batch_key not in adata.obs:
        raise KeyError(f"adata.obs has no {batch_key}")
    feat = np.asarray(adata.obsm[feat_key], dtype=np.float64)
    feat = feat.reshape(feat.shape[0], -1)
    x = StandardScaler().fit_transform(feat)
    dim = min(int(pca_dim), x.shape[0] - 1, x.shape[1])
    if dim >= 2:
        x = PCA(n_components=dim, random_state=42).fit_transform(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    x = x / norms
    batches = adata.obs[batch_key].astype(str).to_numpy()
    labels = np.unique(batches)
    adj = np.array(adata.obsm["graph_neigh"], dtype=np.float64, copy=True)
    probs = np.array(adata.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    n_new = 0
    sims = []
    for a, b in combinations(labels, 2):
        ia = np.flatnonzero(batches == a)
        ib = np.flatnonzero(batches == b)
        if ia.size == 0 or ib.size == 0:
            continue
        sim = x[ia] @ x[ib].T
        kk = min(int(k), ia.size, ib.size)
        knn_ab = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        knn_ba = np.argpartition(-sim.T, kth=kk - 1, axis=1)[:, :kk]
        ba_sets = [set(row.tolist()) for row in knn_ba]
        for i in range(ia.size):
            for j in knn_ab[i]:
                j = int(j)
                if i not in ba_sets[j]:
                    continue
                w = float(sim[i, j])
                if w < min_sim:
                    continue
                u, v = int(ia[i]), int(ib[j])
                if adj[u, v] > 0:
                    continue
                adj[u, v] = adj[v, u] = 1.0
                p = max(float(drop_floor), 0.5 * (1.0 - w))
                probs[u, v] = probs[v, u] = p
                n_new += 1
                sims.append(w)
    adata.obsm["graph_neigh"] = adj
    adata.obsm["edge_probabilities"] = probs
    if "mask_neigh" in adata.obsm:
        mask = np.array(adata.obsm["mask_neigh"], dtype=np.float64, copy=True)
        mask = np.maximum(mask, (adj > 0).astype(np.float64))
        adata.obsm["mask_neigh"] = mask
    stats = {
        "n_new_undirected": n_new,
        "k": int(k),
        "feat_key": feat_key,
        "mean_sim": float(np.mean(sims)) if sims else 0.0,
        "n_batches": int(labels.size),
    }
    print(
        f"M3 added {n_new} mutual-kNN edges  k={k} feat={feat_key} "
        f"mean_cos={stats['mean_sim']:.3f} batches={list(labels)}",
        flush=True,
    )
    return stats
