"""Unsupervised cluster heads on a trained STAMP/STAIG embedding.

mclust matches the STAIG paper. ensemble = one-shot gene kNN on mclust/KMeans
disagreements. iter = expand only high-confidence gene votes, then fill the rest.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from staig.utils import mclust_R, refine_label
from stamp.edges import _fm_matrix, _spatial_gene_hetero


def align_labels(src, ref) -> np.ndarray:
    src = np.asarray(src).astype(str)
    ref = np.asarray(ref).astype(str)
    a, b = np.unique(src), np.unique(ref)
    c = np.zeros((len(a), len(b)), dtype=np.int64)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i, j] = int(((src == x) & (ref == y)).sum())
    ri, cj = linear_sum_assignment(-c)
    mapping = {a[i]: b[j] for i, j in zip(ri, cj)}
    return np.array([mapping.get(x, x) for x in src])


def spatial_refine(adata, labels, radius: int = 15) -> np.ndarray:
    tmp = adata.copy()
    tmp.obs["domain"] = np.asarray(labels).astype(str)
    return np.asarray(refine_label(tmp, radius=radius, key="domain"))


def kmeans_head(adata, ncl: int, seed: int = 42, radius: int = 15) -> np.ndarray:
    z = normalize(StandardScaler().fit_transform(np.asarray(adata.obsm["emb"], dtype=np.float64)))
    pred = KMeans(n_clusters=ncl, random_state=seed, n_init=10).fit_predict(z).astype(str)
    return spatial_refine(adata, pred, radius=radius)


def _gene_space(adata, gene_pca: int = 32) -> np.ndarray:
    g = _fm_matrix(adata, "feat", pca_dim=gene_pca)
    return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)


def gene_knn_resolve(adata, mclust, kmeans, gene_k: int = 15, gene_pca: int = 32) -> np.ndarray:
    km_al = align_labels(kmeans, mclust)
    agree = km_al == np.asarray(mclust).astype(str)
    g = _gene_space(adata, gene_pca)
    conf = np.flatnonzero(agree)
    nn = NearestNeighbors(n_neighbors=min(gene_k, len(conf)), metric="cosine").fit(g[conf])
    mix = km_al.copy()
    if (~agree).any():
        _, ind = nn.kneighbors(g[~agree])
        votes = km_al[conf][ind]
        for i, row in zip(np.flatnonzero(~agree), votes):
            vals, cnt = np.unique(row, return_counts=True)
            mix[i] = vals[int(cnt.argmax())]
    print(
        f"ensemble disagree={int((~agree).sum())}/{len(agree)} "
        f"gene_k={gene_k} resolved on agreeing spots",
        flush=True,
    )
    return mix


def gene_knn_resolve_iter(
    adata,
    mclust,
    kmeans,
    gene_k: int = 15,
    gene_pca: int = 32,
    min_agree: int = 10,
    rounds: int = 5,
) -> np.ndarray:
    """Expand labels from agreeing spots; only accept a vote if it is confident.

    Last round assigns remaining disagreements by majority. Caller should
    spatial-refine afterwards.
    """
    km_al = align_labels(kmeans, mclust)
    mclust = np.asarray(mclust).astype(str)
    labeled = km_al == mclust
    g = _gene_space(adata, gene_pca)
    mix = km_al.copy()
    for it in range(rounds):
        if labeled.all():
            break
        conf = np.flatnonzero(labeled)
        unk = np.flatnonzero(~labeled)
        kk = min(gene_k, len(conf))
        nn = NearestNeighbors(n_neighbors=kk, metric="cosine").fit(g[conf])
        _, ind = nn.kneighbors(g[unk])
        votes = mix[conf][ind]
        last = it == rounds - 1
        n_new = 0
        for i, row in zip(unk, votes):
            vals, cnt = np.unique(row, return_counts=True)
            if last or int(cnt.max()) >= min_agree:
                mix[i] = vals[int(cnt.argmax())]
                labeled[i] = True
                n_new += 1
        print(
            f"iter {it+1}/{rounds} labeled={int(labeled.sum())}/{len(labeled)} "
            f"new={n_new} min_agree={min_agree} last={last}",
            flush=True,
        )
    return mix


def dual_radius_refine(
    adata,
    labels,
    r_in: int = 50,
    r_if: int = 8,
    protect_q: float = 0.15,
) -> np.ndarray:
    """Large spatial majority in the interior; small radius on gene-heterogeneous spots."""
    h = _spatial_gene_hetero(adata, "feat")
    thr = float(np.quantile(h, 1.0 - protect_q))
    maj_in = spatial_refine(adata, labels, radius=r_in)
    maj_if = spatial_refine(adata, labels, radius=r_if)
    print(
        f"dual-radius refine r_in={r_in} r_if={r_if} protect_q={protect_q} "
        f"interface={int((h >= thr).sum())}/{len(h)}",
        flush=True,
    )
    return np.where(h >= thr, maj_if, maj_in)


def cluster_ensemble(
    adata,
    ncl: int,
    kmeans_seed: int = 42,
    radius: int = 15,
    kind: str = "ensemble",
) -> np.ndarray:
    tmp = adata.copy()
    mclust_R(tmp, num_cluster=ncl, used_obsm="emb")
    mclust = spatial_refine(adata, tmp.obs["mclust"].astype(str).to_numpy(), radius=radius)
    km = kmeans_head(adata, ncl=ncl, seed=kmeans_seed, radius=radius)
    if kind in ("iter", "dual"):
        mix = gene_knn_resolve_iter(adata, mclust, km)
        if kind == "dual":
            return dual_radius_refine(adata, mix)
        return spatial_refine(adata, mix, radius=radius)
    mix = gene_knn_resolve(adata, mclust, km)
    return spatial_refine(adata, mix, radius=radius)
