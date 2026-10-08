"""Recompute STAIG drop-edge probabilities from a frozen FM.

STAIG's image path that actually trains the GNN is edge_probabilities
(euclidean on img_emb → log-softmax). Node features stay gene.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def recompute_edge_probabilities(data, key: str, pca_dim: int = 16) -> None:
    x = np.asarray(data.obsm[key], dtype=np.float64).reshape(data.n_obs, -1)
    x = StandardScaler().fit_transform(x)
    if pca_dim and x.shape[1] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=42).fit_transform(x)
        x = StandardScaler().fit_transform(x)
    graph = np.asarray(data.obsm["graph_neigh"])
    n = graph.shape[0]
    weights = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        js = np.flatnonzero(graph[i] == 1)
        if js.size == 0:
            continue
        dist = np.linalg.norm(x[i] - x[js], axis=1)
        weights[i, js] = dist
    probs = np.zeros_like(weights)
    for i in range(n):
        nz = weights[i] > 0
        if not nz.any():
            continue
        probs[i, nz] = softmax(np.log(weights[i, nz]))
    data.obsm["edge_probabilities"] = probs
    print(
        f"recomputed edge_probabilities from {key} dim={x.shape[1]} "
        f"pca={pca_dim or 'none'} mean_keep={probs[graph == 1].mean():.4f}",
        flush=True,
    )


def scale_drop_by_gene_z(
    data,
    scale: float,
    gene_key: str = "feat",
    pca_dim: int = 32,
) -> None:
    """Multiply BYOL drop-probs by (1 + scale * z(gene_dist)) on spatial edges.

    Unlike softmax mixing, this changes the drop rate: gene-disagreeing
    neighbors (typical of Tumor_edge vs IDC) are dropped more.
    """
    if scale == 0:
        return
    g = _fm_matrix(data, gene_key, pca_dim=min(pca_dim, np.asarray(data.obsm[gene_key]).shape[1]))
    graph = np.asarray(data.obsm["graph_neigh"])
    old = np.array(data.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    new = old.copy()
    n = graph.shape[0]
    for i in range(n):
        js = np.flatnonzero(graph[i] == 1)
        if js.size == 0:
            continue
        dist = np.linalg.norm(g[i] - g[js], axis=1)
        z = (dist - dist.mean()) / (dist.std() + 1e-8)
        new[i, js] = np.clip(old[i, js] * (1.0 + scale * z), 0.01, 0.95)
    data.obsm["edge_probabilities"] = new
    print(
        f"scaled gene-z drop scale={scale} mean_p {old[graph == 1].mean():.4f} -> {new[graph == 1].mean():.4f}",
        flush=True,
    )


def mix_gene_drop_probabilities(
    data,
    alpha: float,
    gene_key: str = "feat",
    pca_dim: int = 32,
) -> None:
    """Convex-mix STAIG (image) drop-probs with gene-space drop-probs on current edges.

    Tumor-edge vs IDC can look similar on H&E, so BYOL keeps those spatial edges.
    Gene distance on the same neighbors should raise drop probability.
    """
    if alpha <= 0:
        return
    g = _fm_matrix(data, gene_key, pca_dim=min(pca_dim, np.asarray(data.obsm[gene_key]).shape[1]))
    graph = np.asarray(data.obsm["graph_neigh"])
    old = np.array(data.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    gene_p = np.zeros_like(old)
    n = graph.shape[0]
    for i in range(n):
        js = np.flatnonzero(graph[i] == 1)
        if js.size == 0:
            continue
        dist = np.linalg.norm(g[i] - g[js], axis=1)
        gene_p[i, js] = softmax(np.log(np.maximum(dist, 1e-8)))
    mixed = (1.0 - alpha) * old + alpha * gene_p
    data.obsm["edge_probabilities"] = mixed
    print(
        f"mixed gene drop alpha={alpha} mean_p {old[graph == 1].mean():.4f} -> {mixed[graph == 1].mean():.4f}",
        flush=True,
    )


def _spatial_gene_hetero(data, gene_key: str = "feat", pca_dim: int = 32) -> np.ndarray:
    g = _fm_matrix(data, gene_key, pca_dim=min(pca_dim, np.asarray(data.obsm[gene_key]).shape[1]))
    graph = np.asarray(data.obsm["graph_neigh"])
    h = np.zeros(data.n_obs, dtype=np.float64)
    for i in range(data.n_obs):
        js = np.flatnonzero(graph[i] == 1)
        if js.size:
            h[i] = float(np.linalg.norm(g[i] - g[js], axis=1).mean())
    return h


def gene_module_mask(data, genes: list[str], top_frac: float = 0.2) -> np.ndarray:
    """True on spots in the top_frac of a canonical gene-module score."""
    present = [g for g in genes if g in data.var_names]
    if not present:
        print(f"gene-module none of {genes} in var_names", flush=True)
        return np.zeros(data.n_obs, dtype=bool)
    x = data[:, present].X
    x = np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.log1p(np.clip(x, 0, None))
    x = StandardScaler().fit_transform(x)
    score = x.mean(axis=1)
    thr = float(np.quantile(score, 1.0 - top_frac))
    mask = score >= thr
    print(
        f"gene-module genes={present} top_frac={top_frac} n={int(mask.sum())}/{len(mask)} thr={thr:.3f}",
        flush=True,
    )
    return mask


def _fm_matrix(data, key: str, pca_dim: int = 16) -> np.ndarray:
    x = np.asarray(data.obsm[key], dtype=np.float64).reshape(data.n_obs, -1)
    x = StandardScaler().fit_transform(x)
    if pca_dim and x.shape[1] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=42).fit_transform(x)
        x = StandardScaler().fit_transform(x)
    return x


def _pairwise_hops(graph, max_h: int) -> np.ndarray:
    adj = (np.asarray(graph) > 0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    dist = np.zeros(adj.shape, dtype=np.int16)
    dist[adj > 0] = 1
    hop = adj.copy()
    for h in range(2, max_h + 1):
        hop = ((hop @ adj) > 0).astype(np.float32)
        np.fill_diagonal(hop, 0.0)
        new = (hop > 0) & (dist == 0)
        dist[new] = h
    return dist


def add_fm_skip_edges(
    data,
    key: str,
    k: int = 1,
    drop_prob: float = 0.05,
    pca_dim: int = 16,
    gene_key: str = "",
    gene_k: int = 0,
    gene_pca: int = 32,
    gene_mutual: bool = True,
    also_key: str = "",
    protect_frac: float = 0.0,
    protect_bridge: bool = False,
    protect_mask=None,
    only_mask=None,
    min_hops: int = 0,
    max_hops: int = 0,
) -> None:
    """Keep STAIG spatial edges; add a few FM-nearest non-neighbors as extra edges.

    If gene_k>0, an FM neighbor is kept only when it is also a gene-space
    non-spatial neighbor (consensus skip). That is meant to block IDC–DCIS
    histology lookalikes that disagree on expression. gene_mutual=True
    requires a two-way gene kNN; False is one-way (looser).
    """
    if k <= 0:
        return
    x = _fm_matrix(data, key, pca_dim)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    graph = np.array(data.obsm["graph_neigh"], dtype=np.float64, copy=True)
    probs = np.array(data.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    sim[graph > 0] = -np.inf
    cand_k = k
    gene_ok = None
    if gene_k and gene_key:
        cand_k = max(k, gene_k)
        g = _fm_matrix(data, gene_key, pca_dim=min(gene_pca, np.asarray(data.obsm[gene_key]).shape[1]))
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
        gsim = g @ g.T
        np.fill_diagonal(gsim, -np.inf)
        gsim[graph > 0] = -np.inf
        g_nn = np.argpartition(-gsim, kth=min(gene_k, data.n_obs - 1) - 1, axis=1)[:, :gene_k]
        gene_ok = np.zeros((data.n_obs, data.n_obs), dtype=bool)
        for i in range(data.n_obs):
            gene_ok[i, g_nn[i]] = True
        n_oneway = int(gene_ok.sum())
        if gene_mutual:
            gene_ok = gene_ok & gene_ok.T
        print(
            f"gene-gate {gene_key} pca<={gene_pca} k={gene_k} "
            f"mutual={gene_mutual} one_way={n_oneway} "
            f"kept_pairs={int(np.triu(gene_ok, 1).sum())}",
            flush=True,
        )
    also_ok = None
    if also_key:
        y = _fm_matrix(data, also_key, pca_dim)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
        ysim = y @ y.T
        np.fill_diagonal(ysim, -np.inf)
        ysim[graph > 0] = -np.inf
        y_nn = np.argpartition(-ysim, kth=min(cand_k, data.n_obs - 1) - 1, axis=1)[:, :cand_k]
        also_ok = np.zeros((data.n_obs, data.n_obs), dtype=bool)
        for i in range(data.n_obs):
            also_ok[i, y_nn[i]] = True
        print(
            f"also-gate {also_key} top-{cand_k} directed={int(also_ok.sum())}",
            flush=True,
        )
    protect = None
    if protect_frac > 0:
        h = _spatial_gene_hetero(data, gene_key or "feat")
        thr = float(np.quantile(h, 1.0 - protect_frac))
        protect = h >= thr
        print(
            f"skip-protect frac={protect_frac} n={int(protect.sum())}/{len(protect)} "
            f"thr={thr:.4f} bridge={protect_bridge}",
            flush=True,
        )
    if protect_mask is not None:
        extra = np.asarray(protect_mask, dtype=bool).reshape(-1)
        protect = extra if protect is None else (protect | extra)
        print(f"skip-protect mask n={int(extra.sum())}/{len(extra)}", flush=True)
    only = None
    if only_mask is not None:
        only = np.asarray(only_mask, dtype=bool).reshape(-1)
        print(f"skip-only-mask n={int(only.sum())}/{len(only)}", flush=True)
    hopdist = None
    if min_hops or max_hops:
        cap = int(max_hops or 12)
        hopdist = _pairwise_hops(graph, cap)
        print(f"skip-hops min={min_hops} max={max_hops or 'inf'} cap={cap}", flush=True)
    nn = np.argpartition(-sim, kth=min(cand_k, data.n_obs - 1) - 1, axis=1)[:, :cand_k]
    added = 0
    blocked = 0
    skip_pairs: set[tuple[int, int]] = set()
    for i in range(data.n_obs):
        order = nn[i][np.argsort(-sim[i, nn[i]])]
        picked = 0
        for j in order:
            if graph[i, j] > 0:
                continue
            if gene_ok is not None and not gene_ok[i, j]:
                blocked += 1
                continue
            if also_ok is not None and not (also_ok[i, j] or also_ok[j, i]):
                blocked += 1
                continue
            if protect is not None:
                if protect_bridge:
                    if bool(protect[i]) != bool(protect[j]):
                        blocked += 1
                        continue
                elif protect[i] or protect[j]:
                    blocked += 1
                    continue
            if only is not None and not (only[i] and only[j]):
                blocked += 1
                continue
            if hopdist is not None:
                d = int(hopdist[i, j])
                if max_hops and (d == 0 or d > max_hops):
                    blocked += 1
                    continue
                if min_hops and d != 0 and d < min_hops:
                    blocked += 1
                    continue
            graph[i, j] = 1.0
            graph[j, i] = 1.0
            probs[i, j] = drop_prob
            probs[j, i] = drop_prob
            added += 1
            skip_pairs.add((min(i, j), max(i, j)))
            picked += 1
            if picked >= k:
                break
    data.obsm["graph_neigh"] = graph
    data.obsm["edge_probabilities"] = probs
    if skip_pairs and "ground_truth" in data.obs:
        _skip_pair_stats(data, list(skip_pairs))
    print(
        f"added {added} directed skip edges from {key} k={k} drop_prob={drop_prob} "
        f"gene_k={gene_k or 0} blocked={blocked} (undirected pairs {len(skip_pairs)})",
        flush=True,
    )


def _coarse_label(s: str) -> str:
    s = str(s)
    if s.startswith("IDC"):
        return "IDC"
    if s.startswith("DCIS"):
        return "DCIS"
    if s.startswith("Healthy"):
        return "Healthy"
    if s.startswith("Tumor_edge"):
        return "Edge"
    if s.startswith("Invasive"):
        return "Invasive"
    if s == "DCIS":
        return "DCIS"
    if s.startswith("Lymphocytes"):
        return "Immune"
    if s.startswith("Stroma"):
        return "Stroma"
    if s.startswith("Normal"):
        return "Normal"
    return s.split()[0]


def _skip_pair_stats(data, pairs: list[tuple[int, int]]) -> None:
    gt = data.obs["ground_truth"].astype(str).to_numpy()
    c = np.array([_coarse_label(x) for x in gt])
    same = sum(1 for i, j in pairs if c[i] == c[j])
    cross_dcis = sum(
        1 for i, j in pairs if {c[i], c[j]} == {"IDC", "DCIS"} or {c[i], c[j]} == {"Invasive", "DCIS"}
    )
    print(
        f"skip-pair stats n={len(pairs)} same_coarse={same / max(len(pairs), 1):.3f} "
        f"IDC/Invasive-DCIS={cross_dcis}",
        flush=True,
    )


def add_hop_gene_skip(
    data,
    hops: int = 2,
    k: int = 1,
    drop_prob: float = 0.05,
    gene_pca: int = 32,
) -> None:
    """Gene-similar edges among spatial 2..hops-hop spots. Not global FM skip."""
    if hops < 2 or k <= 0:
        return
    graph = np.array(data.obsm["graph_neigh"], dtype=np.float64, copy=True)
    probs = np.array(data.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    adj = (graph > 0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    hop = adj.copy()
    band = np.zeros_like(adj)
    for _ in range(2, hops + 1):
        hop = ((hop @ adj) > 0).astype(np.float32)
        np.fill_diagonal(hop, 0.0)
        band = np.maximum(band, hop * (1.0 - (adj > 0).astype(np.float32)))
    g = _fm_matrix(data, "feat", pca_dim=min(gene_pca, np.asarray(data.obsm["feat"]).shape[1]))
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
    added = 0
    pairs: set[tuple[int, int]] = set()
    n = data.n_obs
    for i in range(n):
        js = np.flatnonzero(band[i] > 0)
        if js.size == 0:
            continue
        sim = g[i] @ g[js].T
        order = js[np.argsort(-sim)]
        picked = 0
        for j in order:
            j = int(j)
            if graph[i, j] > 0:
                continue
            graph[i, j] = 1.0
            graph[j, i] = 1.0
            probs[i, j] = drop_prob
            probs[j, i] = drop_prob
            added += 1
            pairs.add((min(i, j), max(i, j)))
            picked += 1
            if picked >= k:
                break
    data.obsm["graph_neigh"] = graph
    data.obsm["edge_probabilities"] = probs
    if pairs and "ground_truth" in data.obs:
        _skip_pair_stats(data, list(pairs))
    print(
        f"added {added} directed hop-gene skip hops={hops} k={k} "
        f"pairs={len(pairs)} band_mean={float(band.mean()):.4f}",
        flush=True,
    )
