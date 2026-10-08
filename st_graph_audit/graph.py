"""Spatial graph and FM skip edges, with and without the gene gate.

Graphs are dense symmetric 0/1 arrays of shape (n_spots, n_spots) with an empty diagonal.
Similarities are computed as full N x N matrices: O(N^2) memory.
"""
from __future__ import annotations

import numpy as np

from .features import fm_vectors, gene_vectors


def spatial_knn_graph(xy, k: int = 5) -> np.ndarray:
    """Link each spot to its k nearest spots in physical space and symmetrize (STAIG's default graph)."""
    import ot

    xy = np.asarray(xy)
    dist = ot.dist(xy, xy, metric="euclidean")
    n = dist.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        order = dist[i].argsort()
        adj[i, order[1 : k + 1]] = 1
    adj = adj + adj.T
    return np.where(adj > 1, 1, adj)


def mutual_gene_knn(gene_unit: np.ndarray, graph: np.ndarray, k: int = 10, mutual: bool = True) -> np.ndarray:
    """Boolean (n, n) mask: j is among i's k most gene-similar non-neighbors (and vice versa if mutual)."""
    n = gene_unit.shape[0]
    gsim = gene_unit @ gene_unit.T
    np.fill_diagonal(gsim, -np.inf)
    gsim[graph > 0] = -np.inf
    nn = np.argpartition(-gsim, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    ok = np.zeros((n, n), dtype=bool)
    for i in range(n):
        ok[i, nn[i]] = True
    return ok & ok.T if mutual else ok


def add_skip_edges(
    graph,
    fm,
    gene=None,
    *,
    k: int = 1,
    gene_k: int = 10,
    mutual: bool = True,
    fm_components: int = 16,
    gene_components: int = 32,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Add FM skip edges to ``graph``; returns (new graph, sorted undirected skip pairs).

    Ungated (``gene=None``): each spot gets its k most FM-similar non-neighbors.
    Gated: each spot's ``max(k, gene_k)`` most FM-similar non-neighbors are scanned in order of FM
    similarity and the first k that are also (mutual) gene-space ``gene_k``-nearest non-neighbors are kept.
    """
    graph = np.array(graph, dtype=np.float64, copy=True)
    n = graph.shape[0]
    x = fm_vectors(fm, fm_components)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    sim[graph > 0] = -np.inf

    gate = None
    cand_k = k
    if gene is not None and gene_k:
        cand_k = max(k, gene_k)
        gate = mutual_gene_knn(gene_vectors(gene, gene_components), graph, gene_k, mutual)

    cand = np.argpartition(-sim, kth=min(cand_k, n - 1) - 1, axis=1)[:, :cand_k]
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        picked = 0
        for j in cand[i][np.argsort(-sim[i, cand[i]])]:
            if graph[i, j] > 0:
                continue
            if gate is not None and not gate[i, j]:
                continue
            graph[i, j] = graph[j, i] = 1.0
            pairs.add((min(i, j), max(i, j)))
            picked += 1
            if picked >= k:
                break
    return graph, sorted(pairs)


def edge_pairs(graph) -> list[tuple[int, int]]:
    """Undirected edges (i < j) of a 0/1 or weighted adjacency."""
    a = (np.asarray(graph) > 0).astype(np.uint8)
    np.fill_diagonal(a, 0)
    a = np.maximum(a, a.T)
    ii, jj = np.where(np.triu(a, 1))
    return list(zip(ii.tolist(), jj.tolist()))
