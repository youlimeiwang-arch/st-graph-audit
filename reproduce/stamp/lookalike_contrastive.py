"""Track C: lookalike-aware training objectives.

Two complementary tools (no gene_k hard gate required):

1. build_lookalike_mask — mark graph edges that are high-FM / low-gene.
2. lookalike_repulsion_loss — push GNN embeddings apart on those edges
   (penalize message-passing agreement across lookalike wiring).

Legacy lookalike_edge_penalty keeps the frozen-feature form (constant if
FM/gene are frozen; useful for auditing edge weights).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def lookalike_edge_penalty(
    z_fm: torch.Tensor,
    z_gene: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    *,
    tau_fm: float = 0.80,
    tau_gene: float = 0.20,
) -> torch.Tensor:
    """Scalar score of lookalike mass on edges (frozen FM/gene features)."""
    src, dst = edge_index[0], edge_index[1]
    zf = F.normalize(z_fm, dim=-1)
    zg = F.normalize(z_gene, dim=-1)
    cos_fm = (zf[src] * zf[dst]).sum(dim=-1)
    cos_gene = (zg[src] * zg[dst]).sum(dim=-1)
    look = F.relu(cos_fm - tau_fm) * F.relu(tau_gene - cos_gene)
    w = edge_weight if edge_weight is not None else torch.ones_like(look)
    return (look * w).mean()


def build_lookalike_edge_index(
    graph_neigh: np.ndarray | torch.Tensor,
    fm: np.ndarray | torch.Tensor,
    gene: np.ndarray | torch.Tensor,
    *,
    tau_fm: float = 0.80,
    tau_gene: float = 0.20,
    spatial: np.ndarray | None = None,
    min_dist_mult: float = 0.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict]:
    """Return edge_index (2, E_look) of undirected lookalike edges in the graph.

    If spatial is given and min_dist_mult>0, keep only edges with
    euclidean distance >= min_dist_mult * median spatial NN distance
    (remote lookalike; matches ST-Graph Audit story).
    """
    if torch.is_tensor(graph_neigh):
        g = graph_neigh.detach().cpu().numpy()
    else:
        g = np.asarray(graph_neigh)
    if torch.is_tensor(fm):
        fm_np = fm.detach().cpu().numpy()
    else:
        fm_np = np.asarray(fm)
    if torch.is_tensor(gene):
        gene_np = gene.detach().cpu().numpy()
    else:
        gene_np = np.asarray(gene)

    fm_np = fm_np / (np.linalg.norm(fm_np, axis=1, keepdims=True) + 1e-8)
    gene_np = gene_np / (np.linalg.norm(gene_np, axis=1, keepdims=True) + 1e-8)
    ii, jj = np.where(np.triu(g > 0, 1))
    if len(ii) == 0:
        empty = torch.zeros(2, 0, dtype=torch.long, device=device)
        return empty, {"n_graph_edges": 0, "n_lookalike_edges": 0}

    cos_fm = (fm_np[ii] * fm_np[jj]).sum(axis=1)
    cos_gene = (gene_np[ii] * gene_np[jj]).sum(axis=1)
    keep = (cos_fm >= tau_fm) & (cos_gene <= tau_gene)
    nn_scale = None
    if spatial is not None and min_dist_mult > 0:
        xy = np.asarray(spatial, dtype=np.float64)
        from sklearn.neighbors import NearestNeighbors

        d1 = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]
        nn_scale = float(np.median(d1))
        dist = np.linalg.norm(xy[ii] - xy[jj], axis=1)
        keep = keep & (dist >= min_dist_mult * nn_scale)
    src = np.concatenate([ii[keep], jj[keep]])
    dst = np.concatenate([jj[keep], ii[keep]])
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)
    stats = {
        "n_graph_edges": int(len(ii)),
        "n_lookalike_edges": int(keep.sum()),
        "frac_lookalike": float(keep.sum() / max(len(ii), 1)),
        "tau_fm": tau_fm,
        "tau_gene": tau_gene,
        "min_dist_mult": min_dist_mult,
        "nn_scale": nn_scale,
    }
    return edge_index, stats


def lookalike_repulsion_loss(
    z: torch.Tensor,
    lookalike_edge_index: torch.Tensor,
    *,
    margin: float = 0.5,
) -> torch.Tensor:
    """Discourage agreement of GNN embeddings across lookalike edges.

    L = mean( relu( cos(z_i, z_j) - margin ) ) over lookalike edges.
    """
    if lookalike_edge_index.numel() == 0:
        return z.new_zeros(())
    src, dst = lookalike_edge_index[0], lookalike_edge_index[1]
    zn = F.normalize(z, dim=-1)
    cos = (zn[src] * zn[dst]).sum(dim=-1)
    return F.relu(cos - margin).mean()


def soft_downweight_lookalike_probs(
    edge_probabilities: np.ndarray,
    graph_neigh: np.ndarray,
    fm: np.ndarray,
    gene: np.ndarray,
    *,
    tau_fm: float = 0.80,
    tau_gene: float = 0.20,
    factor: float = 0.1,
    spatial: np.ndarray | None = None,
    min_dist_mult: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Optional preprocess: raise drop prob on lookalike edges (soft gate).

    Keep probability of lookalike edges is multiplied by ``factor`` (smaller =
    stronger soft gate). If spatial + min_dist_mult>0, only remote lookalikes.
    """
    p = np.array(edge_probabilities, dtype=np.float64, copy=True)
    g = np.asarray(graph_neigh)
    fm_n = fm / (np.linalg.norm(fm, axis=1, keepdims=True) + 1e-8)
    gene_n = gene / (np.linalg.norm(gene, axis=1, keepdims=True) + 1e-8)
    ii, jj = np.where(np.triu(g > 0, 1))
    nn_scale = None
    dist = None
    if spatial is not None and min_dist_mult > 0 and len(ii):
        xy = np.asarray(spatial, dtype=np.float64)
        from sklearn.neighbors import NearestNeighbors

        d1 = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]
        nn_scale = float(np.median(d1))
        dist = np.linalg.norm(xy[ii] - xy[jj], axis=1)
    n = 0
    for k, (i, j) in enumerate(zip(ii, jj)):
        if float(fm_n[i] @ fm_n[j]) < tau_fm or float(gene_n[i] @ gene_n[j]) > tau_gene:
            continue
        if dist is not None and dist[k] < min_dist_mult * nn_scale:
            continue
        # higher drop probability → less likely kept in STAIG dropout
        p[i, j] = min(1.0, 1.0 - (1.0 - p[i, j]) * factor)
        p[j, i] = p[i, j]
        n += 1
    return p, {
        "n_downweighted": n,
        "factor": factor,
        "min_dist_mult": min_dist_mult,
        "nn_scale": nn_scale,
    }
