"""Edge-level harm counts and the section-level switch. Scores a graph; trains nothing."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .features import fm_vectors, gene_vectors
from .graph import edge_pairs
from .labels import DEFAULT_COLLIDE_PAIRS, coarse_labels, relation


@dataclass(frozen=True)
class Thresholds:
    """Fixed on Visium breast cancer Block A and used unchanged on every other section."""

    tau_gene: float = 0.20
    tau_fm: float = 0.80
    tau_remote: float = 3.0
    switch_min_collide: int = 10
    switch_min_nn_mult: float = 3.0
    switch_max_gene_cos: float = 0.20


def median_nn_distance(xy) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    d = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]
    return float(np.median(d))


def edge_table(
    graph,
    xy,
    gene,
    fm,
    labels,
    *,
    collide_pairs=DEFAULT_COLLIDE_PAIRS,
    thresholds: Thresholds | None = None,
) -> pd.DataFrame:
    """One row per undirected edge: labels, relation, gene/FM cosine, nn_mult, lookalike and remote flags."""
    thr = thresholds or Thresholds()
    pairs = edge_pairs(graph)
    xy = np.asarray(xy, dtype=np.float64)
    unit = median_nn_distance(xy)
    g = gene_vectors(gene)
    f = fm_vectors(fm)
    coarse = coarse_labels(labels)
    if not pairs:
        return pd.DataFrame(columns=["i", "j", "label_i", "label_j", "relation", "gene_cos", "fm_cos",
                                     "nn_mult", "lookalike", "remote"])
    i, j = (np.array(v) for v in zip(*pairs))
    gene_cos = np.array([float(np.dot(g[a], g[b])) for a, b in pairs])
    fm_cos = np.array([float(np.dot(f[a], f[b])) for a, b in pairs])
    nn_mult = np.array([float(np.linalg.norm(xy[a] - xy[b])) for a, b in pairs]) / max(unit, 1e-8)
    return pd.DataFrame({
        "i": i,
        "j": j,
        "label_i": coarse[i],
        "label_j": coarse[j],
        "relation": relation(coarse[i], coarse[j], collide_pairs),
        "gene_cos": gene_cos,
        "fm_cos": fm_cos,
        "nn_mult": nn_mult,
        "lookalike": ((gene_cos <= thr.tau_gene) & (fm_cos >= thr.tau_fm)).astype(int),
        "remote": (nn_mult >= thr.tau_remote).astype(int),
    })


def harm(edges: pd.DataFrame, primary: str = "collide") -> dict:
    """Counts on the primary edges: ``collide`` pairs, or every cross-label edge (``cross``, e.g. cortex layers)."""
    if primary not in ("collide", "cross"):
        raise ValueError(f"primary must be 'collide' or 'cross', got {primary!r}")
    sel = edges[edges["relation"] == "collide"] if primary == "collide" else edges[edges["relation"] != "same"]

    def med(col):
        return float(sel[col].median()) if len(sel) else None

    return {
        "n_edges": int(len(edges)),
        "n_primary": int(len(sel)),
        "lookalike": int(sel["lookalike"].sum()),
        "remote": int(sel["remote"].sum()),
        "nn_mult_median": med("nn_mult"),
        "gene_cos_median": med("gene_cos"),
    }


def switch_on(n_collide: int, nn_mult_median, gene_cos_median, thresholds: Thresholds | None = None) -> bool:
    """ON iff many collide edges exist, most are long-range, and they join spots with different expression."""
    thr = thresholds or Thresholds()
    if nn_mult_median is None or gene_cos_median is None or not np.isfinite(gene_cos_median):
        return False
    return (
        n_collide >= thr.switch_min_collide
        and nn_mult_median >= thr.switch_min_nn_mult
        and gene_cos_median <= thr.switch_max_gene_cos
    )


def thresholds_dict(thresholds: Thresholds | None = None) -> dict:
    return asdict(thresholds or Thresholds())
