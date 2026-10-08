"""One call: build ungated and gated FM skip graphs on a spatial graph, score both, decide the switch."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .audit import Thresholds, edge_table, harm, switch_on, thresholds_dict
from .graph import add_skip_edges
from .labels import DEFAULT_COLLIDE_PAIRS


@dataclass
class SectionAudit:
    switch: str  # "ON", "OFF" or "N/A" (primary == "cross")
    ungated: dict
    gated: dict
    graph_ungated: np.ndarray = field(repr=False)
    graph_gated: np.ndarray = field(repr=False)
    skip_ungated: list = field(repr=False)
    skip_gated: list = field(repr=False)
    edges_ungated: pd.DataFrame = field(repr=False)
    edges_gated: pd.DataFrame = field(repr=False)
    settings: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "switch": self.switch,
            "ungated": self.ungated,
            "gated": self.gated,
            "n_skip_ungated": len(self.skip_ungated),
            "n_skip_gated": len(self.skip_gated),
            **self.settings,
        }


def audit_section(
    spatial_graph,
    xy,
    gene,
    fm,
    labels,
    *,
    primary: str = "collide",
    collide_pairs=DEFAULT_COLLIDE_PAIRS,
    k: int = 1,
    gene_k: int = 10,
    thresholds: Thresholds | None = None,
) -> SectionAudit:
    """Audit one labeled section.

    spatial_graph: (n, n) 0/1 spatial graph, e.g. ``spatial_knn_graph(xy, 5)``.
    gene: (n, p) expression features, e.g. ``gene_features(adata)``. fm: (n, d) FM embedding.
    labels: n spot annotations. primary: ``collide`` (fixed label pairs) or ``cross`` (every cross-label edge).
    """
    g_u, skip_u = add_skip_edges(spatial_graph, fm, k=k)
    g_g, skip_g = add_skip_edges(spatial_graph, fm, gene, k=k, gene_k=gene_k, mutual=True)
    kw = dict(collide_pairs=collide_pairs, thresholds=thresholds)
    e_u = edge_table(g_u, xy, gene, fm, labels, **kw)
    e_g = edge_table(g_g, xy, gene, fm, labels, **kw)
    h_u, h_g = harm(e_u, primary), harm(e_g, primary)
    if primary == "cross":
        switch = "N/A"
    else:
        on = switch_on(h_u["n_primary"], h_u["nn_mult_median"], h_u["gene_cos_median"], thresholds)
        switch = "ON" if on else "OFF"
    settings = {
        "primary": primary,
        "collide_pairs": sorted("|".join(sorted(p)) for p in collide_pairs),
        "k": k,
        "gene_k": gene_k,
        "thresholds": thresholds_dict(thresholds),
    }
    return SectionAudit(switch, h_u, h_g, g_u, g_g, skip_u, skip_g, e_u, e_g, settings)
