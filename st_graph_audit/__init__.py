"""ST-Graph Audit: edge-level audit and gene gate for FM skip edges in spatial transcriptomics graphs."""
from .audit import Thresholds, edge_table, harm, median_nn_distance, switch_on
from .features import fm_vectors, gene_features, gene_vectors, project
from .graph import add_skip_edges, edge_pairs, mutual_gene_knn, spatial_knn_graph
from .labels import DEFAULT_COLLIDE_PAIRS, coarse_label, coarse_labels, parse_pairs
from .pipeline import SectionAudit, audit_section

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_COLLIDE_PAIRS",
    "SectionAudit",
    "Thresholds",
    "add_skip_edges",
    "audit_section",
    "coarse_label",
    "coarse_labels",
    "edge_pairs",
    "edge_table",
    "fm_vectors",
    "gene_features",
    "gene_vectors",
    "harm",
    "median_nn_distance",
    "mutual_gene_knn",
    "parse_pairs",
    "project",
    "spatial_knn_graph",
    "switch_on",
]
