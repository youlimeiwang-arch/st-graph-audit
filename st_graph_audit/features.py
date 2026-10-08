"""Feature matrices used for graph construction and edge scoring."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def project(x, n_components: int) -> np.ndarray:
    """Standardize, project to the first ``n_components`` principal components, standardize again."""
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(x.shape[0], -1)
    x = StandardScaler().fit_transform(x)
    if n_components and x.shape[1] > n_components:
        x = PCA(n_components=n_components, random_state=42).fit_transform(x)
        x = StandardScaler().fit_transform(x)
    return x


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def fm_vectors(fm, n_components: int = 16) -> np.ndarray:
    """L2-normalized FM projection; cosine similarity is then a dot product."""
    return unit_rows(project(fm, n_components))


def gene_vectors(gene, n_components: int = 32) -> np.ndarray:
    gene = np.asarray(gene)
    return unit_rows(project(gene, min(n_components, gene.reshape(gene.shape[0], -1).shape[1])))


def gene_features(adata, n_top_genes: int = 3000) -> np.ndarray:
    """STAIG's expression input: seurat_v3 HVGs on raw counts, normalize to 1e4, log1p, scale (no centering, max 10).

    ``adata.X`` must hold raw counts. Returns a dense (n_spots, n_hvg) array; ``adata`` is not modified.
    """
    import scanpy as sc

    ad = adata.copy()
    sc.pp.highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.scale(ad, zero_center=False, max_value=10)
    x = ad[:, ad.var["highly_variable"]].X
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x)
