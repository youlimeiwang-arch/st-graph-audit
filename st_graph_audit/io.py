"""Load a section into the arrays the audit needs: coordinates, expression features, FM embedding, labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import gene_features


@dataclass
class Section:
    barcodes: np.ndarray
    xy: np.ndarray
    gene: np.ndarray
    fm: np.ndarray
    labels: np.ndarray

    @property
    def n_spots(self) -> int:
        return len(self.barcodes)


def read_labels(path) -> pd.Series:
    """Tab-separated, no header: barcode, label."""
    df = pd.read_csv(path, sep="\t", header=None, index_col=0)
    return df[1]


def _keep_labeled(adata, labels: pd.Series):
    lab = labels.reindex(adata.obs_names)
    keep = lab.notna().to_numpy() & (lab.astype(str).str.len() > 0).to_numpy()
    if not keep.any():
        raise ValueError("no spot barcode matches the label file")
    adata = adata[keep].copy()
    adata.obs["label"] = lab[keep].astype(str).to_numpy()
    return adata


def _align_fm(fm: np.ndarray, kept: np.ndarray, n_all: int) -> np.ndarray:
    fm = fm.reshape(fm.shape[0], -1)
    if fm.shape[0] == kept.sum():
        return fm
    if fm.shape[0] == n_all:
        return fm[kept]
    raise ValueError(f"FM embedding has {fm.shape[0]} rows; expected {kept.sum()} (labeled spots) or {n_all} (all spots)")


def load_visium(folder, labels_file, fm_file, *, n_top_genes: int = 3000,
                count_file: str = "filtered_feature_bc_matrix.h5") -> Section:
    """Space Ranger output folder + label file + FM embedding (.npy, rows in Space Ranger spot order).

    Unlabeled spots are dropped before features and graphs are computed.
    """
    import scanpy as sc

    folder = Path(folder)
    adata = sc.read_visium(str(folder), count_file=count_file, load_images=True)
    adata.var_names_make_unique()
    labels = read_labels(labels_file)
    lab = labels.reindex(adata.obs_names)
    kept = (lab.notna() & (lab.astype(str).str.len() > 0)).to_numpy()
    fm = _align_fm(np.load(fm_file), kept, adata.n_obs)
    adata = _keep_labeled(adata, labels)
    return Section(
        barcodes=np.asarray(adata.obs_names),
        xy=np.asarray(adata.obsm["spatial"]),
        gene=gene_features(adata, n_top_genes),
        fm=fm,
        labels=adata.obs["label"].to_numpy(),
    )


def load_h5ad(path, *, fm_key: str, label_key: str, gene_key: str | None = None,
              spatial_key: str = "spatial", n_top_genes: int = 3000) -> Section:
    """AnnData with raw counts in ``X`` (or ready features in ``obsm[gene_key]``), FM in ``obsm[fm_key]``."""
    import anndata

    adata = anndata.read_h5ad(path)
    for key, where in ((fm_key, adata.obsm), (spatial_key, adata.obsm)):
        if key not in where:
            raise KeyError(f"obsm[{key!r}] not found; available: {list(adata.obsm)}")
    if label_key not in adata.obs:
        raise KeyError(f"obs[{label_key!r}] not found; available: {list(adata.obs.columns)}")
    lab = adata.obs[label_key]
    keep = (lab.notna() & (lab.astype(str).str.len() > 0)).to_numpy()
    adata = adata[keep].copy()
    gene = np.asarray(adata.obsm[gene_key]) if gene_key else gene_features(adata, n_top_genes)
    return Section(
        barcodes=np.asarray(adata.obs_names),
        xy=np.asarray(adata.obsm[spatial_key]),
        gene=gene,
        fm=np.asarray(adata.obsm[fm_key]),
        labels=adata.obs[label_key].astype(str).to_numpy(),
    )
