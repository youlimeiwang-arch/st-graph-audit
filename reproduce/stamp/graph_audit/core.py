"""ST-Graph Audit: method-agnostic harm metrics on spatial / histology graphs.

Standard (short form):
  Any histology-augmented spatial GNN should report, on labeled Visium:
    1. n_edges, frac_cross_coarse, n_idc_dcis (tissue-specific collide)
    2. collision_rate (= primary collide / n_edges)
    3. median gene cosine on all / collide edges
    4. median appearance cosine (img or FM) on collide edges if histology used
    5. median nn_mult (edge_dist / median spatial NN)
    6. lookalike_frac: gene_cos<=τ_g ∧ appearance_cos>=τ_a
    7. remote_frac: nn_mult >= τ_r
    8. inconsistency_median: continuous lookalike pressure on collide edges
    9. skip_rule_on: frozen ON/OFF rule for FM skip (breast)
   10. remote_gene_agree_*: remote ∧ gene_cos>τ_gene on collide (gate blind spot)
   11. spatial_lookalike_*: lookalike ∧ ¬remote on collide (spatial baseline harm)
   12. (optional) graph-harm 5-seed IQR — via harm_seed_stability()
   13. (optional v0.7) max_hops=2 skip cap arm — clears remote_gene_agree blind spot

This module only scores a graph; it does not train models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from stamp.edges import _coarse_label, _fm_matrix

# Frozen thresholds (match lookalike enablement rule family)
TAU_GENE = 0.20
TAU_IMG = 0.70
TAU_FM = 0.80
TAU_REMOTE = 3.0

# Frozen skip ON/OFF rule (breast FM skip; matches run_second_visium_screen)
SKIP_RULE_N_MIN = 10
SKIP_RULE_NN_MIN = 3.0
SKIP_RULE_GENE_MAX = 0.20


@dataclass(frozen=True)
class AuditThresholds:
    tau_gene: float = TAU_GENE
    tau_img: float = TAU_IMG
    tau_fm: float = TAU_FM
    tau_remote: float = TAU_REMOTE


def skip_rule_on(n_collide: int, nn_mult: float, gene_median: float) -> bool:
    """Frozen breast skip rule: ON iff remote IDC–DCIS collide pattern."""
    if not np.isfinite(gene_median):
        return False
    return (
        n_collide >= SKIP_RULE_N_MIN
        and nn_mult >= SKIP_RULE_NN_MIN
        and gene_median <= SKIP_RULE_GENE_MAX
    )


def skip_rule_from_summary(summary: dict) -> bool | None:
    """Infer skip_rule_on from an audit summary (breast idc_dcis primary)."""
    if summary.get("primary") == "cross":
        return None
    n = summary.get("n_idc_dcis") or summary.get("primary_collide")
    nn = summary.get("nn_mult_idc_dcis_median")
    gene = summary.get("gene_cos_idc_dcis_median")
    if n is None or nn is None or gene is None:
        return None
    return skip_rule_on(int(n), float(nn), float(gene))


def coarse_labels(adata) -> np.ndarray:
    return np.array([_coarse_label(x) for x in adata.obs["ground_truth"].astype(str)])


def median_nn_dist(xy: np.ndarray) -> float:
    d = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]
    return float(np.median(d))


def adj_to_undirected_pairs(adj: np.ndarray) -> list[tuple[int, int]]:
    a = (np.asarray(adj) > 0).astype(np.uint8)
    np.fill_diagonal(a, 0)
    a = np.maximum(a, a.T)
    ii, jj = np.where(np.triu(a, 1))
    return list(zip(ii.tolist(), jj.tolist()))


def edge_table(
    adata,
    pairs: list[tuple[int, int]],
    *,
    img_key: Optional[str] = None,
    fm_key: Optional[str] = None,
    gene_key: str = "feat",
    thresholds: AuditThresholds | None = None,
) -> pd.DataFrame:
    thr = thresholds or AuditThresholds()
    coarse = coarse_labels(adata)
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    nn = median_nn_dist(xy)
    gene = _fm_matrix(adata, gene_key, pca_dim=min(32, np.asarray(adata.obsm[gene_key]).shape[1]))
    gene = gene / (np.linalg.norm(gene, axis=1, keepdims=True) + 1e-8)

    img = None
    if img_key and img_key in adata.obsm:
        img = np.asarray(adata.obsm[img_key], dtype=np.float64)
        img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)

    fm = None
    if fm_key and fm_key in adata.obsm:
        fm = _fm_matrix(adata, fm_key, pca_dim=16)
        fm = fm / (np.linalg.norm(fm, axis=1, keepdims=True) + 1e-8)

    # tissue-specific "primary collide" pairs
    # Breast: IDC/Invasive–DCIS. CRC: Tumor–Stroma.
    # Liver Phase 1-B: periportal–pericentral. Kidney Phase 1-C: proximal–distal / tubule–glom.
    collide_sets = {
        frozenset({"IDC", "DCIS"}),
        frozenset({"Invasive", "DCIS"}),
        frozenset({"Tumor", "Stroma"}),
        frozenset({"Zone_PP", "Zone_PC"}),
        frozenset({"Periportal", "Pericentral"}),
        frozenset({"Proximal", "Distal"}),
        frozenset({"PT", "DT"}),
        frozenset({"Tubule", "Glomerulus"}),
        frozenset({"LH", "CD"}),
    }

    rows = []
    for i, j in pairs:
        pair = frozenset({coarse[i], coarse[j]})
        if coarse[i] == coarse[j]:
            relation = "same"
        elif pair in collide_sets:
            relation = "idc_dcis"  # includes Invasive–DCIS
        else:
            relation = "other_cross"
        gcos = float(np.dot(gene[i], gene[j]))
        spat = float(np.linalg.norm(xy[i] - xy[j]))
        icos = float(np.dot(img[i], img[j])) if img is not None else np.nan
        fcos = float(np.dot(fm[i], fm[j])) if fm is not None else np.nan
        app = fcos if fm is not None else icos
        app_thr = thr.tau_fm if fm is not None else thr.tau_img
        lookalike = int(np.isfinite(app) and gcos <= thr.tau_gene and app >= app_thr)
        # continuous lookalike pressure: high appearance + low gene
        inc = 0.0
        if np.isfinite(app):
            inc = max(0.0, app - app_thr) * max(0.0, thr.tau_gene - gcos)
        rows.append(
            {
                "i": i,
                "j": j,
                "ci": coarse[i],
                "cj": coarse[j],
                "relation": relation,
                "gene_cos": gcos,
                "img_cos": icos,
                "fm_cos": fcos,
                "appearance_cos": app,
                "spatial_dist": spat,
                "nn_mult": spat / max(nn, 1e-8),
                "lookalike": lookalike,
                "inconsistency": inc,
                "remote": int(spat / max(nn, 1e-8) >= thr.tau_remote),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, method: str, notes: str = "") -> dict:
    n = len(df)
    if n == 0:
        return {
            "method": method,
            "n_edges": 0,
            "n_idc_dcis": 0,
            "n_cross_coarse": 0,
            "lookalike_idc_dcis_n": 0,
            "lookalike_cross_n": 0,
            "remote_idc_dcis_n": 0,
            "remote_cross_n": 0,
            "remote_gene_agree_idc_dcis_n": 0,
            "remote_gene_agree_cross_n": 0,
            "spatial_lookalike_idc_dcis_n": 0,
            "spatial_lookalike_cross_n": 0,
            "frac_cross_coarse": None,
            "notes": notes,
            "error": "empty graph",
        }
    cross = df["relation"] != "same"
    idc = df["relation"] == "idc_dcis"
    collide = df[idc]
    cross_df = df[cross]

    def med(s):
        s = s.dropna()
        return float(s.median()) if len(s) else None

    return {
        "method": method,
        "n_edges": int(n),
        "frac_cross_coarse": float(cross.mean()),
        "n_cross_coarse": int(cross.sum()),
        "n_idc_dcis": int(idc.sum()),
        "frac_idc_dcis": float(idc.mean()),
        "collision_rate": float(idc.mean()),
        "collision_rate_cross": float(cross.mean()),
        "gene_cos_median": med(df["gene_cos"]),
        "gene_cos_idc_dcis_median": med(collide["gene_cos"]) if idc.any() else None,
        "gene_cos_cross_median": med(cross_df["gene_cos"]) if cross.any() else None,
        "img_cos_median": med(df["img_cos"]),
        "img_cos_idc_dcis_median": med(collide["img_cos"]) if idc.any() else None,
        "fm_cos_median": med(df["fm_cos"]),
        "fm_cos_idc_dcis_median": med(collide["fm_cos"]) if idc.any() else None,
        "fm_cos_cross_median": med(cross_df["fm_cos"]) if cross.any() else None,
        "nn_mult_median": med(df["nn_mult"]),
        "nn_mult_idc_dcis_median": med(collide["nn_mult"]) if idc.any() else None,
        "nn_mult_cross_median": med(cross_df["nn_mult"]) if cross.any() else None,
        "inconsistency_median": med(df["inconsistency"]),
        "inconsistency_idc_dcis_median": med(collide["inconsistency"]) if idc.any() else None,
        "inconsistency_cross_median": med(cross_df["inconsistency"]) if cross.any() else None,
        "lookalike_n": int(df["lookalike"].sum()),
        "lookalike_frac": float(df["lookalike"].mean()),
        "lookalike_idc_dcis_n": int(df.loc[idc, "lookalike"].sum()) if idc.any() else 0,
        "lookalike_cross_n": int(df.loc[cross, "lookalike"].sum()) if cross.any() else 0,
        "remote_frac": float(df["remote"].mean()),
        "remote_idc_dcis_n": int(df.loc[idc, "remote"].sum()) if idc.any() else 0,
        "remote_cross_n": int(df.loc[cross, "remote"].sum()) if cross.any() else 0,
        "remote_gene_agree_idc_dcis_n": int(
            ((collide["remote"] == 1) & (collide["gene_cos"] > TAU_GENE)).sum()
        )
        if idc.any()
        else 0,
        "remote_gene_agree_cross_n": int(
            ((cross_df["remote"] == 1) & (cross_df["gene_cos"] > TAU_GENE)).sum()
        )
        if cross.any()
        else 0,
        "spatial_lookalike_idc_dcis_n": int(
            ((collide["lookalike"] == 1) & (collide["remote"] == 0)).sum()
        )
        if idc.any()
        else 0,
        "spatial_lookalike_cross_n": int(
            ((cross_df["lookalike"] == 1) & (cross_df["remote"] == 0)).sum()
        )
        if cross.any()
        else 0,
        "skip_rule_on": skip_rule_from_summary(
            {
                "primary": "idc_dcis",
                "n_idc_dcis": int(idc.sum()),
                "nn_mult_idc_dcis_median": med(collide["nn_mult"]) if idc.any() else None,
                "gene_cos_idc_dcis_median": med(collide["gene_cos"]) if idc.any() else None,
            }
        ),
        "thresholds": {
            "tau_gene": TAU_GENE,
            "tau_img": TAU_IMG,
            "tau_fm": TAU_FM,
            "tau_remote": TAU_REMOTE,
            "skip_rule": f"n>={SKIP_RULE_N_MIN}, nn_mult>={SKIP_RULE_NN_MIN}, gene<={SKIP_RULE_GENE_MAX}",
            "remote_gene_agree": f"remote ∧ gene_cos>{TAU_GENE} on collide",
            "spatial_lookalike": "lookalike ∧ ¬remote on collide",
        },
        "notes": notes,
    }


def score_graph(
    adata,
    adj: np.ndarray,
    method: str,
    *,
    img_key: Optional[str] = None,
    fm_key: Optional[str] = None,
    notes: str = "",
    thresholds: AuditThresholds | None = None,
) -> tuple[dict, pd.DataFrame]:
    pairs = adj_to_undirected_pairs(adj)
    df = edge_table(adata, pairs, img_key=img_key, fm_key=fm_key, thresholds=thresholds)
    return summarize(df, method, notes=notes), df


def apply_edge_dropout(adj: np.ndarray, pdrop: np.ndarray, seed: int) -> np.ndarray:
    """STAIG net.dropout_adj(force_undirected=True): each stored entry with i <= j survives iff
    rand >= pdrop[i, j] (one draw per undirected edge), then the kept edges are mirrored."""
    rng = np.random.default_rng(seed)
    a = np.asarray(adj, dtype=np.float64)
    p = np.asarray(pdrop, dtype=np.float64)
    i, j = np.nonzero(np.triu(a != 0))
    keep = rng.random(i.size) >= p[i, j]
    out = np.zeros_like(a)
    out[i[keep], j[keep]] = a[i[keep], j[keep]]
    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 0.0)
    return out


def harm_seed_stability(
    adata,
    adj_fn: Callable[[int], np.ndarray],
    method: str,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    img_key: Optional[str] = None,
    fm_key: Optional[str] = None,
    primary: str = "idc_dcis",
) -> dict:
    """Five-seed IQR on graph-harm metrics (lookalike / remote primary collide)."""
    look_vals, remote_vals, collide_vals = [], [], []
    for seed in seeds:
        adj = adj_fn(seed)
        summary, _ = score_graph(adata, adj, method, img_key=img_key, fm_key=fm_key, notes=f"seed={seed}")
        if primary == "cross":
            look_vals.append(summary.get("lookalike_cross_n") or 0)
            remote_vals.append(summary.get("remote_cross_n") or 0)
            collide_vals.append(summary.get("n_cross_coarse") or 0)
        else:
            look_vals.append(summary.get("lookalike_idc_dcis_n") or 0)
            remote_vals.append(summary.get("remote_idc_dcis_n") or 0)
            collide_vals.append(summary.get("n_idc_dcis") or 0)

    def iqr_stats(vals: list[int]) -> dict:
        arr = np.asarray(vals, dtype=np.float64)
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        return {
            "values": vals,
            "median": float(med),
            "iqr": float(q75 - q25),
            "span": float(arr.max() - arr.min()),
        }

    return {
        "method": method,
        "seeds": list(seeds),
        "primary": primary,
        "lookalike": iqr_stats(look_vals),
        "remote": iqr_stats(remote_vals),
        "collide": iqr_stats(collide_vals),
    }


def export_edge_csv(df: pd.DataFrame, path: str, *, method: str = "", slide: str = "") -> None:
    """Write per-edge audit table for supplementary."""
    out = df.copy()
    if method:
        out.insert(0, "method", method)
    if slide:
        out.insert(0, "slide", slide)
    out.to_csv(path, index=False)


def ensure_feat(adata, n_hvg: int = 3000) -> None:
    """Put HVG log-normalized matrix into obsm['feat'] if missing."""
    if "feat" in adata.obsm:
        return
    import scanpy as sc

    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=n_hvg)
    mat = ad[:, ad.var["highly_variable"]].X
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    adata.obsm["feat"] = StandardScaler().fit_transform(np.asarray(mat, dtype=np.float64))
