"""ST-Graph Audit Benchmark engine (Track A product).

Shared by CLI (`python -m stamp.graph_audit`) and thin script wrappers.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from sklearn.preprocessing import StandardScaler

from stamp.edges import add_fm_skip_edges
from stamp.graph_audit.core import (
    apply_edge_dropout,
    ensure_feat,
    export_edge_csv,
    harm_seed_stability,
    score_graph,
    skip_rule_on,
)

# ---------------------------------------------------------------------------
# Project root & third-party paths
# ---------------------------------------------------------------------------

def project_root() -> Path:
    env = os.environ.get("STAIG_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent.parent
    if (here / "stamp").is_dir() and (here / "Dataset").is_dir():
        return here
    return Path.cwd().resolve()


def setup_env(root: Path | None = None) -> Path:
    root = root or project_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    tp = root.parent / "third_party" / "STAGATE_pyG"
    if tp.is_dir() and str(tp) not in sys.path:
        sys.path.insert(0, str(tp))
    conda = os.environ.get("STAIG_CONDA", "/root/autodl-tmp/conda_envs/staig_env")
    if Path(conda).is_dir():
        os.environ.setdefault("R_HOME", f"{conda}/lib/R")
        os.environ["PATH"] = f"{conda}/bin:" + os.environ.get("PATH", "")
        lib = f"{conda}/lib"
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if lib not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{lib}:{ld}" if ld else lib
    return root


ROOT = project_root()
TAB = ROOT / "lab/static/tables"
FIG = ROOT / "lab/static/figs/lookalike_mainline"
LOG = ROOT / "logs"
EDGE_DIR = TAB / "st_graph_audit_edges"
SLIDES_YAML = ROOT / "config/st_graph_audit_slides.yaml"


def load_slide_registry() -> list[dict]:
    if not SLIDES_YAML.is_file():
        raise FileNotFoundError(f"missing slide registry: {SLIDES_YAML}")
    data = yaml.safe_load(SLIDES_YAML.read_text(encoding="utf-8"))
    slides = []
    for s in data.get("slides", []):
        spec = dict(s)
        spec["path"] = ROOT / spec["path"]
        slides.append(spec)
    return slides


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def _patch_networkx_sparse() -> None:
    import networkx as nx

    if hasattr(nx, "to_scipy_sparse_matrix"):
        return

    def _to_scipy_sparse_matrix(G, *args, **kwargs):
        return nx.to_scipy_sparse_array(G, *args, **kwargs).tocsr()

    nx.to_scipy_sparse_matrix = _to_scipy_sparse_matrix  # type: ignore[attr-defined]


def graph_graphst(raw) -> np.ndarray:
    from GraphST.preprocess import construct_interaction

    ad = raw.copy()
    construct_interaction(ad, n_neighbors=3)
    return np.asarray(ad.obsm["adj"])


def graph_stagate(raw, rad_cutoff: float | None = None) -> tuple[np.ndarray, str]:
    from STAGATE_pyG import Cal_Spatial_Net
    from sklearn.neighbors import NearestNeighbors

    ad = raw.copy()
    xy = np.asarray(ad.obsm["spatial"], dtype=np.float64)
    med_nn = float(np.median(NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]))
    if rad_cutoff is None:
        rad_cutoff = max(150.0, 1.5 * med_nn)
    Cal_Spatial_Net(ad, rad_cutoff=float(rad_cutoff))
    net = ad.uns["Spatial_Net"]
    cells = list(ad.obs_names)
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    adj = np.zeros((n, n), dtype=np.float64)
    for _, r in net.iterrows():
        i, j = idx.get(r["Cell1"]), idx.get(r["Cell2"])
        if i is None or j is None:
            continue
        adj[i, j] = 1
        adj[j, i] = 1
    note = f"Cal_Spatial_Net rad_cutoff={rad_cutoff:.1f} (median_nn={med_nn:.1f})"
    return adj, note


def graph_spaceflow(raw, n_neighbors: int = 10) -> tuple[np.ndarray, str]:
    from sklearn.neighbors import kneighbors_graph

    ad = raw.copy()
    if "spatial" not in ad.obsm:
        raise KeyError("adata.obsm['spatial'] required for SpaceFlow graph_alpha")
    try:
        _patch_networkx_sparse()
        from SpaceFlow.SpaceFlow import SpaceFlow

        sf = SpaceFlow(adata=ad)
        g = sf.graph_alpha(ad.obsm["spatial"], n_neighbors=n_neighbors)
        adj = (g.toarray() > 0).astype(np.float64) if hasattr(g, "toarray") else np.asarray(g > 0, dtype=np.float64)
        adj = np.maximum(adj, adj.T)
        return adj, f"SpaceFlow graph_alpha n_neighbors={n_neighbors}"
    except Exception as e:
        g = kneighbors_graph(ad.obsm["spatial"], n_neighbors=n_neighbors, mode="connectivity", include_self=False)
        adj = g.toarray().astype(np.float64)
        adj = np.maximum(adj, adj.T)
        return adj, f"fallback spatial kNN k={n_neighbors} (SpaceFlow alpha failed: {type(e).__name__}: {e})"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_staig(path: Path, kind: str, truth_file: str = "truth.txt"):
    from staig.adata_processing import LoadSingle10xAdata

    tf = path / truth_file
    backup = None
    if truth_file != "truth.txt" and tf.is_file():
        if (path / "truth.txt").is_file():
            backup = path / ".truth_audit_backup"
            shutil.copy2(path / "truth.txt", backup)
        shutil.copy2(tf, path / "truth.txt")
    try:
        has_byol = (path / "embeddings.npy").is_file()
        adata = LoadSingle10xAdata(
            path=str(path),
            n_neighbors=5,
            n_top_genes=3000,
            image_emb=has_byol,
            label=tf.is_file() or (path / "truth.txt").is_file(),
        ).run()
        if "ground_truth" in adata.obs:
            adata = adata[adata.obs["ground_truth"].astype(str).str.len() > 0].copy()
        for key in ("phikonv2_emb", "phikon_emb"):
            p = path / f"{key}.npy"
            if not p.is_file():
                continue
            raw = np.load(p)
            raw = raw.reshape(raw.shape[0], -1) if raw.ndim > 1 else raw.reshape(-1, 1024)
            if raw.shape[0] == adata.n_obs:
                adata.obsm[key] = StandardScaler().fit_transform(raw).astype(np.float64)
                break
            full = LoadSingle10xAdata(
                path=str(path), n_neighbors=5, n_top_genes=3000, image_emb=False, label=False
            ).run()
            if raw.shape[0] != full.n_obs:
                continue
            idx = full.obs_names.get_indexer(adata.obs_names)
            if (idx < 0).any():
                continue
            adata.obsm[key] = StandardScaler().fit_transform(raw[idx]).astype(np.float64)
            break
        return adata
    finally:
        if backup is not None and backup.is_file():
            shutil.copy2(backup, path / "truth.txt")
            backup.unlink(missing_ok=True)


def load_raw(path: Path, kind: str, truth_file: str = "truth.txt"):
    adata = sc.read_visium(str(path), count_file="filtered_feature_bc_matrix.h5", load_images=True)
    adata.var_names_make_unique()
    truth = path / truth_file
    if truth.is_file():
        meta = pd.read_csv(truth, sep="\t", header=None, index_col=0)
        lab = meta.reindex(adata.obs_names)[1]
        adata.obs["ground_truth"] = lab
        adata = adata[
            ~pd.isnull(adata.obs["ground_truth"])
            & (adata.obs["ground_truth"].astype(str).str.len() > 0)
        ].copy()
    return adata


def fm_key(adata) -> str | None:
    for k in ("phikonv2_emb", "phikon_emb"):
        if k in adata.obsm:
            return k
    return None


def _attach_primary(summary: dict, spec: dict) -> dict:
    if spec["primary"] == "cross":
        summary["primary"] = "cross"
        summary["primary_collide"] = summary.get("n_cross_coarse")
        summary["primary_lookalike"] = summary.get("lookalike_cross_n")
        summary["primary_remote"] = summary.get("remote_cross_n")
        summary["primary_remote_gene_agree"] = summary.get("remote_gene_agree_cross_n")
        summary["primary_spatial_lookalike"] = summary.get("spatial_lookalike_cross_n")
        summary["collision_rate_primary"] = summary.get("collision_rate_cross")
        summary["skip_rule_on"] = None
    else:
        summary["primary"] = "idc_dcis"
        summary["primary_collide"] = summary.get("n_idc_dcis")
        summary["primary_lookalike"] = summary.get("lookalike_idc_dcis_n")
        summary["primary_remote"] = summary.get("remote_idc_dcis_n")
        summary["primary_remote_gene_agree"] = summary.get("remote_gene_agree_idc_dcis_n")
        summary["primary_spatial_lookalike"] = summary.get("spatial_lookalike_idc_dcis_n")
        summary["collision_rate_primary"] = summary.get("collision_rate")
    return summary


# ---------------------------------------------------------------------------
# Audit one slide
# ---------------------------------------------------------------------------

def audit_slide(
    spec: dict,
    *,
    export_edges: bool = False,
    seed_harm: bool = True,
) -> tuple[list[dict], list[dict]]:
    path = spec["path"]
    truth_file = spec.get("truth_file", "truth.txt")
    ref = load_staig(path, spec["kind"], truth_file)
    raw = load_raw(path, spec["kind"], truth_file)
    names = [b for b in ref.obs_names if b in raw.obs_names]
    ref = ref[names].copy()
    raw = raw[names].copy()
    ensure_feat(raw)
    if "feat" not in ref.obsm:
        ref.obsm["feat"] = raw.obsm["feat"]
    fk = fm_key(ref)
    rows: list[dict] = []
    seed_rows: list[dict] = []
    edge_exports: list[tuple[str, pd.DataFrame]] = []

    def add(method: str, adj, notes: str, *, store_edges: bool = False):
        try:
            summary, edf = score_graph(
                ref,
                adj,
                method,
                img_key="img_emb" if "img_emb" in ref.obsm else None,
                fm_key=fk,
                notes=notes,
            )
            summary["slide"] = spec["id"]
            summary["slide_role"] = spec["role"]
            summary["label_note"] = spec.get("label_note", "")
            _attach_primary(summary, spec)
            rows.append(summary)
            if store_edges and export_edges:
                edge_exports.append((method, edf))
        except Exception as e:
            rows.append(
                {
                    "slide": spec["id"],
                    "method": method,
                    "error": f"{type(e).__name__}: {e}",
                    "notes": notes,
                    "primary": spec["primary"],
                }
            )

    add("GraphST (k=3)", graph_graphst(raw), "construct_interaction k=3")
    adj_sg, note_sg = graph_stagate(raw)
    add("STAGATE (radius)", adj_sg, note_sg)
    adj_sf, note_sf = graph_spaceflow(raw, 10)
    add("SpaceFlow (spatial)", adj_sf, note_sf)
    add("STAIG spatial (k=5)", np.asarray(ref.obsm["graph_neigh"]), "n_neighbors=5")

    g0 = np.array(ref.obsm["graph_neigh"], dtype=np.float64, copy=True)
    ep0 = np.array(ref.obsm["edge_probabilities"], dtype=np.float64, copy=True)

    if "edge_probabilities" in ref.obsm:
        from stamp.graph_audit_deep import build_effective_graph_row, effective_graph_harm_median

        eff = effective_graph_harm_median(ref, primary=spec.get("primary", "idc_dcis"))
        rows.append(build_effective_graph_row(ref, spec, eff))

    if seed_harm and "edge_probabilities" in ref.obsm:
        def dropout_fn(seed: int) -> np.ndarray:
            return apply_edge_dropout(g0, ep0, seed)

        stab = harm_seed_stability(
            ref,
            dropout_fn,
            "STAIG spatial dropout (5-seed)",
            img_key="img_emb" if "img_emb" in ref.obsm else None,
            fm_key=fk,
            primary=spec["primary"],
        )
        stab["slide"] = spec["id"]
        seed_rows.append(stab)

    if fk:
        ad_u = ref.copy()
        ad_u.obsm["graph_neigh"] = g0.copy()
        ad_u.obsm["edge_probabilities"] = ep0.copy()
        add_fm_skip_edges(ad_u, fk, k=1, drop_prob=0.05, pca_dim=16)
        add(
            f"STAIG+ungated skip ({fk})",
            np.asarray(ad_u.obsm["graph_neigh"]),
            f"skip k=1 {fk}; skip_rule applies to this arm",
            store_edges=True,
        )

        # skip rule evaluated on ungated skip harm pattern
        ungated = next((r for r in rows if "ungated skip" in r.get("method", "")), None)
        if ungated and spec["primary"] == "idc_dcis":
            ungated["skip_rule_on"] = skip_rule_on(
                int(ungated.get("n_idc_dcis") or 0),
                float(ungated.get("nn_mult_idc_dcis_median") or 0),
                float(ungated.get("gene_cos_idc_dcis_median") or float("nan")),
            )

        if spec["gate"]:
            ad_g = ref.copy()
            ad_g.obsm["graph_neigh"] = g0.copy()
            ad_g.obsm["edge_probabilities"] = ep0.copy()
            add_fm_skip_edges(
                ad_g, fk, k=1, drop_prob=0.05, pca_dim=16, gene_key="feat", gene_k=10, gene_mutual=True
            )
            add("STAMP gene-gated skip", np.asarray(ad_g.obsm["graph_neigh"]), "gene_k=10 mutual")
            ad_mh = ref.copy()
            ad_mh.obsm["graph_neigh"] = g0.copy()
            ad_mh.obsm["edge_probabilities"] = ep0.copy()
            add_fm_skip_edges(
                ad_mh,
                fk,
                k=1,
                drop_prob=0.05,
                pca_dim=16,
                gene_key="feat",
                gene_k=10,
                gene_mutual=True,
                max_hops=2,
            )
            add(
                "STAMP gene-gated + max_hops=2",
                np.asarray(ad_mh.obsm["graph_neigh"]),
                "gene_k=10 mutual; skip max_hops=2 (v0.7 audit extension)",
            )
        elif spec["primary"] == "cross":
            # DLPFC: still run gated arm for cross-layer comparison
            ad_g = ref.copy()
            ad_g.obsm["graph_neigh"] = g0.copy()
            ad_g.obsm["edge_probabilities"] = ep0.copy()
            add_fm_skip_edges(
                ad_g, fk, k=1, drop_prob=0.05, pca_dim=16, gene_key="feat", gene_k=10, gene_mutual=True
            )
            add("STAMP gene-gated skip", np.asarray(ad_g.obsm["graph_neigh"]), "gene_k=10 mutual (cross-layer)")

        if seed_harm:
            def ungated_dropout(seed: int) -> np.ndarray:
                ad = ref.copy()
                ad.obsm["graph_neigh"] = g0.copy()
                ad.obsm["edge_probabilities"] = ep0.copy()
                add_fm_skip_edges(ad, fk, k=1, drop_prob=0.05, pca_dim=16)
                base = np.asarray(ad.obsm["graph_neigh"], dtype=np.float64)
                return apply_edge_dropout(base, np.asarray(ad.obsm["edge_probabilities"]), seed)

            stab2 = harm_seed_stability(
                ref,
                ungated_dropout,
                f"STAIG+ungated skip dropout (5-seed)",
                fm_key=fk,
                primary=spec["primary"],
            )
            stab2["slide"] = spec["id"]
            seed_rows.append(stab2)

    if export_edges and edge_exports:
        EDGE_DIR.mkdir(parents=True, exist_ok=True)
        for method, edf in edge_exports:
            safe = method.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
            p = EDGE_DIR / f"{spec['id']}_{safe}_edges.csv"
            export_edge_csv(edf, str(p), method=method, slide=spec["id"])

    return rows, seed_rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def fmt_cell(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, bool):
        return "ON" if x else "OFF"
    return str(x)


def _int0(x) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    return int(x)


def write_gate_boundary(all_rows: list[dict], path: Path) -> None:
    """Gate repair boundary: 4 harm channels on STAMP gene-gated arm."""
    lines = [
        "# ST-Graph Audit · Gate 边界表（v0.7 四通道 harm）",
        "",
        "> 必报：**lookalike · remote · remote_gene_agree · spatial_lookalike**（gene-gated 臂）",
        "",
        "| slide | skip_rule | lookalike | remote | remote_gene_agree | spatial_lookalike | 全清? |",
        "|-------|-----------|-----------|--------|-------------------|-------------------|-------|",
    ]
    for slide in sorted({r.get("slide") for r in all_rows if r.get("slide")}):
        gated = _row_gated(all_rows, slide)
        if not gated:
            continue
        ungated = next(
            (r for r in all_rows if r.get("slide") == slide and "ungated skip" in r.get("method", "")),
            None,
        )
        skip_rule = ungated.get("skip_rule_on") if ungated else gated.get("skip_rule_on")
        pl = gated.get("primary_lookalike") or 0
        pr = gated.get("primary_remote") or 0
        prga = gated.get("primary_remote_gene_agree") or 0
        psl = gated.get("primary_spatial_lookalike") or 0
        all_clear = pl == 0 and pr == 0 and prga == 0 and psl == 0
        lines.append(
            f"| {slide} | {fmt_cell(skip_rule)} | {pl} | {pr} | {prga} | {psl} | "
            f"{'✅' if all_clear else '—'} |"
        )
    lines += [
        "",
        "**读法**",
        "",
        "- `remote_gene_agree`：gate 设计盲区（高 gene 互邻的远程跨类 skip）→ Block A 应为 **0**",
        "- `spatial_lookalike`：空间基线 lookalike，skip gate 不覆盖 → CytAssist 允许 ≤1",
        "- Sec2 remote_gene_agree=16 为 xfer 标签 + 高 gene 远程 skip 的预期边界 case",
        "- **v0.7**：`max_hops=2` 扩展臂可清零 Sec2/Wu 的 rga → 见 `st_graph_audit_remote_cap_boundary.md`",
        "",
        f"_generated {_now_iso()}_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _row_gated(all_rows: list[dict], slide: str) -> dict | None:
    for r in all_rows:
        if r.get("slide") == slide and "gene-gated" in r.get("method", "") and "error" not in r:
            return r
    return None


def write_leaderboard(rows: list[dict], path: Path) -> None:
    """Who wires wrong classes? Sorted by primary_remote then primary_lookalike."""
    ok = [r for r in rows if "error" not in r]
    ok.sort(
        key=lambda r: (
            -(r.get("primary_remote") or 0),
            -(r.get("primary_lookalike") or 0),
            -(r.get("primary_remote_gene_agree") or 0),
            -(r.get("primary_collide") or 0),
        )
    )
    lines = [
        "# ST-Graph Audit Leaderboard — 谁在偷偷连错类？",
        "",
        "> 按 **remote → lookalike → remote_gene_agree → collide** 降序。",
        "",
        "| rank | slide | method | collide | look | remote | rga | spatial_look | skip_rule |",
        "|------|-------|--------|---------|------|--------|-----|--------------|-----------|",
    ]
    for i, r in enumerate(ok, 1):
        lines.append(
            f"| {i} | {r.get('slide')} | {r.get('method')} | {r.get('primary_collide')} | "
            f"{r.get('primary_lookalike')} | {r.get('primary_remote')} | "
            f"{r.get('primary_remote_gene_agree', 0)} | {r.get('primary_spatial_lookalike', 0)} | "
            f"{fmt_cell(r.get('skip_rule_on'))} |"
        )
    path.write_text("\n".join(lines) + f"\n\n_generated {_now_iso()}_\n", encoding="utf-8")


def plot_bars(rows: list[dict], out: Path, title: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if "error" not in r]
    methods = [r["method"] for r in ok]
    collide = [r.get("primary_collide") or 0 for r in ok]
    look = [r.get("primary_lookalike") or 0 for r in ok]
    remote = [r.get("primary_remote") or 0 for r in ok]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, max(4.0, 0.35 * len(methods))), sharey=True)
    specs = [
        (axes[0], collide, "primary collide", "#ef4444"),
        (axes[1], look, "lookalike collide", "#f59e0b"),
        (axes[2], remote, "remote collide (nn×≥3)", "#2563eb"),
    ]
    x = np.arange(len(methods))
    for ax, vals, xlab, color in specs:
        ax.barh(x, vals, color=color, alpha=0.85)
        ax.set_xlabel(xlab)
        for i, v in enumerate(vals):
            ax.text(v + max(vals + [1]) * 0.02, i, str(v), va="center", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(methods, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_extreme_panel(df: pd.DataFrame, out: Path) -> None:
    slides = list(df["slide"].unique())
    fig, axes = plt.subplots(1, len(slides), figsize=(4.2 * len(slides), 4.8), sharey=False)
    if len(slides) == 1:
        axes = [axes]
    for ax, slide in zip(axes, slides):
        sub = df[df["slide"] == slide]
        methods = sub["method"].tolist()
        remote = sub["primary_remote"].fillna(0).astype(int).tolist()
        look = sub["primary_lookalike"].fillna(0).astype(int).tolist()
        x = np.arange(len(methods))
        w = 0.35
        ax.barh(x - w / 2, remote, w, color="#2563eb", label="remote")
        ax.barh(x + w / 2, look, w, color="#f59e0b", label="lookalike")
        ax.set_yticks(x)
        ax.set_yticklabels([m.replace(" (", "\n(") for m in methods], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{slide}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if slide == slides[0]:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("ST-Graph Audit extreme v0.7", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_iqr_appendix() -> dict:
    out: dict[str, Any] = {"sources": [], "rows": []}
    for fname, parser in (
        ("variance_reduction_stability.json", _parse_vr),
        ("gnn_training_seed_stability.json", _parse_gnn),
        ("variance_extreme.json", _parse_ve),
    ):
        p = TAB / fname
        if p.is_file():
            out["sources"].append(str(p.relative_to(ROOT)))
            out["rows"].extend(parser(json.loads(p.read_text(encoding="utf-8"))))
    return out


def _parse_vr(data: dict) -> list[dict]:
    return [
        {
            "axis": "mclust_seed_on_frozen_emb",
            "arm": row.get("arm"),
            "median": row.get("median"),
            "iqr": row.get("iqr"),
            "span": row.get("span"),
        }
        for row in data.get("stability_same_embedding", [])
    ]


def _parse_gnn(data: dict) -> list[dict]:
    rows = []
    for arm, block in data.get("summary", {}).items():
        for metric, st in block.items():
            rows.append(
                {
                    "axis": f"gnn_training_seed/{metric}",
                    "arm": arm,
                    "median": st.get("median"),
                    "iqr": st.get("iqr"),
                    "span": st.get("span"),
                }
            )
    return rows


def _parse_ve(data: dict) -> list[dict]:
    return [
        {
            "axis": f"emb_fusion_loo/{metric}",
            "arm": "gated_mean_l2",
            "median": st.get("median"),
            "iqr": st.get("iqr"),
            "span": st.get("span"),
        }
        for metric, st in data.get("loo_mean_l2", {}).items()
    ]


def write_seed_harm_table(seed_rows: list[dict], path: Path) -> None:
    lines = [
        "# Graph-Harm 5-Seed IQR（图拓扑层，非聚类 ARI）",
        "",
        "| slide | method | metric | median | IQR | span | seed values |",
        "|-------|--------|--------|--------|-----|------|-------------|",
    ]
    for r in seed_rows:
        for metric in ("lookalike", "remote", "collide"):
            st = r.get(metric, {})
            lines.append(
                f"| {r.get('slide')} | {r.get('method')} | {metric} | "
                f"{st.get('median')} | {st.get('iqr')} | {st.get('span')} | {st.get('values')} |"
            )
    path.write_text("\n".join(lines) + f"\n\n_generated {_now_iso()}_\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public runners
# ---------------------------------------------------------------------------

def run_block_a_benchmark(*, export_edges: bool = True) -> list[dict]:
    setup_env()
    slides = load_slide_registry()
    spec = next(s for s in slides if s["id"] == "BlockA")
    rows, seed_rows = audit_slide(spec, export_edges=export_edges, seed_harm=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "st_graph_audit_benchmark.csv", index=False)
    payload = {
        "title": "ST-Graph Audit Benchmark",
        "version": "1.0",
        "dataset": "Human_Breast_Cancer (Block A)",
        "time": _now_iso(),
        "freeze_note": "gene-gated mclust 0.617 unchanged",
        "rows": rows,
        "graph_harm_seeds": seed_rows,
    }
    (TAB / "st_graph_audit_benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "method", "n_edges", "n_idc_dcis", "collision_rate", "lookalike_idc_dcis_n",
        "remote_idc_dcis_n", "remote_gene_agree_idc_dcis_n", "spatial_lookalike_idc_dcis_n",
        "skip_rule_on", "inconsistency_idc_dcis_median",
        "gene_cos_idc_dcis_median", "fm_cos_idc_dcis_median", "nn_mult_idc_dcis_median", "notes",
    ]
    show = df[[c for c in cols if c in df.columns]]
    lines = [
        "# ST-Graph Audit Benchmark（Block A · v0.7）",
        "",
        "> Spec: `st_graph_audit_spec.md` · Gate 边界: `st_graph_audit_gate_boundary.md`",
        "",
        "## 谁在偷偷连错类？",
        "",
        "| method | n_edges | IDC–DCIS | lookalike | remote | rga | spatial_look | skip_rule | notes |",
        "|--------|---------|----------|-----------|--------|-----|--------------|-----------|-------|",
    ]
    for _, r in show.iterrows():
        lines.append(
            f"| {r['method']} | {int(r['n_edges'])} | {int(r['n_idc_dcis'])} | "
            f"{_int0(r.get('lookalike_idc_dcis_n'))} | {_int0(r.get('remote_idc_dcis_n'))} | "
            f"{_int0(r.get('remote_gene_agree_idc_dcis_n'))} | "
            f"{_int0(r.get('spatial_lookalike_idc_dcis_n'))} | "
            f"{fmt_cell(r.get('skip_rule_on'))} | {r.get('notes', '')} |"
        )
    lines += ["", f"_generated {payload['time']}_"]
    (TAB / "st_graph_audit_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    plot_bars(rows, FIG / "st_graph_audit_bars.png", "ST-Graph Audit Benchmark · Block A v0.7")
    write_seed_harm_table(seed_rows, TAB / "st_graph_audit_harm_seeds.md")
    (TAB / "st_graph_audit_harm_seeds.json").write_text(
        json.dumps({"time": payload["time"], "rows": seed_rows}, indent=2), encoding="utf-8"
    )
    return rows


def run_extreme(*, export_edges: bool = True) -> list[dict]:
    setup_env()
    all_rows: list[dict] = []
    all_seeds: list[dict] = []
    for spec in load_slide_registry():
        try:
            rows, seeds = audit_slide(spec, export_edges=export_edges, seed_harm=True)
            all_rows.extend(rows)
            all_seeds.extend(seeds)
        except Exception as e:
            all_rows.append({"slide": spec["id"], "method": "(slide)", "error": str(e)})
    iqr = load_iqr_appendix()
    df = pd.DataFrame(all_rows)
    df.to_csv(TAB / "st_graph_audit_extreme.csv", index=False)
    payload = {
        "title": "ST-Graph Audit Extreme",
        "version": "1.0",
        "time": _now_iso(),
        "freeze_note": "0.617 unchanged",
        "rows": all_rows,
        "graph_harm_seeds": all_seeds,
        "iqr_appendix": iqr,
    }
    (TAB / "st_graph_audit_extreme.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# ST-Graph Audit Extreme（多切片 × 多方法 · v1.0）",
        "",
        "> Spec · Gate 边界 · Leaderboard · Pair matrix · Repair Δ",
        "",
        "| slide | method | collide | lookalike | remote | rga | spatial_look | skip_rule | notes |",
        "|-------|--------|---------|-----------|--------|-----|--------------|-----------|-------|",
    ]
    for _, r in df.iterrows():
        if pd.notna(r.get("error")):
            lines.append(f"| {r.get('slide')} | {r.get('method')} | ERR | — | — | — | — | — | {r.get('error')} |")
            continue
        lines.append(
            f"| {r.get('slide')} | {r.get('method')} | "
            f"{_int0(r.get('primary_collide'))} | {_int0(r.get('primary_lookalike'))} | "
            f"{_int0(r.get('primary_remote'))} | {_int0(r.get('primary_remote_gene_agree'))} | "
            f"{_int0(r.get('primary_spatial_lookalike'))} | "
            f"{fmt_cell(r.get('skip_rule_on'))} | {r.get('notes', '')} |"
        )
    lines += ["", f"_generated {payload['time']}_"]
    (TAB / "st_graph_audit_extreme.md").write_text("\n".join(lines), encoding="utf-8")
    write_leaderboard(all_rows, TAB / "st_graph_audit_leaderboard.md")
    write_seed_harm_table(all_seeds, TAB / "st_graph_audit_harm_seeds.md")
    ok = df[df.apply(lambda r: pd.isna(r.get("error")), axis=1)]
    if len(ok):
        plot_extreme_panel(ok, FIG / "st_graph_audit_extreme_panel.png")

    from stamp.graph_audit_deep import run_v03_deep

    run_v03_deep(all_rows)

    from stamp.graph_audit_supplement import run_a_supplement

    run_a_supplement(all_rows)

    from stamp.graph_audit_residual_deep import run_residual_deep

    run_residual_deep()

    from stamp.graph_audit_remote_cap import run_remote_cap_v07

    run_remote_cap_v07()

    from stamp.graph_audit_decision import run_decision_v08

    run_decision_v08(all_rows)

    from stamp.graph_audit_budget import run_budget_v09

    run_budget_v09(all_rows)

    from stamp.graph_audit_certify import run_certify_v10

    run_certify_v10(live=False)
    return all_rows


def run_full_suite(*, live_certify: bool = False) -> int:
    """Track A one-shot: v0.1 benchmark → v1.0 certify (+ verify). Single entry point."""
    import importlib.util

    setup_env()
    stages = [
        "Block A benchmark (v0.1)",
        "Extreme multi-slide (v0.2)",
        "Deep pair/threshold/repair (v0.3)",
        "Supplement bridge/residuals (v0.4)",
        "Residual deep + spatial (v0.5)",
        "Remote cap sweep (v0.7)",
        "Decision protocol (v0.8)",
        "Harm budget + audit_config (v0.9)",
        f"Certify ({'live' if live_certify else 'artifact'}) (v1.0)",
        "Second Visium screen",
        "Verify deliverables",
    ]
    print("[Track A] ONE-SHOT — " + " · ".join(stages[:3]) + " · …", flush=True)
    print("[Track A] 1/4 Block A benchmark …", flush=True)
    run_block_a_benchmark(export_edges=True)
    print("[Track A] 2/4 Extreme stack (v0.2–v0.9 embedded in run_extreme) …", flush=True)
    run_extreme(export_edges=True)
    if live_certify:
        print("[Track A] 3/4 Live certify (re-score recommended arms) …", flush=True)
        from stamp.graph_audit_certify import run_certify_v10

        run_certify_v10(live=True)
    print("[Track A] 4/4 Second Visium screen …", flush=True)
    spec = importlib.util.spec_from_file_location(
        "run_second_visium_screen", ROOT / "scripts" / "run_second_visium_screen.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()
    ok, report = verify_track_a()
    print(report, flush=True)
    if ok:
        print(
            "[Track A] COMPLETE — one command produced all tables/figs/config/certificate.",
            flush=True,
        )
    return 0 if ok else 1


TRACK_A_TABLES = (
    "st_graph_audit_spec.md",
    "st_graph_audit_benchmark.md",
    "st_graph_audit_extreme.md",
    "st_graph_audit_leaderboard.md",
    "st_graph_audit_gate_boundary.md",
    "st_graph_audit_harm_seeds.md",
    "st_graph_audit_pair_matrix.md",
    "st_graph_audit_threshold_sweep.md",
    "st_graph_audit_effective_graph.md",
    "st_graph_audit_repair_delta.md",
    "st_graph_audit_fm_graph_bridge.md",
    "st_graph_audit_gate_residuals.md",
    "st_graph_audit_cross_slide_harm.md",
    "st_graph_audit_skip_delta.md",
    "st_graph_audit_residual_deep.md",
    "st_graph_audit_remote_cap_sweep.md",
    "st_graph_audit_tau_remote_sweep.md",
    "st_graph_audit_remote_cap_boundary.md",
    "st_graph_audit_decision.md",
    "st_graph_audit_harm_budget.md",
    "st_graph_audit_certificate.md",
    "st_graph_audit_harm_report.md",
    "second_visium_screen.md",
)

TRACK_A_FIGS = (
    "st_graph_audit_bars.png",
    "st_graph_audit_extreme_panel.png",
    "st_graph_audit_pair_matrix.png",
    "st_graph_audit_threshold_sweep.png",
    "st_graph_audit_repair_delta.png",
    "st_graph_audit_fm_graph_bridge.png",
    "st_graph_audit_remote_cap_sweep.png",
    "st_graph_audit_decision_tree.png",
    "st_graph_audit_decision_cards.png",
    "st_graph_audit_harm_budget.png",
)

CONFIG_YAML = ROOT / "config" / "st_graph_audit_config.yaml"


def verify_track_a() -> tuple[bool, str]:
    """Check all Track A v0.9 deliverables exist and extreme.json has max_hops arm."""
    missing: list[str] = []
    for name in TRACK_A_TABLES:
        if not (TAB / name).is_file():
            missing.append(f"table/{name}")
    for name in TRACK_A_FIGS:
        if not (FIG / name).is_file():
            missing.append(f"fig/{name}")
    edge_dir = TAB / "st_graph_audit_edges"
    if not edge_dir.is_dir() or not any(edge_dir.glob("*.csv")):
        missing.append("table/st_graph_audit_edges/*.csv")
    residual_dir = TAB / "st_graph_audit_gate_residuals"
    if not residual_dir.is_dir() or not any(residual_dir.glob("*.csv")):
        missing.append("table/st_graph_audit_gate_residuals/*.csv")

    if not CONFIG_YAML.is_file():
        missing.append("config/st_graph_audit_config.yaml")
    if not (TAB / "st_graph_audit_config.yaml").is_file():
        missing.append("table/st_graph_audit_config.yaml")

    if not (TAB / "st_graph_audit_certificate.json").is_file():
        missing.append("table/st_graph_audit_certificate.json")

    ext = TAB / "st_graph_audit_extreme.json"
    has_mh2 = False
    n_rows = 0
    if ext.is_file():
        payload = json.loads(ext.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        n_rows = len(rows)
        has_mh2 = any("max_hops=2" in str(r.get("method", "")) for r in rows)

    lines = [
        "==== Track A verify (v1.0) ====",
        f"tables OK: {len(TRACK_A_TABLES) - sum(1 for n in TRACK_A_TABLES if not (TAB / n).is_file())}/{len(TRACK_A_TABLES)}",
        f"figs OK: {len(TRACK_A_FIGS) - sum(1 for n in TRACK_A_FIGS if not (FIG / n).is_file())}/{len(TRACK_A_FIGS)}",
        f"extreme rows: {n_rows}",
        f"max_hops=2 arm in extreme: {'yes' if has_mh2 else 'no'}",
    ]
    if missing:
        lines.append(f"MISSING ({len(missing)}): " + ", ".join(missing))
        return False, "\n".join(lines)
    if not has_mh2:
        lines.append("WARN: extreme.json lacks max_hops=2 arm — re-run extreme")
        return False, "\n".join(lines)
    lines.append("ALL PASS")
    return True, "\n".join(lines)
