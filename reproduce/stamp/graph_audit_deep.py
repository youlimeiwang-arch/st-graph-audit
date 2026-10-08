"""ST-Graph Audit v0.3 deep analyses: pair matrix, threshold sweep, repair delta."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stamp.graph_audit.core import (
    TAU_FM,
    TAU_GENE,
    TAU_REMOTE,
    AuditThresholds,
    apply_edge_dropout,
    score_graph,
)
from stamp.graph_audit_benchmark import (
    FIG,
    ROOT,
    TAB,
    audit_slide,
    fm_key,
    load_slide_registry,
    setup_env,
    _now_iso,
    fmt_cell,
)
from stamp.edges import add_fm_skip_edges


def pair_collision_matrix(edf: pd.DataFrame, *, cross_only: bool = True) -> pd.DataFrame:
    """Count undirected coarse-pair edges; cross_only excludes same-class."""
    df = edf.copy()
    if cross_only:
        df = df[df["relation"] != "same"]
    if df.empty:
        return pd.DataFrame(columns=["pair", "n", "lookalike", "remote"])
    pairs = []
    for _, r in df.iterrows():
        a, b = sorted([str(r["ci"]), str(r["cj"])])
        pairs.append((a, b))
    out = df.copy()
    out["pair"] = [f"{a}–{b}" for a, b in pairs]
    agg = (
        out.groupby("pair", as_index=False)
        .agg(n=("pair", "count"), lookalike=("lookalike", "sum"), remote=("remote", "sum"))
        .sort_values("n", ascending=False)
    )
    return agg


def score_with_thresholds(adata, adj, method: str, tau_gene: float, tau_fm: float, tau_remote: float = TAU_REMOTE):
    thr = AuditThresholds(tau_gene=tau_gene, tau_fm=tau_fm, tau_remote=tau_remote)
    fk = fm_key(adata)
    summary, _ = score_graph(
        adata,
        adj,
        method,
        img_key="img_emb" if "img_emb" in adata.obsm else None,
        fm_key=fk,
        thresholds=thr,
        notes=f"τ_g={tau_gene}, τ_fm={tau_fm}",
    )
    return summary


def threshold_sweep_ungated(
    adata,
    fk: str,
    *,
    tau_gene_grid: list[float] | None = None,
    tau_fm_grid: list[float] | None = None,
) -> list[dict]:
    """Sweep lookalike thresholds on ungated FM skip graph."""
    tau_gene_grid = tau_gene_grid or [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    tau_fm_grid = tau_fm_grid or [0.70, 0.75, 0.80, 0.85, 0.90]
    g0 = np.array(adata.obsm["graph_neigh"], dtype=np.float64, copy=True)
    ep0 = np.array(adata.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    ad_u = adata.copy()
    ad_u.obsm["graph_neigh"] = g0.copy()
    ad_u.obsm["edge_probabilities"] = ep0.copy()
    add_fm_skip_edges(ad_u, fk, k=1, drop_prob=0.05, pca_dim=16)
    adj = np.asarray(ad_u.obsm["graph_neigh"], dtype=np.float64)

    rows = []
    for tg in tau_gene_grid:
        for tf in tau_fm_grid:
            s = score_with_thresholds(ad_u, adj, "ungated_skip", tg, tf)
            rows.append(
                {
                    "tau_gene": tg,
                    "tau_fm": tf,
                    "lookalike_idc_dcis": s.get("lookalike_idc_dcis_n", 0),
                    "remote_idc_dcis": s.get("remote_idc_dcis_n", 0),
                    "n_idc_dcis": s.get("n_idc_dcis", 0),
                    "is_frozen": tg == TAU_GENE and tf == TAU_FM,
                }
            )
    return rows


def effective_graph_harm_median(
    adata,
    *,
    primary: str = "idc_dcis",
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict:
    """STAIG spatial + img-dropout effective graph; median harm over seeds."""
    if primary == "cross":
        k_look, k_remote, k_collide = "lookalike_cross_n", "remote_cross_n", "n_cross_coarse"
    else:
        k_look, k_remote, k_collide = "lookalike_idc_dcis_n", "remote_idc_dcis_n", "n_idc_dcis"
    g0 = np.array(adata.obsm["graph_neigh"], dtype=np.float64, copy=True)
    ep0 = np.array(adata.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    fk = fm_key(adata)
    looks, remotes, collides, n_edges = [], [], [], []
    for seed in seeds:
        adj = apply_edge_dropout(g0, ep0, seed)
        s, _ = score_graph(
            adata,
            adj,
            f"STAIG effective seed={seed}",
            img_key="img_emb" if "img_emb" in adata.obsm else None,
            fm_key=fk,
        )
        looks.append(s.get(k_look) or 0)
        remotes.append(s.get(k_remote) or 0)
        collides.append(s.get(k_collide) or 0)
        n_edges.append(s.get("n_edges") or 0)

    def med(xs):
        return float(np.median(xs))

    return {
        "method": "STAIG img-dropout effective (5-seed median)",
        "notes": "spatial graph after img edge_probabilities dropout; median over 5 seeds",
        "n_edges_median": med(n_edges),
        "lookalike_median": med(looks),
        "remote_median": med(remotes),
        "collide_median": med(collides),
        "seed_values": {
            "lookalike": looks,
            "remote": remotes,
            "collide": collides,
            "n_edges": n_edges,
        },
    }


def compute_repair_delta(all_rows: list[dict]) -> list[dict]:
    """Per slide: Δ(ungated−spatial), Δ(ungated−gated), gate repair rate."""
    out = []
    slides = sorted({r.get("slide") for r in all_rows if r.get("slide")})
    for slide in slides:
        sub = [r for r in all_rows if r.get("slide") == slide and "error" not in r]
        spatial = next((r for r in sub if r.get("method") == "STAIG spatial (k=5)"), None)
        gated = next((r for r in sub if "gene-gated" in r.get("method", "")), None)
        ungated = next((r for r in sub if "ungated skip" in r.get("method", "")), None)
        if not ungated:
            continue
        pl = ungated.get("primary_lookalike") or 0
        pr = ungated.get("primary_remote") or 0
        pc = ungated.get("primary_collide") or 0
        sl = spatial.get("primary_lookalike") or 0 if spatial else 0
        sr = spatial.get("primary_remote") or 0 if spatial else 0
        sc = spatial.get("primary_collide") or 0 if spatial else 0
        gl = gated.get("primary_lookalike") if gated else None
        gr = gated.get("primary_remote") if gated else None
        gc = gated.get("primary_collide") if gated else None
        repair_look = (pl - gl) if gl is not None else None
        repair_remote = (pr - gr) if gr is not None else None
        repair_collide = (pc - gc) if gc is not None else None
        out.append(
            {
                "slide": slide,
                "skip_rule_on": ungated.get("skip_rule_on"),
                "spatial_collide": sc,
                "ungated_collide": pc,
                "gated_collide": gc,
                "delta_ungated_minus_spatial_collide": pc - sc,
                "delta_ungated_minus_spatial_lookalike": pl - sl,
                "delta_ungated_minus_spatial_remote": pr - sr,
                "repair_lookalike": repair_look,
                "repair_remote": repair_remote,
                "repair_collide": repair_collide,
                "gate_clears_lookalike": gl == 0 if gl is not None else None,
                "gate_clears_remote": gr == 0 if gr is not None else None,
            }
        )
    return out


def plot_pair_heatmaps(pair_by_slide: dict[str, pd.DataFrame], out: Path) -> None:
    slides = list(pair_by_slide.keys())
    n = len(slides)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), squeeze=False)
    for ax, slide in zip(axes[0], slides):
        df = pair_by_slide[slide]
        if df.empty:
            ax.set_title(slide)
            ax.text(0.5, 0.5, "no cross pairs", ha="center", va="center", transform=ax.transAxes)
            continue
        top = df.head(12)
        y = np.arange(len(top))
        ax.barh(y, top["n"].values, color="#ef4444", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(top["pair"].tolist(), fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("cross-coarse edges")
        ax.set_title(f"{slide}\nungated skip", fontsize=9)
        for i, v in enumerate(top["n"].values):
            ax.text(v + 0.5, i, str(int(v)), va="center", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("ST-Graph Audit v0.3 · coarse-pair collision (ungated skip)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(sweep_by_slide: dict[str, list[dict]], out: Path) -> None:
    slides = list(sweep_by_slide.keys())
    fig, axes = plt.subplots(1, len(slides), figsize=(5 * len(slides), 4.5), squeeze=False)
    for ax, slide in zip(axes[0], slides):
        rows = sweep_by_slide[slide]
        df = pd.DataFrame(rows)
        # heatmap: rows tau_gene, cols tau_fm, values lookalike_idc_dcis
        pivot = df.pivot(index="tau_gene", columns="tau_fm", values="lookalike_idc_dcis")
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y:.2f}" for y in pivot.index], fontsize=8)
        ax.set_xlabel("τ_fm")
        ax.set_ylabel("τ_gene")
        ax.set_title(slide, fontsize=10)
        # mark frozen point
        if TAU_GENE in pivot.index and TAU_FM in pivot.columns:
            j = list(pivot.columns).index(TAU_FM)
            i = list(pivot.index).index(TAU_GENE)
            ax.scatter([j], [i], s=120, facecolors="none", edgecolors="blue", linewidths=2)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="lookalike IDC–DCIS")
    fig.suptitle("Threshold sweep · ungated skip · lookalike IDC–DCIS", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_repair_delta(deltas: list[dict], out: Path) -> None:
    if not deltas:
        return
    slides = [d["slide"] for d in deltas]
    x = np.arange(len(slides))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(slides)), 4.5))
    d_sp_l = [d["delta_ungated_minus_spatial_lookalike"] for d in deltas]
    d_sp_r = [d["delta_ungated_minus_spatial_remote"] for d in deltas]
    rep_l = [d["repair_lookalike"] if d["repair_lookalike"] is not None else 0 for d in deltas]
    rep_r = [d["repair_remote"] if d["repair_remote"] is not None else 0 for d in deltas]
    ax.bar(x - 1.5 * w, d_sp_l, w, label="Δ ungated−spatial lookalike", color="#f59e0b")
    ax.bar(x - 0.5 * w, d_sp_r, w, label="Δ ungated−spatial remote", color="#2563eb")
    ax.bar(x + 0.5 * w, rep_l, w, label="repair lookalike (ungated→gated)", color="#10b981")
    ax.bar(x + 1.5 * w, rep_r, w, label="repair remote (ungated→gated)", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(slides, rotation=20, ha="right")
    ax.set_ylabel("edge count")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("ST-Graph Audit v0.3 · repair delta")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_effective_graph_row(adata, spec: dict, eff: dict) -> dict:
    """Turn effective_graph_harm_median output into audit summary row."""
    primary = spec["primary"]
    row = {
        "method": eff["method"],
        "notes": eff["notes"],
        "n_edges": int(round(eff["n_edges_median"])),
        "slide": spec["id"],
        "slide_role": spec["role"],
        "label_note": spec.get("label_note", ""),
    }
    if primary == "cross":
        row["primary"] = "cross"
        row["primary_collide"] = int(round(eff["collide_median"]))
        row["primary_lookalike"] = int(round(eff["lookalike_median"]))
        row["primary_remote"] = int(round(eff["remote_median"]))
        row["n_cross_coarse"] = row["primary_collide"]
        row["lookalike_cross_n"] = row["primary_lookalike"]
        row["remote_cross_n"] = row["primary_remote"]
        row["collision_rate_primary"] = row["primary_collide"] / max(row["n_edges"], 1)
        row["skip_rule_on"] = None
    else:
        row["primary"] = "idc_dcis"
        row["primary_collide"] = int(round(eff["collide_median"]))
        row["primary_lookalike"] = int(round(eff["lookalike_median"]))
        row["primary_remote"] = int(round(eff["remote_median"]))
        row["n_idc_dcis"] = row["primary_collide"]
        row["lookalike_idc_dcis_n"] = row["primary_lookalike"]
        row["remote_idc_dcis_n"] = row["primary_remote"]
        row["collision_rate_primary"] = row["primary_collide"] / max(row["n_edges"], 1)
        row["skip_rule_on"] = None
    row["seed_values"] = eff["seed_values"]
    return row


def run_v03_deep(all_rows: list[dict] | None = None) -> dict[str, Any]:
    """Run all v0.3 deep analyses; optionally refresh extreme rows first."""
    setup_env()
    TAB.mkdir(parents=True, exist_ok=True)

    if all_rows is None:
        all_rows = []
        for spec in load_slide_registry():
            rows, _ = audit_slide(spec, export_edges=False, seed_harm=False)
            all_rows.extend(rows)

    # --- 1) Pair matrices from ungated skip edge tables (re-audit with edges) ---
    pair_by_slide: dict[str, pd.DataFrame] = {}
    effective_rows: list[dict] = []
    for spec in load_slide_registry():
        path = spec["path"]
        truth_file = spec.get("truth_file", "truth.txt")
        from stamp.graph_audit_benchmark import load_staig, load_raw, ensure_feat

        ref = load_staig(path, spec["kind"], truth_file)
        raw = load_raw(path, spec["kind"], truth_file)
        names = [b for b in ref.obs_names if b in raw.obs_names]
        ref = ref[names].copy()
        ensure_feat(raw)
        if "feat" not in ref.obsm:
            ref.obsm["feat"] = raw.obsm["feat"]
        fk = fm_key(ref)
        if fk:
            g0 = np.array(ref.obsm["graph_neigh"], dtype=np.float64, copy=True)
            ep0 = np.array(ref.obsm["edge_probabilities"], dtype=np.float64, copy=True)
            ad_u = ref.copy()
            ad_u.obsm["graph_neigh"] = g0.copy()
            ad_u.obsm["edge_probabilities"] = ep0.copy()
            add_fm_skip_edges(ad_u, fk, k=1, drop_prob=0.05, pca_dim=16)
            _, edf = score_graph(ref, np.asarray(ad_u.obsm["graph_neigh"]), f"ungated ({fk})", fm_key=fk)
            pair_by_slide[spec["id"]] = pair_collision_matrix(edf)
        if "edge_probabilities" in ref.obsm:
            eff = effective_graph_harm_median(ref, primary=spec.get("primary", "idc_dcis"))
            effective_rows.append(build_effective_graph_row(ref, spec, eff))

    # merge effective graph rows into all_rows if not present
    existing = {(r.get("slide"), r.get("method")) for r in all_rows}
    for er in effective_rows:
        key = (er.get("slide"), er.get("method"))
        if key not in existing:
            all_rows.append(er)

    # --- 2) Threshold sweep Block A + CytAssist ---
    sweep_by_slide: dict[str, list[dict]] = {}
    for slide_id in ("BlockA", "CytAssist_Patho"):
        spec = next(s for s in load_slide_registry() if s["id"] == slide_id)
        from stamp.graph_audit_benchmark import load_staig

        ref = load_staig(spec["path"], spec["kind"], spec.get("truth_file", "truth.txt"))
        fk = fm_key(ref)
        if fk:
            sweep_by_slide[slide_id] = threshold_sweep_ungated(ref, fk)

    # --- 3) Repair delta ---
    deltas = compute_repair_delta(all_rows)

    # --- Write outputs ---
    pair_lines = ["# ST-Graph Audit · 粗类对撞车矩阵（ungated skip · v0.3）", ""]
    pair_payload = {"time": _now_iso(), "slides": {}}
    for slide, df in pair_by_slide.items():
        pair_payload["slides"][slide] = df.to_dict(orient="records")
        pair_lines += [f"## {slide}", "", "| pair | n | lookalike | remote |", "|------|---|-----------|--------|"]
        for _, r in df.head(15).iterrows():
            pair_lines.append(f"| {r['pair']} | {int(r['n'])} | {int(r['lookalike'])} | {int(r['remote'])} |")
        pair_lines.append("")
    (TAB / "st_graph_audit_pair_matrix.md").write_text("\n".join(pair_lines), encoding="utf-8")
    (TAB / "st_graph_audit_pair_matrix.json").write_text(json.dumps(pair_payload, indent=2), encoding="utf-8")
    plot_pair_heatmaps(pair_by_slide, FIG / "st_graph_audit_pair_matrix.png")

    sweep_lines = [
        "# ST-Graph Audit · 阈值敏感性 sweep（ungated skip · v0.3）",
        "",
        f"冻结点：**τ_gene={TAU_GENE}, τ_fm={TAU_FM}**（蓝框）",
        "",
    ]
    sweep_payload = {"time": _now_iso(), "frozen": {"tau_gene": TAU_GENE, "tau_fm": TAU_FM}, "slides": sweep_by_slide}
    for slide, rows in sweep_by_slide.items():
        sweep_lines += [f"## {slide}", "", "| τ_gene | τ_fm | lookalike | remote | idc_dcis | frozen |", "|--------|------|-----------|--------|----------|--------|"]
        for r in rows:
            sweep_lines.append(
                f"| {r['tau_gene']} | {r['tau_fm']} | {r['lookalike_idc_dcis']} | {r['remote_idc_dcis']} | "
                f"{r['n_idc_dcis']} | {'✓' if r['is_frozen'] else ''} |"
            )
        frozen = next(r for r in rows if r["is_frozen"])
        sweep_lines += [
            "",
            f"**冻结点 lookalike={frozen['lookalike_idc_dcis']}**；"
            f"τ_gene±0.05 内 lookalike 范围 "
            f"{min(x['lookalike_idc_dcis'] for x in rows if abs(x['tau_gene']-TAU_GENE)<=0.05)}–"
            f"{max(x['lookalike_idc_dcis'] for x in rows if abs(x['tau_gene']-TAU_GENE)<=0.05)}。",
            "",
        ]
    (TAB / "st_graph_audit_threshold_sweep.md").write_text("\n".join(sweep_lines), encoding="utf-8")
    (TAB / "st_graph_audit_threshold_sweep.json").write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")
    plot_threshold_sweep(sweep_by_slide, FIG / "st_graph_audit_threshold_sweep.png")

    eff_lines = [
        "# ST-Graph Audit · STAIG img-dropout 有效图臂（5-seed median · v0.3）",
        "",
        "> 空间邻域经 `edge_probabilities` 随机 dropout 后的 **实际 message-passing 图** harm 中位数。",
        "",
        "| slide | n_edges | collide | lookalike | remote | seed lookalike | seed remote |",
        "|-------|---------|---------|-----------|--------|----------------|-------------|",
    ]
    for er in effective_rows:
        sv = er.get("seed_values", {})
        eff_lines.append(
            f"| {er['slide']} | {er['n_edges']} | {er['primary_collide']} | {er['primary_lookalike']} | "
            f"{er['primary_remote']} | {sv.get('lookalike')} | {sv.get('remote')} |"
        )
    eff_lines += [
        "",
        "**读法**：有效图 remote/lookalike 远低于 ungated FM skip → 主害在 skip 层，非原生 img 丢边。",
        "",
        f"_generated {_now_iso()}_",
    ]
    (TAB / "st_graph_audit_effective_graph.md").write_text("\n".join(eff_lines), encoding="utf-8")
    (TAB / "st_graph_audit_effective_graph.json").write_text(
        json.dumps({"time": _now_iso(), "rows": effective_rows}, indent=2), encoding="utf-8"
    )

    delta_lines = [
        "# ST-Graph Audit · Repair Δ（ungated→gated · v0.3）",
        "",
        "| slide | skip_rule | Δ collide (U−S) | Δ lookalike (U−S) | Δ remote (U−S) | repair look | repair remote | gate clears L/R |",
        "|-------|-----------|-----------------|-------------------|----------------|-------------|---------------|-----------------|",
    ]
    for d in deltas:
        delta_lines.append(
            f"| {d['slide']} | {fmt_cell(d.get('skip_rule_on'))} | {d['delta_ungated_minus_spatial_collide']} | "
            f"{d['delta_ungated_minus_spatial_lookalike']} | {d['delta_ungated_minus_spatial_remote']} | "
            f"{fmt_cell(d.get('repair_lookalike'), nd=0)} | {fmt_cell(d.get('repair_remote'), nd=0)} | "
            f"{fmt_cell(d.get('gate_clears_lookalike'))}/{fmt_cell(d.get('gate_clears_remote'))} |"
        )
    delta_lines += ["", f"_generated {_now_iso()}_"]
    (TAB / "st_graph_audit_repair_delta.md").write_text("\n".join(delta_lines), encoding="utf-8")
    (TAB / "st_graph_audit_repair_delta.json").write_text(
        json.dumps({"time": _now_iso(), "rows": deltas}, indent=2), encoding="utf-8"
    )
    plot_repair_delta(deltas, FIG / "st_graph_audit_repair_delta.png")

    return {
        "pair_by_slide": {k: v.to_dict(orient="records") for k, v in pair_by_slide.items()},
        "sweep_by_slide": sweep_by_slide,
        "effective_rows": effective_rows,
        "repair_delta": deltas,
        "all_rows_extended": all_rows,
    }
