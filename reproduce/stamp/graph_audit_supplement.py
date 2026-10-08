"""ST-Graph Audit v0.4 supplements: A↔B bridge, gate residuals, cross-slide, skip delta."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stamp.edges import add_fm_skip_edges
from stamp.graph_audit.core import adj_to_undirected_pairs, score_graph
from stamp.graph_audit_benchmark import (
    FIG,
    TAB,
    _now_iso,
    fmt_cell,
    fm_key,
    load_slide_registry,
    load_staig,
    setup_env,
)

# slide_id (audit) -> fm atlas slide_label
SLIDE_FM_MAP = {
    "BlockA": "BlockA_pathologist",
    "BlockA_Sec2": "BlockA_Sec2_xfer",
    "CytAssist_Patho": "CytAssist_pathologist",
    "Wu": "Wu_CID44971",
}

RESIDUAL_DIR = TAB / "st_graph_audit_gate_residuals"


def _row(all_rows: list[dict], slide: str, needle: str) -> dict | None:
    for r in all_rows:
        if r.get("slide") == slide and needle in r.get("method", "") and "error" not in r:
            return r
    return None


def build_fm_graph_bridge(all_rows: list[dict]) -> list[dict]:
    atlas_path = TAB / "fm_lookalike_atlas.json"
    if not atlas_path.is_file():
        return []
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    fm_by_label = {
        (r["slide_label"], r["encoder"]): r
        for r in atlas.get("rows", [])
        if "error" not in r and r.get("encoder") == "phikonv2_emb"
    }
    bridge = []
    for slide_id, fm_label in SLIDE_FM_MAP.items():
        fm = fm_by_label.get((fm_label, "phikonv2_emb"))
        ung = _row(all_rows, slide_id, "ungated skip")
        gated = _row(all_rows, slide_id, "gene-gated")
        if not fm or not ung:
            continue
        fm_look = int(fm.get("idc_dcis_lookalike") or 0)
        g_look = int(ung.get("primary_lookalike") or 0)
        g_rem = int(ung.get("primary_remote") or 0)
        gate_look = int(gated.get("primary_lookalike") or 0) if gated else None
        gate_rem = int(gated.get("primary_remote") or 0) if gated else None
        bridge.append(
            {
                "slide": slide_id,
                "fm_label": fm_label,
                "fm_idc_dcis_lookalike": fm_look,
                "fm_idc_dcis_nn": int(fm.get("idc_dcis_fm_nn") or 0),
                "graph_ungated_lookalike": g_look,
                "graph_ungated_remote": g_rem,
                "graph_gated_lookalike": gate_look,
                "graph_gated_remote": gate_rem,
                "lookalike_graph_over_fm": round(g_look / max(fm_look, 1), 2),
                "remote_graph_over_fm": round(g_rem / max(fm_look, 1), 2),
                "skip_rule_on": ung.get("skip_rule_on"),
                "narrative": (
                    f"FM 1-NN lookalike {fm_look} → graph ungated {g_look}/{g_rem} (L/R); "
                    f"amplification {g_rem / max(fm_look, 1):.1f}× remote"
                ),
            }
        )
    return bridge


def analyze_gate_residuals(all_rows: list[dict]) -> list[dict]:
    """Edges surviving gene gate with lookalike or remote on primary collide."""
    setup_env()
    RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in load_slide_registry():
        if spec["primary"] != "idc_dcis" or not spec.get("gate"):
            continue
        gated_row = _row(all_rows, spec["id"], "gene-gated")
        if not gated_row:
            continue
        pl = gated_row.get("primary_lookalike") or 0
        pr = gated_row.get("primary_remote") or 0
        if pl == 0 and pr == 0:
            continue
        ref = load_staig(spec["path"], spec["kind"], spec.get("truth_file", "truth.txt"))
        fk = fm_key(ref)
        if not fk:
            continue
        g0 = np.array(ref.obsm["graph_neigh"], dtype=np.float64, copy=True)
        ep0 = np.array(ref.obsm["edge_probabilities"], dtype=np.float64, copy=True)
        ad_g = ref.copy()
        ad_g.obsm["graph_neigh"] = g0.copy()
        ad_g.obsm["edge_probabilities"] = ep0.copy()
        add_fm_skip_edges(
            ad_g, fk, k=1, drop_prob=0.05, pca_dim=16, gene_key="feat", gene_k=10, gene_mutual=True
        )
        _, edf = score_graph(
            ref,
            np.asarray(ad_g.obsm["graph_neigh"]),
            "STAMP gene-gated skip",
            img_key="img_emb" if "img_emb" in ref.obsm else None,
            fm_key=fk,
        )
        idc = edf["relation"] == "idc_dcis"
        harmful = edf[idc & ((edf["lookalike"] == 1) | (edf["remote"] == 1))].copy()
        out_csv = RESIDUAL_DIR / f"{spec['id']}_gate_residual_edges.csv"
        harmful.to_csv(out_csv, index=False)
        pair_agg = (
            harmful.assign(pair=harmful.apply(lambda r: "–".join(sorted([str(r["ci"]), str(r["cj"])])), axis=1))
            .groupby("pair", as_index=False)
            .agg(n=("pair", "count"), lookalike=("lookalike", "sum"), remote=("remote", "sum"))
            .sort_values("n", ascending=False)
        )
        summaries.append(
            {
                "slide": spec["id"],
                "residual_lookalike": int(pl),
                "residual_remote": int(pr),
                "n_harmful_edges": len(harmful),
                "top_pairs": pair_agg.head(5).to_dict(orient="records"),
                "csv": str(out_csv.relative_to(TAB.parent.parent)),
            }
        )
    return summaries


def skip_introduced_harm(spec: dict) -> dict | None:
    """New harmful edges in ungated vs spatial baseline."""
    ref = load_staig(spec["path"], spec["kind"], spec.get("truth_file", "truth.txt"))
    fk = fm_key(ref)
    if not fk:
        return None
    g0 = np.array(ref.obsm["graph_neigh"], dtype=np.float64, copy=True)
    ep0 = np.array(ref.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    _, edf_sp = score_graph(
        ref, g0, "spatial", img_key="img_emb" if "img_emb" in ref.obsm else None, fm_key=fk
    )
    ad_u = ref.copy()
    ad_u.obsm["graph_neigh"] = g0.copy()
    ad_u.obsm["edge_probabilities"] = ep0.copy()
    add_fm_skip_edges(ad_u, fk, k=1, drop_prob=0.05, pca_dim=16)
    _, edf_u = score_graph(
        ref,
        np.asarray(ad_u.obsm["graph_neigh"]),
        "ungated",
        img_key="img_emb" if "img_emb" in ref.obsm else None,
        fm_key=fk,
    )
    sp_pairs = set(zip(edf_sp["i"], edf_sp["j"]))
    new = edf_u[~edf_u.apply(lambda r: (r["i"], r["j"]) in sp_pairs or (r["j"], r["i"]) in sp_pairs, axis=1)]
    cross = new[new["relation"] != "same"]
    harmful = cross[(cross["lookalike"] == 1) | (cross["remote"] == 1)]
    pair_counts = (
        harmful.assign(pair=harmful.apply(lambda r: "–".join(sorted([str(r["ci"]), str(r["cj"])])), axis=1))
        .groupby("pair", as_index=False)
        .agg(n=("pair", "count"), lookalike=("lookalike", "sum"), remote=("remote", "sum"))
        .sort_values("n", ascending=False)
    )
    idc = harmful[harmful["relation"] == "idc_dcis"]
    return {
        "slide": spec["id"],
        "n_skip_new_edges": int(len(new)),
        "n_skip_new_cross": int(len(cross)),
        "n_skip_new_harmful": int(len(harmful)),
        "n_skip_new_idc_dcis_harmful": int(len(idc)),
        "top_harmful_pairs": pair_counts.head(8).to_dict(orient="records"),
    }


def cross_slide_harm_consistency(all_rows: list[dict]) -> list[dict]:
    """Harm ratios vs Block A reference (skip_rule ON breast slides)."""
    ref = _row(all_rows, "BlockA", "ungated skip")
    if not ref:
        return []
    ref_look = ref.get("primary_lookalike") or 1
    ref_rem = ref.get("primary_remote") or 1
    ref_col = ref.get("primary_collide") or 1
    rows = []
    for slide in ("BlockA", "BlockA_Sec2", "CytAssist_Patho", "Wu"):
        ung = _row(all_rows, slide, "ungated skip")
        if not ung:
            continue
        rows.append(
            {
                "slide": slide,
                "skip_rule_on": ung.get("skip_rule_on"),
                "lookalike": ung.get("primary_lookalike"),
                "remote": ung.get("primary_remote"),
                "collide": ung.get("primary_collide"),
                "lookalike_vs_blockA": round((ung.get("primary_lookalike") or 0) / ref_look, 2),
                "remote_vs_blockA": round((ung.get("primary_remote") or 0) / ref_rem, 2),
                "collide_vs_blockA": round((ung.get("primary_collide") or 0) / ref_col, 2),
                "is_reference": slide == "BlockA",
            }
        )
    return rows


def plot_bridge(bridge: list[dict], out: Path) -> None:
    if not bridge:
        return
    slides = [b["slide"] for b in bridge]
    x = np.arange(len(slides))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(slides)), 4.5))
    fm = [b["fm_idc_dcis_lookalike"] for b in bridge]
    gl = [b["graph_ungated_lookalike"] for b in bridge]
    gr = [b["graph_ungated_remote"] for b in bridge]
    ax.bar(x - w, fm, w, label="FM 1-NN lookalike", color="#a855f7")
    ax.bar(x, gl, w, label="graph ungated lookalike", color="#f59e0b")
    ax.bar(x + w, gr, w, label="graph ungated remote", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(slides, rotation=15, ha="right")
    ax.set_ylabel("IDC–DCIS count")
    ax.legend(fontsize=8)
    ax.set_title("A↔B bridge: FM lookalike → graph harm amplification")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_a_supplement(all_rows: list[dict]) -> dict[str, Any]:
    setup_env()
    TAB.mkdir(parents=True, exist_ok=True)

    bridge = build_fm_graph_bridge(all_rows)
    residuals = analyze_gate_residuals(all_rows)
    consistency = cross_slide_harm_consistency(all_rows)
    skip_deltas = []
    for spec in load_slide_registry():
        if spec.get("gate") and fm_key(load_staig(spec["path"], spec["kind"], spec.get("truth_file", "truth.txt"))):
            try:
                sd = skip_introduced_harm(spec)
                if sd:
                    skip_deltas.append(sd)
            except Exception:
                pass

    # --- FM↔Graph bridge ---
    blines = [
        "# ST-Graph Audit · A↔B 桥接（FM 1-NN → graph harm · v0.4）",
        "",
        "> FM 空间撞车（赛道 B）如何被 skip 放大为 graph remote/lookalike（赛道 A）",
        "",
        "| slide | FM lookalike | graph L | graph R | L/FM | R/FM | skip_rule | gated L/R |",
        "|-------|--------------|---------|---------|------|------|-----------|-----------|",
    ]
    for b in bridge:
        blines.append(
            f"| {b['slide']} | {b['fm_idc_dcis_lookalike']} | {b['graph_ungated_lookalike']} | "
            f"{b['graph_ungated_remote']} | {b['lookalike_graph_over_fm']}× | {b['remote_graph_over_fm']}× | "
            f"{fmt_cell(b.get('skip_rule_on'))} | "
            f"{b.get('graph_gated_lookalike')}/{b.get('graph_gated_remote')} |"
        )
    blines += [
        "",
        "**读法**：remote 放大倍数 >> lookalike → FM 撞车被 skip **几何拉远** 后变成 remote harm。",
        "",
        f"_generated {_now_iso()}_",
    ]
    (TAB / "st_graph_audit_fm_graph_bridge.md").write_text("\n".join(blines), encoding="utf-8")
    (TAB / "st_graph_audit_fm_graph_bridge.json").write_text(
        json.dumps({"time": _now_iso(), "rows": bridge}, indent=2), encoding="utf-8"
    )
    plot_bridge(bridge, FIG / "st_graph_audit_fm_graph_bridge.png")

    # --- Gate residuals ---
    rlines = [
        "# ST-Graph Audit · Gate 残留有害边（v0.4）",
        "",
        "gene gate 后仍 lookalike/remote 的 IDC–DCIS 边（应趋近 0）。",
        "",
    ]
    for r in residuals:
        rlines += [
            f"## {r['slide']}",
            "",
            f"- 残留 lookalike/remote（表级）：**{r['residual_lookalike']} / {r['residual_remote']}**",
            f"- 有害边数：**{r['n_harmful_edges']}** · CSV: `{r['csv']}`",
            "",
            "| pair | n | lookalike | remote |",
            "|------|---|-----------|--------|",
        ]
        for p in r.get("top_pairs", []):
            rlines.append(f"| {p['pair']} | {p['n']} | {p['lookalike']} | {p['remote']} |")
        rlines.append("")
    if not residuals:
        rlines.append("_所有 gate=ON 片 gate 后 primary harm 已清零。_")
    (TAB / "st_graph_audit_gate_residuals.md").write_text("\n".join(rlines), encoding="utf-8")

    # --- Cross-slide consistency ---
    clines = [
        "# ST-Graph Audit · 跨片 harm 一致性（vs Block A · v0.4）",
        "",
        "| slide | skip_rule | lookalike | remote | collide | L/BlockA | R/BlockA | C/BlockA |",
        "|-------|-----------|-----------|--------|---------|----------|----------|----------|",
    ]
    for c in consistency:
        clines.append(
            f"| {c['slide']} | {fmt_cell(c.get('skip_rule_on'))} | {c['lookalike']} | {c['remote']} | "
            f"{c['collide']} | {c['lookalike_vs_blockA']} | {c['remote_vs_blockA']} | {c['collide_vs_blockA']} |"
        )
    clines += ["", f"_generated {_now_iso()}_"]
    (TAB / "st_graph_audit_cross_slide_harm.md").write_text("\n".join(clines), encoding="utf-8")

    # --- Skip-introduced harmful pairs ---
    slines = [
        "# ST-Graph Audit · Skip 新增有害边（spatial 基线之外 · v0.4）",
        "",
        "| slide | new skip edges | new cross | new harmful | new IDC–DCIS harmful | top pair |",
        "|-------|----------------|-----------|-------------|----------------------|----------|",
    ]
    for sd in skip_deltas:
        top = sd["top_harmful_pairs"][0]["pair"] if sd["top_harmful_pairs"] else "—"
        slines.append(
            f"| {sd['slide']} | {sd['n_skip_new_edges']} | {sd['n_skip_new_cross']} | "
            f"{sd['n_skip_new_harmful']} | {sd['n_skip_new_idc_dcis_harmful']} | {top} |"
        )
    slines += ["", "## Top harmful pairs per slide", ""]
    for sd in skip_deltas:
        slines += [f"### {sd['slide']}", ""]
        for p in sd.get("top_harmful_pairs", []):
            slines.append(f"- **{p['pair']}**: n={p['n']}, lookalike={p['lookalike']}, remote={p['remote']}")
        slines.append("")
    (TAB / "st_graph_audit_skip_delta.md").write_text("\n".join(slines), encoding="utf-8")
    (TAB / "st_graph_audit_skip_delta.json").write_text(
        json.dumps({"time": _now_iso(), "rows": skip_deltas}, indent=2), encoding="utf-8"
    )

    return {
        "bridge": bridge,
        "residuals": residuals,
        "consistency": consistency,
        "skip_deltas": skip_deltas,
    }
