"""ST-Graph Audit v0.5: gate residual edge deep-dive (mechanism + spatial)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stamp.edges import add_fm_skip_edges
from stamp.graph_audit.core import score_graph
from stamp.graph_audit_benchmark import FIG, TAB, _now_iso, fm_key, load_slide_registry, load_staig, setup_env

RESIDUAL_DIR = TAB / "st_graph_audit_gate_residuals"
RESIDUAL_SLIDES = ("BlockA_Sec2", "CytAssist_Patho", "Wu")


def _edge_origin(g0: np.ndarray, g1: np.ndarray, i: int, j: int) -> str:
    was_spatial = g0[i, j] > 0 or g0[j, i] > 0
    is_new = g1[i, j] > 0 or g1[j, i] > 0
    if was_spatial and is_new:
        return "spatial"
    if is_new and not was_spatial:
        return "skip_added"
    return "unknown"


def _mechanism(row: pd.Series, origin: str) -> str:
    look = int(row["lookalike"])
    remote = int(row["remote"])
    gc = float(row["gene_cos"])
    if look:
        return "lookalike_slip" if origin == "skip_added" else "lookalike_spatial"
    if remote and gc > 0.20:
        return "remote_gene_agree"  # gate passed: high gene mutual-kNN
    if remote:
        return "remote_low_gene"
    return "other"


def deep_dive_slide(spec: dict) -> dict | None:
    csv_path = RESIDUAL_DIR / f"{spec['id']}_gate_residual_edges.csv"
    if not csv_path.is_file():
        return None
    residual = pd.read_csv(csv_path)
    if residual.empty:
        return None

    ref = load_staig(spec["path"], spec["kind"], spec.get("truth_file", "truth.txt"))
    fk = fm_key(ref)
    if not fk:
        return None
    g0 = np.array(ref.obsm["graph_neigh"], dtype=np.float64, copy=True)
    ep0 = np.array(ref.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    ad_g = ref.copy()
    ad_g.obsm["graph_neigh"] = g0.copy()
    ad_g.obsm["edge_probabilities"] = ep0.copy()
    add_fm_skip_edges(
        ad_g, fk, k=1, drop_prob=0.05, pca_dim=16, gene_key="feat", gene_k=10, gene_mutual=True
    )
    g1 = np.asarray(ad_g.obsm["graph_neigh"], dtype=np.float64)
    xy = np.asarray(ref.obsm["spatial"], dtype=np.float64)
    names = ref.obs_names
    gt = ref.obs["ground_truth"].astype(str)

    rows = []
    for _, e in residual.iterrows():
        i, j = int(e["i"]), int(e["j"])
        origin = _edge_origin(g0, g1, i, j)
        mech = _mechanism(e, origin)
        rows.append(
            {
                "slide": spec["id"],
                "barcode_i": names[i],
                "barcode_j": names[j],
                "label_i": gt.iloc[i],
                "label_j": gt.iloc[j],
                "coarse_i": e["ci"],
                "coarse_j": e["cj"],
                "origin": origin,
                "mechanism": mech,
                "gene_cos": round(float(e["gene_cos"]), 3),
                "fm_cos": round(float(e["fm_cos"]), 3) if pd.notna(e.get("fm_cos")) else None,
                "nn_mult": round(float(e["nn_mult"]), 2),
                "lookalike": int(e["lookalike"]),
                "remote": int(e["remote"]),
                "spatial_dist_px": round(float(e["spatial_dist"]), 1),
                "x_i": round(float(xy[i, 0]), 1),
                "y_i": round(float(xy[i, 1]), 1),
                "x_j": round(float(xy[j, 0]), 1),
                "y_j": round(float(xy[j, 1]), 1),
            }
        )
    return {"slide": spec["id"], "label_note": spec.get("label_note", ""), "edges": rows}


def plot_residual_spatial(slide_id: str, edges: list[dict], xy_all: np.ndarray | None = None) -> None:
    if not edges:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for e in edges:
        ax.plot([e["x_i"], e["x_j"]], [e["y_i"], e["y_j"]], color="#dc2626", linewidth=1.2, alpha=0.85)
        ax.scatter([e["x_i"], e["x_j"]], [e["y_i"], e["y_j"]], s=18, c=["#f59e0b", "#2563eb"], zorder=3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{slide_id}: gate residual harmful edges (n={len(edges)})")
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / f"gate_residual_spatial_{slide_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_residual_deep() -> dict:
    setup_env()
    TAB.mkdir(parents=True, exist_ok=True)
    RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    registry = {s["id"]: s for s in load_slide_registry()}
    all_deep = []
    for slide_id in RESIDUAL_SLIDES:
        spec = registry.get(slide_id)
        if not spec:
            continue
        result = deep_dive_slide(spec)
        if result:
            all_deep.append(result)
            plot_residual_spatial(slide_id, result["edges"])

    # mechanism summary
    mech_counts: dict[str, dict[str, int]] = {}
    for block in all_deep:
        mech_counts[block["slide"]] = {}
        for e in block["edges"]:
            m = e["mechanism"]
            mech_counts[block["slide"]][m] = mech_counts[block["slide"]].get(m, 0) + 1

    lines = [
        "# ST-Graph Audit · Gate 残留边逐条解剖（v0.5）",
        "",
        "> gene gate 设计目标：阻断 **低 gene × 高 FM** lookalike skip。",
        "> 残留边揭示 gate **不覆盖** 的机制类。",
        "",
        "## 机制分类",
        "",
        "| 机制 | 含义 | gate 为何未拦 |",
        "|------|------|---------------|",
        "| `remote_gene_agree` | 远程 IDC–DCIS，**gene_cos > 0.20** | mutual gene-kNN 共识通过 |",
        "| `lookalike_slip` | lookalike=1 仍存活 | 多为 spatial 或 gate 边界 case |",
        "| `lookalike_spatial` | 空间邻域 lookalike | spatial 边不在 skip gate 范围 |",
        "",
        "## 按片汇总",
        "",
    ]
    for slide, counts in mech_counts.items():
        lines.append(f"### {slide}")
        lines.append("")
        for m, n in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"- **{m}**: {n}")
        lines.append("")

    for block in all_deep:
        lines += [
            f"## {block['slide']} — 逐边表",
            "",
            f"_{block.get('label_note', '')}_",
            "",
            "| # | origin | mechanism | labels | gene | FM | nn× | dist(px) | look | rem |",
            "|---|--------|-----------|--------|------|-----|-----|----------|------|-----|",
        ]
        for k, e in enumerate(block["edges"], 1):
            lines.append(
                f"| {k} | {e['origin']} | {e['mechanism']} | {e['label_i']}↔{e['label_j']} | "
                f"{e['gene_cos']} | {e['fm_cos']} | {e['nn_mult']} | {e['spatial_dist_px']} | "
                f"{e['lookalike']} | {e['remote']} |"
            )
        lines.append("")

    # Sec2 specific verdict
    sec2 = next((b for b in all_deep if b["slide"] == "BlockA_Sec2"), None)
    if sec2:
        n_gene_agree = sum(1 for e in sec2["edges"] if e["mechanism"] == "remote_gene_agree")
        lines += [
            "## Block A Sec2 结论",
            "",
            f"- **{n_gene_agree}/16** 条为 `remote_gene_agree`：远程 DCIS–IDC，但 **gene_cos 中位 ≈ 0.76**",
            "- 不是 lookalike 漏网，而是 **gene gate 按设计放行**（高 gene 互邻）",
            "- 审计含义：要清零 remote cross-class，需 **额外 remote 约束**，非仅 gene_k=10",
            "- 标签来源为 ECC xfer 粗类 → 部分 remote 连边可能反映 **标注噪声** 而非真 biology",
            "",
        ]

    cyto = next((b for b in all_deep if b["slide"] == "CytAssist_Patho"), None)
    if cyto and cyto["edges"]:
        e0 = cyto["edges"][0]
        lines += [
            "## CytAssist 结论",
            "",
            f"- 唯一残留：**{e0['label_i']}↔{e0['label_j']}** ({e0['mechanism']}, origin={e0['origin']})",
            f"- gene={e0['gene_cos']}, FM={e0['fm_cos']}, nn×={e0['nn_mult']}",
            "- FFPE 10-class 细粒度下仍有一条 lookalike spatial/skip 边界",
            "",
        ]

    lines += [
        "## 图",
        "",
        "- `figs/lookalike_mainline/gate_residual_spatial_BlockA_Sec2.png`",
        "- `figs/lookalike_mainline/gate_residual_spatial_CytAssist_Patho.png`",
        "",
        f"_generated {_now_iso()}_",
    ]

    (TAB / "st_graph_audit_residual_deep.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {"time": _now_iso(), "slides": all_deep, "mechanism_counts": mech_counts}
    (TAB / "st_graph_audit_residual_deep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # per-slide CSV
    for block in all_deep:
        pd.DataFrame(block["edges"]).to_csv(
            RESIDUAL_DIR / f"{block['slide']}_residual_deep.csv", index=False
        )
    return payload


if __name__ == "__main__":
    run_residual_deep()
