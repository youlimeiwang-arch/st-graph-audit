"""ST-Graph Audit v0.7: remote / nn× cap sensitivity (Track A extension).

Audit-only arms — does not change freeze gene-gated mclust 0.617 training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stamp.edges import add_fm_skip_edges
from stamp.graph_audit.core import (
    TAU_REMOTE,
    AuditThresholds,
    score_graph,
)
from stamp.graph_audit_benchmark import (
    FIG,
    ROOT,
    TAB,
    _attach_primary,
    _now_iso,
    ensure_feat,
    fm_key,
    load_slide_registry,
    load_staig,
    setup_env,
)


def build_gated_skip_graph(
    ref,
    fk: str,
    *,
    max_hops: int = 0,
    min_hops: int = 0,
) -> np.ndarray:
    """Gene-gated FM skip on frozen spatial graph; optional spatial hop cap."""
    g0 = np.array(ref.obsm["graph_neigh"], dtype=np.float64, copy=True)
    ep0 = np.array(ref.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    ad = ref.copy()
    ad.obsm["graph_neigh"] = g0.copy()
    ad.obsm["edge_probabilities"] = ep0.copy()
    add_fm_skip_edges(
        ad,
        fk,
        k=1,
        drop_prob=0.05,
        pca_dim=16,
        gene_key="feat",
        gene_k=10,
        gene_mutual=True,
        min_hops=min_hops,
        max_hops=max_hops,
    )
    return np.asarray(ad.obsm["graph_neigh"])


def post_cap_remote_skip(
    ref,
    g0: np.ndarray,
    g1: np.ndarray,
    fk: str,
    *,
    nn_mult_cap: float = TAU_REMOTE,
) -> np.ndarray:
    """Counterfactual: drop skip-added IDC–DCIS edges with nn_mult >= cap."""
    g0a = (np.asarray(g0) > 0).astype(np.uint8)
    g1a = (np.asarray(g1) > 0).astype(np.uint8)
    skip_mask = (g1a > 0) & (g0a == 0)
    _, edf = score_graph(
        ref,
        g1,
        "post_cap_scratch",
        img_key="img_emb" if "img_emb" in ref.obsm else None,
        fm_key=fk,
    )
    out = np.asarray(g1, dtype=np.float64).copy()
    for _, r in edf.iterrows():
        i, j = int(r["i"]), int(r["j"])
        if not skip_mask[i, j]:
            continue
        if r.get("relation") != "idc_dcis":
            continue
        if float(r["nn_mult"]) >= nn_mult_cap:
            out[i, j] = 0.0
            out[j, i] = 0.0
    return out


def _primary_from_summary(summary: dict, spec: dict) -> dict[str, Any]:
    _attach_primary(summary, spec)
    return {
        "lookalike": summary.get("primary_lookalike") or 0,
        "remote": summary.get("primary_remote") or 0,
        "rga": summary.get("primary_remote_gene_agree") or 0,
        "spatial_look": summary.get("primary_spatial_lookalike") or 0,
        "collide": summary.get("primary_collide") or 0,
    }


def score_arm(ref, adj, spec: dict, fk: str, method: str, notes: str) -> dict:
    summary, _ = score_graph(
        ref,
        adj,
        method,
        img_key="img_emb" if "img_emb" in ref.obsm else None,
        fm_key=fk,
        notes=notes,
    )
    summary["slide"] = spec["id"]
    metrics = _primary_from_summary(summary, spec)
    return {"method": method, "notes": notes, **metrics}


def max_hops_sweep_row(ref, spec: dict, fk: str) -> list[dict]:
    """Sweep spatial hop cap at skip-add time (gene-gated)."""
    rows = []
    for mh in (0, 2, 3, 4):
        adj = build_gated_skip_graph(ref, fk, max_hops=mh)
        label = "STAMP gene-gated" if mh == 0 else f"STAMP gene-gated max_hops={mh}"
        note = "baseline gated" if mh == 0 else f"skip max_hops={mh} on spatial graph"
        rows.append(score_arm(ref, adj, spec, fk, label, note))
    return rows


def tau_remote_sweep_row(ref, adj, spec: dict, fk: str, *, method: str) -> list[dict]:
    """Reporting-only τ_remote sweep on a fixed graph."""
    rows = []
    for tr in (2.0, 2.5, 3.0, 3.5, 4.0):
        thr = AuditThresholds(tau_remote=tr)
        summary, _ = score_graph(
            ref,
            adj,
            method,
            img_key="img_emb" if "img_emb" in ref.obsm else None,
            fm_key=fk,
            thresholds=thr,
            notes=f"τ_remote={tr}",
        )
        m = _primary_from_summary(summary, spec)
        rows.append({"slide": spec["id"], "tau_remote": tr, "method": method, **m})
    return rows


def run_remote_cap_v07(*, slides: list[dict] | None = None) -> dict[str, Any]:
    setup_env()
    slides = slides or [s for s in load_slide_registry() if s.get("gate") and s["primary"] == "idc_dcis"]
    max_hops_rows: list[dict] = []
    tau_rows: list[dict] = []
    cap_rows: list[dict] = []

    for spec in slides:
        ref = load_staig(spec["path"], spec["kind"], spec.get("truth_file", "truth.txt"))
        ensure_feat(ref)
        fk = fm_key(ref)
        if not fk:
            continue

        g0 = np.asarray(ref.obsm["graph_neigh"])
        for row in max_hops_sweep_row(ref, spec, fk):
            max_hops_rows.append({"slide": spec["id"], **row})

        g_gated = build_gated_skip_graph(ref, fk, max_hops=0)
        for tr in tau_remote_sweep_row(ref, g_gated, spec, fk, method="STAMP gene-gated"):
            tau_rows.append(tr)

        g_cap = post_cap_remote_skip(ref, g0, g_gated, fk, nn_mult_cap=TAU_REMOTE)
        cap_rows.append(
            {"slide": spec["id"], **score_arm(
                ref,
                g_cap,
                spec,
                fk,
                "STAMP gene-gated + post nn× cap",
                f"drop skip-added IDC–DCIS with nn_mult≥{TAU_REMOTE}",
            )}
        )

        g_mh2 = build_gated_skip_graph(ref, fk, max_hops=2)
        cap_rows.append(
            {"slide": spec["id"], **score_arm(
                ref,
                g_mh2,
                spec,
                fk,
                "STAMP gene-gated + max_hops=2",
                "skip max_hops=2 at add time (recommended audit extension)",
            )}
        )

    _write_max_hops_table(max_hops_rows, TAB / "st_graph_audit_remote_cap_sweep.md")
    _write_tau_remote_table(tau_rows, TAB / "st_graph_audit_tau_remote_sweep.md")
    _write_cap_boundary(max_hops_rows, cap_rows, TAB / "st_graph_audit_remote_cap_boundary.md")
    _plot_max_hops(max_hops_rows, FIG / "st_graph_audit_remote_cap_sweep.png")
    return {"max_hops_rows": max_hops_rows, "tau_rows": tau_rows, "cap_rows": cap_rows}


def _write_max_hops_table(rows: list[dict], path: Path) -> None:
    lines = [
        "# ST-Graph Audit · max_hops 敏感性（gene-gated skip · v0.7）",
        "",
        "> **审计扩展**：在 skip 添加时限制空间图 hop 距离；**不改变** freeze 0.617 训练图。",
        "",
        "| slide | arm | max_hops | collide | lookalike | remote | rga | spatial_look |",
        "|-------|-----|----------|---------|-----------|--------|-----|--------------|",
    ]
    for r in rows:
        mh = "0 (baseline)" if "max_hops=0" not in r["method"] and r["method"] == "STAMP gene-gated" else (
            r["method"].split("max_hops=")[-1] if "max_hops=" in r["method"] else "0"
        )
        lines.append(
            f"| {r['slide']} | {r['method']} | {mh} | {r['collide']} | {r['lookalike']} | "
            f"{r['remote']} | {r['rga']} | {r['spatial_look']} |"
        )
    lines += [
        "",
        "**读法**",
        "",
        "- `max_hops=2`：只允许在空间图上 ≤2 hop 的 spot 对之间加 skip → 阻断远程 FM 撞车。",
        "- Sec2 **rga 16 → 0** 预期在此臂出现（xfer 标签 + 高 gene 远程 skip 为 gate 盲区）。",
        "- Block A baseline gated 已全 0 → max_hops 不改变主结论。",
        "",
        f"_generated {_now_iso()}_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_tau_remote_table(rows: list[dict], path: Path) -> None:
    lines = [
        "# ST-Graph Audit · τ_remote 报告敏感性（gene-gated · v0.7）",
        "",
        "> 固定图拓扑，仅改变 **remote / rga 分类阈值**（默认 τ_remote=3.0）。",
        "",
        "| slide | τ_remote | lookalike | remote | rga | spatial_look |",
        "|-------|----------|-----------|--------|-----|--------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['slide']} | {r['tau_remote']:.1f} | {r['lookalike']} | {r['remote']} | "
            f"{r['rga']} | {r['spatial_look']} |"
        )
    lines += [
        "",
        "**读法**：τ_remote 升高 → remote/rga 计数下降（定义变严）；图结构不变。",
        "机制修复见 `st_graph_audit_remote_cap_sweep.md`（max_hops 臂）。",
        "",
        f"_generated {_now_iso()}_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_cap_boundary(max_hops_rows: list[dict], cap_rows: list[dict], path: Path) -> None:
    baseline = {r["slide"]: r for r in max_hops_rows if r["method"] == "STAMP gene-gated"}
    mh2 = {r["slide"]: r for r in cap_rows if "max_hops=2" in r["method"]}
    post = {r["slide"]: r for r in cap_rows if "post nn" in r["method"]}

    lines = [
        "# ST-Graph Audit · Remote cap 边界（v0.7）",
        "",
        "> gene-gated 四通道 harm 在 **remote 盲区** 上的可选修复臂。",
        "",
        "| slide | gated rga | max_hops=2 rga | post nn× cap rga | rga cleared? |",
        "|-------|-----------|----------------|------------------|--------------|",
    ]
    for slide in sorted(baseline.keys()):
        b = baseline[slide]
        m = mh2.get(slide, {})
        p = post.get(slide, {})
        brga = b.get("rga", 0)
        mrga = m.get("rga", brga)
        prga = p.get("rga", brga)
        cleared = mrga == 0 and brga > 0
        lines.append(
            f"| {slide} | {brga} | {mrga} | {prga} | {'✅' if cleared else ('—' if brga == 0 else 'partial')} |"
        )
    lines += [
        "",
        "### 推荐 audit 扩展（非 freeze 训练）",
        "",
        "```",
        "if primary_remote_gene_agree > 0 on gene-gated arm:",
        "    report STAMP gene-gated + max_hops=2 as optional repair",
        "    (spatial hop cap at skip-add; blocks remote high-gene skip)",
        "```",
        "",
        f"_generated {_now_iso()}_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_max_hops(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    slides = sorted({r["slide"] for r in rows})
    metrics = [("rga", "remote_gene_agree"), ("remote", "remote"), ("lookalike", "lookalike")]
    fig, axes = plt.subplots(1, 3, figsize=(12, max(3.5, 2.2 * len(slides))), sharex=True)
    hop_order = ["STAMP gene-gated", "STAMP gene-gated max_hops=2", "STAMP gene-gated max_hops=3", "STAMP gene-gated max_hops=4"]
    x = np.arange(len(hop_order))
    w = 0.18
    for ax, (key, title) in zip(axes, metrics):
        for si, slide in enumerate(slides):
            sub = {r["method"]: r for r in rows if r["slide"] == slide}
            vals = [sub.get(m, {}).get(key, 0) for m in hop_order]
            ax.bar(x + (si - len(slides) / 2 + 0.5) * w, vals, w, label=slide)
        ax.set_xticks(x)
        ax.set_xticklabels(["gated", "mh2", "mh3", "mh4"], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("count")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("ST-Graph Audit v0.7 · max_hops sweep (gene-gated skip)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
