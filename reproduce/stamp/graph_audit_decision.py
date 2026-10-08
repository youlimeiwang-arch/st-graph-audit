"""ST-Graph Audit v0.8: skip_rule × remote_cap joint audit decision protocol."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from stamp.graph_audit_benchmark import FIG, TAB, _int0, _now_iso, fmt_cell


def _row(all_rows: list[dict], slide: str, needle: str) -> dict | None:
    for r in all_rows:
        if r.get("slide") != slide or "error" in r:
            continue
        if needle in str(r.get("method", "")):
            return r
    return None


def _metrics(r: dict | None, *, primary: str = "idc_dcis") -> dict[str, int]:
    if not r:
        return {"collide": 0, "look": 0, "remote": 0, "rga": 0, "spatial_look": 0}
    if primary == "cross":
        return {
            "collide": _int0(r.get("primary_collide")),
            "look": _int0(r.get("primary_lookalike")),
            "remote": _int0(r.get("primary_remote")),
            "rga": _int0(r.get("primary_remote_gene_agree")),
            "spatial_look": _int0(r.get("primary_spatial_lookalike")),
        }
    return {
        "collide": _int0(r.get("primary_collide")),
        "look": _int0(r.get("primary_lookalike")),
        "remote": _int0(r.get("primary_remote")),
        "rga": _int0(r.get("primary_remote_gene_agree")),
        "spatial_look": _int0(r.get("primary_spatial_lookalike")),
    }


def _four_clear(m: dict[str, int]) -> bool:
    return m["look"] == 0 and m["remote"] == 0 and m["rga"] == 0 and m["spatial_look"] == 0


def decide_slide(slide: str, all_rows: list[dict], *, primary: str = "idc_dcis") -> dict[str, Any]:
    """Frozen two-stage protocol: skip ON/OFF → optional max_hops=2 repair."""
    ung = _row(all_rows, slide, "ungated skip")
    gated = _row(all_rows, slide, "gene-gated skip") or _row(all_rows, slide, "STAMP gene-gated skip")
    if not gated:
        gated = next(
            (r for r in all_rows if r.get("slide") == slide and "gene-gated" in r.get("method", "") and "max_hops" not in r.get("method", "")),
            None,
        )
    mh2 = _row(all_rows, slide, "max_hops=2")
    spatial = _row(all_rows, slide, "STAIG spatial")
    has_mh2 = mh2 is not None

    skip_on = ung.get("skip_rule_on") if ung else None
    if skip_on is None and gated:
        skip_on = gated.get("skip_rule_on")

    gu = _metrics(ung, primary=primary)
    gg = _metrics(gated, primary=primary)
    gm = _metrics(mh2, primary=primary) if has_mh2 else None
    gs = _metrics(spatial, primary=primary)

    # Stage 1 — skip eligibility
    if primary == "cross":
        stage1 = "N/A (cross-layer tissue)"
        skip_rec = "spatial baseline + cross-layer audit"
        skip_reason = "breast skip_rule not applicable"
    elif skip_on is False:
        stage1 = "skip OFF"
        skip_rec = "STAIG spatial (k=5) — no FM skip"
        skip_reason = (
            f"ungated collide={gu['collide']} but frozen rule OFF "
            f"(need n≥10 ∧ nn×≥3 ∧ gene≤0.20 on IDC–DCIS collide)"
        )
    elif skip_on is True:
        stage1 = "skip ON"
        skip_rec = "STAMP gene-gated skip (gene_k=10 mutual)"
        skip_reason = f"ungated harm look/remote={gu['look']}/{gu['remote']}; gate → {gg['look']}/{gg['remote']}/{gg['rga']}"
    else:
        stage1 = "skip unknown"
        skip_rec = "audit spatial + ungated if FM available"
        skip_reason = "missing ungated skip arm"

    # Stage 2 — remote cap extension (audit-only; breast skip ON only)
    cap_needed = skip_on is True and (gg["rga"] > 0 or gg["remote"] > 0)
    if cap_needed and has_mh2 and gm is not None:
        if _four_clear(gm):
            cap_rec = "STAMP gene-gated + max_hops=2"
            cap_reason = f"clears rga {gg['rga']}→0 remote {gg['remote']}→{gm['remote']}"
            final_arm = cap_rec
            final_clear = True
        else:
            cap_rec = "max_hops=2 partial — report residual"
            cap_reason = (
                f"after mh2: look/remote/rga/spatial={gm['look']}/{gm['remote']}/{gm['rga']}/{gm['spatial_look']}"
            )
            final_arm = cap_rec
            final_clear = _four_clear(gm)
    elif cap_needed and not has_mh2:
        cap_rec = "recommend max_hops=2 (arm not run)"
        cap_reason = "gated rga>0 or remote>0 under skip ON — add mh2 arm to audit"
        final_arm = skip_rec
        final_clear = _four_clear(gg)
    else:
        cap_rec = "—"
        cap_reason = "gated four-channel clear, skip OFF, or cross tissue"
        final_arm = skip_rec
        final_clear = _four_clear(gg)

    flags: list[str] = []
    if gg["spatial_look"] > 0:
        flags.append(f"spatial_lookalike={gg['spatial_look']} (local; gate blind spot)")
    if skip_on and gu["look"] > 0 and gg["look"] == 0:
        flags.append(f"gate repairs lookalike {gu['look']}→0")
    if primary == "cross" and gg["look"] > 0:
        flags.append("cross-layer lookalike expected on cortex")

    return {
        "slide": slide,
        "primary": primary,
        "skip_rule_on": skip_on,
        "stage1": stage1,
        "skip_recommended": skip_rec,
        "skip_reason": skip_reason,
        "cap_needed": cap_needed,
        "cap_recommended": cap_rec,
        "cap_reason": cap_reason,
        "final_audit_arm": final_arm,
        "four_channel_clear": final_clear,
        "ungated": gu,
        "gated": gg,
        "max_hops_2": gm,
        "has_mh2": has_mh2,
        "spatial": gs,
        "flags": flags,
    }


def run_decision_v08(all_rows: list[dict] | None = None) -> dict[str, Any]:
    p = TAB / "st_graph_audit_extreme.json"
    if all_rows is None:
        if not p.is_file():
            raise FileNotFoundError(f"need extreme.json or rows: {p}")
        all_rows = json.loads(p.read_text(encoding="utf-8")).get("rows", [])

    slides = sorted({r.get("slide") for r in all_rows if r.get("slide")})
    decisions = []
    for slide in slides:
        sample = next(r for r in all_rows if r.get("slide") == slide)
        primary = sample.get("primary") or ("cross" if "DLPFC" in slide else "idc_dcis")
        decisions.append(decide_slide(slide, all_rows, primary=primary))

    _write_decision_table(decisions, TAB / "st_graph_audit_decision.md")
    _write_gate_dual(all_rows, decisions, TAB / "st_graph_audit_gate_boundary.md")
    _plot_decision_tree(decisions, FIG / "st_graph_audit_decision_tree.png")
    _plot_slide_cards(decisions, FIG / "st_graph_audit_decision_cards.png")

    payload = {"version": "0.8", "time": _now_iso(), "decisions": decisions}
    (TAB / "st_graph_audit_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_decision_table(decisions: list[dict], path: Path) -> None:
    lines = [
        "# ST-Graph Audit · 审计决策协议（v0.8）",
        "",
        "> **两阶段 frozen 协议**：① skip_rule ON/OFF → ② 若 gated **rga>0** 或 **remote>0** → 报告 **max_hops=2** 扩展臂。",
        "> **非训练配置**；freeze gene-gated mclust **0.617** 不变。",
        "",
        "| slide | skip | stage1 推荐 | cap 推荐 | **最终 audit 臂** | 四通道清? | flags |",
        "|-------|------|-------------|----------|-------------------|-----------|-------|",
    ]
    for d in decisions:
        flags = "; ".join(d["flags"]) if d["flags"] else "—"
        lines.append(
            f"| {d['slide']} | {fmt_cell(d.get('skip_rule_on'))} | {d['skip_recommended']} | "
            f"{d['cap_recommended']} | **{d['final_audit_arm']}** | "
            f"{'✅' if d['four_channel_clear'] else '—'} | {flags} |"
        )
    lines += [
        "",
        "## 阶段逻辑",
        "",
        "```",
        "Stage 1 (breast): skip_rule_on = (n_collide≥10) ∧ (median_nn×≥3) ∧ (median_gene≤0.20)",
        "  OFF → STAIG spatial only",
        "  ON  → STAMP gene-gated skip",
        "",
        "Stage 2 (audit extension): if gated rga>0 OR (skip ON ∧ gated remote>0):",
        "  report STAMP gene-gated + max_hops=2",
        "  if mh2 four-channel clear → recommend mh2 as audit repair arm",
        "```",
        "",
        "### 切片解读",
        "",
    ]
    for d in decisions:
        g, m = d["gated"], d["max_hops_2"]
        if m is None:
            lines.append(
                f"- **{d['slide']}**: gated L/R/rga/sL={g['look']}/{g['remote']}/{g['rga']}/{g['spatial_look']}; "
                f"mh2=N/A. {d['cap_reason']}"
            )
        else:
            lines.append(
                f"- **{d['slide']}**: gated L/R/rga/sL={g['look']}/{g['remote']}/{g['rga']}/{g['spatial_look']}; "
                f"mh2={m['look']}/{m['remote']}/{m['rga']}/{m['spatial_look']}. {d['cap_reason']}"
            )
    lines += ["", f"_generated {_now_iso()}_"]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_gate_dual(all_rows: list[dict], decisions: list[dict], path: Path) -> None:
    dec_map = {d["slide"]: d for d in decisions}
    lines = [
        "# ST-Graph Audit · Gate 边界表（v0.8 · gated vs max_hops=2）",
        "",
        "> 必报四通道 + **v0.8 推荐 audit 臂**（见 `st_graph_audit_decision.md`）",
        "",
        "| slide | skip | gated L/R/rga/sL | mh2 L/R/rga/sL | 推荐臂 | 全清? |",
        "|-------|------|------------------|----------------|--------|-------|",
    ]
    for slide in sorted(dec_map.keys()):
        d = dec_map[slide]
        g, m = d["gated"], d["max_hops_2"]
        m_str = "—/—/—/—" if m is None else f"{m['look']}/{m['remote']}/{m['rga']}/{m['spatial_look']}"
        lines.append(
            f"| {slide} | {fmt_cell(d.get('skip_rule_on'))} | "
            f"{g['look']}/{g['remote']}/{g['rga']}/{g['spatial_look']} | {m_str} | "
            f"{d['final_audit_arm']} | {'✅' if d['four_channel_clear'] else '—'} |"
        )
    lines += [
        "",
        "**读法**",
        "",
        "- CytAssist `spatial_look=1`：局部 lookalike，gate 不覆盖 → 允许报告 ≤1",
        "- Sec2 `rga=16` → mh2 清零 → **最终臂 = gene-gated + max_hops=2**",
        "- Block A gated 已全清 → **最终臂 = gene-gated**（freeze ref）",
        "",
        f"_generated {_now_iso()}_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_decision_tree(decisions: list[dict], out: Path) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef2ff", ec="#4338ca", fs=7.5):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.1)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=10, color="#334155"))

    box(2.8, 9.0, 4.4, 0.75, "ST-Graph Audit v0.8\nlabeled Visium + FM skip arms", "#f8fafc", "#64748b")
    box(2.6, 7.8, 4.8, 0.7, "Breast IDC–DCIS primary?", "#fef9c3", "#ca8a04")
    arrow(5.0, 9.0, 5.0, 8.5)
    box(0.2, 6.2, 2.6, 0.9, "NO (DLPFC)\nspatial + cross-layer\nharm report", "#fee2e2", "#dc2626", 7)
    box(7.2, 6.2, 2.6, 0.9, "YES → Stage 1\nskip_rule frozen", "#ecfdf5", "#059669", 7)
    arrow(3.5, 7.8, 1.5, 7.1)
    arrow(6.5, 7.8, 8.5, 7.1)

    box(6.0, 4.8, 3.8, 0.75, "skip_rule ON?\nn≥10 ∧ nn×≥3 ∧ gene≤0.20", "#fef9c3", "#ca8a04", 7)
    arrow(8.5, 6.2, 7.9, 5.55)
    box(6.2, 3.2, 1.8, 0.85, "OFF\nWu: spatial only", "#fee2e2", "#dc2626", 7)
    box(8.4, 3.2, 2.0, 0.85, "ON\ngene-gated skip", "#dbeafe", "#2563eb", 7)
    arrow(7.0, 4.8, 7.1, 4.05)
    arrow(8.8, 4.8, 9.4, 4.05)

    box(7.0, 1.5, 3.6, 0.85, "Stage 2: gated rga>0\nOR remote>0 ?", "#fef9c3", "#ca8a04", 7)
    arrow(9.4, 3.2, 8.8, 2.35)
    box(6.5, 0.15, 2.2, 0.85, "NO\nreport gated", "#ecfdf5", "#059669", 7)
    box(9.0, 0.15, 2.8, 0.85, "YES\n+ max_hops=2 arm", "#dbeafe", "#2563eb", 7)
    arrow(8.0, 1.5, 7.6, 1.0)
    arrow(9.2, 1.5, 10.2, 1.0)

    # Annotate with real slides
    y = 5.5
    for d in decisions:
        if d["primary"] == "cross":
            continue
        tag = d["slide"].replace("BlockA", "BA").replace("_", " ")[:14]
        arm = "mh2" if "max_hops" in d["final_audit_arm"] else ("gated" if d["skip_rule_on"] else "spatial")
        ax.text(0.15, y, f"{tag}: {arm}", fontsize=6.5, color="#475569")
        y -= 0.35

    ax.text(0.15, 9.5, "Audit-only · does not change freeze 0.617 training graph", fontsize=7, color="#64748b")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_slide_cards(decisions: list[dict], out: Path) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    n = len(decisions)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.8))
    if n == 1:
        axes = [axes]
    channels = ["look", "remote", "rga", "spatial_look"]
    labels = ["look", "remote", "rga", "sp_look"]
    x = range(4)
    w = 0.35
    for ax, d in zip(axes, decisions):
        g = [d["gated"][c] for c in channels]
        m = d["max_hops_2"]
        if m is None:
            ax.bar([i - w / 2 for i in x], g, w, label="gated", color="#2563eb")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(d["slide"].replace("_", "\n") + "\n(no mh2)", fontsize=7)
            ax.set_ylim(0, max(max(g) * 1.15, 1))
        else:
            mv = [m[c] for c in channels]
            ax.bar([i - w / 2 for i in x], g, w, label="gated", color="#2563eb")
            ax.bar([i + w / 2 for i in x], mv, w, label="mh2", color="#10b981")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(d["slide"].replace("_", "\n"), fontsize=8)
            ax.set_ylim(0, max(max(g + mv) * 1.15, 1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(fontsize=6)
    fig.suptitle("v0.8 · four-channel harm: gated vs max_hops=2", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
