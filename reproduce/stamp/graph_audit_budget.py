"""ST-Graph Audit v0.9: harm budget PASS/FAIL + machine-readable audit_config.yaml."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from stamp.graph_audit.core import TAU_GENE, TAU_FM, TAU_IMG, TAU_REMOTE
from stamp.graph_audit_benchmark import FIG, ROOT, TAB, _int0, _now_iso, load_slide_registry
from stamp.graph_audit_decision import decide_slide

CONFIG_PATH = ROOT / "config" / "st_graph_audit_config.yaml"

# Frozen harm budgets (audit scoring — not training gates)
HARM_BUDGETS: dict[str, dict[str, Any]] = {
    "breast_skip_on_strict": {
        "lookalike_max": 0,
        "remote_max": 0,
        "remote_gene_agree_max": 0,
        "spatial_lookalike_max": 0,
    },
    "breast_skip_on_ffpe": {
        "lookalike_max": 1,
        "remote_max": 0,
        "remote_gene_agree_max": 0,
        "spatial_lookalike_max": 1,
    },
    "breast_skip_off": {
        "skip_rule_must_be": False,
        "mode": "skip_off_confirmed",
    },
    "cross_layer": {
        "mode": "report_only",
    },
}


def _final_metrics(d: dict) -> dict[str, int]:
    arm = d.get("final_audit_arm", "")
    if "max_hops=2" in arm and d.get("max_hops_2"):
        return dict(d["max_hops_2"])
    low = arm.lower()
    if "spatial" in low and ("no fm skip" in low or "skip" not in low):
        return dict(d["spatial"])
    return dict(d["gated"])


def _budget_profile(slide: str, d: dict) -> str:
    if d.get("primary") == "cross":
        return "cross_layer"
    if d.get("skip_rule_on") is False:
        return "breast_skip_off"
    if slide == "CytAssist_Patho":
        return "breast_skip_on_ffpe"
    return "breast_skip_on_strict"


def _arm_to_config(d: dict) -> dict[str, Any]:
    """Map v0.8 decision → runnable graph config (audit-only)."""
    arm = d["final_audit_arm"]
    if "max_hops=2" in arm:
        return {
            "base": "STAIG spatial k=5",
            "fm_skip": True,
            "gene_gate": {"gene_k": 10, "mutual": True, "gene_key": "feat"},
            "max_hops": 2,
        }
    if "gene-gated" in arm:
        return {
            "base": "STAIG spatial k=5",
            "fm_skip": True,
            "gene_gate": {"gene_k": 10, "mutual": True, "gene_key": "feat"},
            "max_hops": 0,
        }
    if d.get("skip_rule_on") is False:
        return {"base": "STAIG spatial k=5", "fm_skip": False}
    return {"base": "STAIG spatial k=5", "fm_skip": False}


def score_budget(d: dict, profile: str | None = None) -> dict[str, Any]:
    profile = profile or _budget_profile(d["slide"], d)
    spec = HARM_BUDGETS[profile]
    m = _final_metrics(d)

    if spec.get("mode") == "report_only":
        return {
            "profile": profile,
            "status": "REPORT",
            "violations": [],
            "metrics": m,
            "note": "cross-layer harm reported; no breast PASS/FAIL",
        }
    if spec.get("mode") == "skip_off_confirmed":
        ok = d.get("skip_rule_on") is False
        return {
            "profile": profile,
            "status": "PASS" if ok else "FAIL",
            "violations": [] if ok else ["skip_rule should be OFF"],
            "metrics": m,
            "note": "Wu-style short-range; audit spatial baseline",
        }

    violations: list[str] = []
    if m["look"] > spec["lookalike_max"]:
        violations.append(f"lookalike {m['look']} > {spec['lookalike_max']}")
    if m["remote"] > spec["remote_max"]:
        violations.append(f"remote {m['remote']} > {spec['remote_max']}")
    if m["rga"] > spec["remote_gene_agree_max"]:
        violations.append(f"remote_gene_agree {m['rga']} > {spec['remote_gene_agree_max']}")
    if m["spatial_look"] > spec["spatial_lookalike_max"]:
        violations.append(
            f"spatial_lookalike {m['spatial_look']} > {spec['spatial_lookalike_max']}"
        )
    return {
        "profile": profile,
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "metrics": m,
        "note": "",
    }


def score_budget_metrics(
    slide: str,
    profile: str,
    metrics: dict[str, int],
    *,
    skip_rule_on: bool | None = None,
    primary: str = "idc_dcis",
) -> dict[str, Any]:
    """Score harm metrics against a budget profile (no full decision dict)."""
    if not profile:
        profile = _budget_profile(slide, {"slide": slide, "primary": primary, "skip_rule_on": skip_rule_on})
    d = {
        "slide": slide,
        "primary": primary,
        "skip_rule_on": skip_rule_on,
        "final_audit_arm": "",
        "gated": metrics,
        "max_hops_2": None,
        "spatial": metrics,
    }
    return score_budget(d, profile)


def score_method_arm(row: dict, *, slide_skip_on: bool | None = None) -> dict[str, Any]:
    """Score one extreme-table row against breast budget (when applicable)."""
    primary = row.get("primary") or "idc_dcis"
    if primary == "cross":
        return {"status": "REPORT", "violations": []}
    slide = row.get("slide", "")
    skip_on = slide_skip_on if slide_skip_on is not None else row.get("skip_rule_on")
    if skip_on is False:
        return {"status": "N/A", "violations": [], "profile": "breast_skip_off"}
    m = {
        "look": _int0(row.get("primary_lookalike")),
        "remote": _int0(row.get("primary_remote")),
        "rga": _int0(row.get("primary_remote_gene_agree")),
        "spatial_look": _int0(row.get("primary_spatial_lookalike")),
    }
    profile = _budget_profile(slide, {"slide": slide, "primary": primary, "skip_rule_on": True})
    spec = HARM_BUDGETS[profile]
    v = []
    for key, attr, mx in (
        ("lookalike_max", "look", "lookalike"),
        ("remote_max", "remote", "remote"),
        ("remote_gene_agree_max", "rga", "remote_gene_agree"),
        ("spatial_lookalike_max", "spatial_look", "spatial_lookalike"),
    ):
        if m[attr] > spec[key]:
            v.append(f"{key.replace('_max','')}={m[attr]}")
    return {"status": "PASS" if not v else "FAIL", "violations": v, "profile": profile, "metrics": m}


def build_audit_config(decisions: list[dict], slide_registry: list[dict]) -> dict[str, Any]:
    reg = {s["id"]: s for s in slide_registry}
    slides_cfg = {}
    for d in decisions:
        sid = d["slide"]
        spec = reg.get(sid, {})
        budget = score_budget(d)
        slides_cfg[sid] = {
            "path": str(spec.get("path", "")).replace(str(ROOT) + "/", ""),
            "primary": d.get("primary"),
            "role": spec.get("role", ""),
            "skip_rule_on": d.get("skip_rule_on"),
            "recommended_arm": d["final_audit_arm"],
            "graph_config": _arm_to_config(d),
            "harm_budget_profile": budget["profile"],
            "harm_budget_status": budget["status"],
            "harm_metrics": budget["metrics"],
            "violations": budget["violations"],
            "flags": d.get("flags", []),
        }
    return {
        "version": "0.9",
        "spec": "st_graph_audit_spec.md",
        "freeze_training": "gene-gated mclust 0.617 unchanged",
        "thresholds": {
            "tau_gene": TAU_GENE,
            "tau_img": TAU_IMG,
            "tau_fm": TAU_FM,
            "tau_remote": TAU_REMOTE,
        },
        "skip_rule": {
            "n_collide_min": 10,
            "nn_mult_min": 3.0,
            "gene_median_max": 0.20,
        },
        "harm_budgets": HARM_BUDGETS,
        "audit_arms_default": [
            "GraphST (k=3)",
            "STAGATE (radius)",
            "SpaceFlow (spatial)",
            "STAIG spatial (k=5)",
            "STAIG+ungated FM skip",
            "STAMP gene-gated skip",
            "STAMP gene-gated + max_hops=2",
        ],
        "slides": slides_cfg,
    }


def run_budget_v09(
    all_rows: list[dict] | None = None,
    decisions: list[dict] | None = None,
) -> dict[str, Any]:
    p = TAB / "st_graph_audit_extreme.json"
    if all_rows is None:
        if not p.is_file():
            raise FileNotFoundError(p)
        all_rows = json.loads(p.read_text(encoding="utf-8")).get("rows", [])

    if decisions is None:
        dec_path = TAB / "st_graph_audit_decision.json"
        if dec_path.is_file():
            decisions = json.loads(dec_path.read_text(encoding="utf-8")).get("decisions", [])
        else:
            slides = sorted({r.get("slide") for r in all_rows if r.get("slide")})
            decisions = []
            for slide in slides:
                sample = next(r for r in all_rows if r.get("slide") == slide)
                primary = sample.get("primary") or ("cross" if "DLPFC" in slide else "idc_dcis")
                decisions.append(decide_slide(slide, all_rows, primary=primary))

    registry = load_slide_registry()
    config = build_audit_config(decisions, registry)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (TAB / "st_graph_audit_config.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    slide_skip = {d["slide"]: d.get("skip_rule_on") for d in decisions}
    method_scores = []
    for row in all_rows:
        if "error" in row:
            continue
        sc = score_method_arm(row, slide_skip_on=slide_skip.get(row.get("slide")))
        method_scores.append(
            {
                "slide": row.get("slide"),
                "method": row.get("method"),
                "skip_rule": row.get("skip_rule_on"),
                **sc,
            }
        )

    _write_budget_md(decisions, method_scores, TAB / "st_graph_audit_harm_budget.md")
    _plot_budget_summary(decisions, FIG / "st_graph_audit_harm_budget.png")

    payload = {
        "version": "0.9",
        "time": _now_iso(),
        "config_path": str(CONFIG_PATH),
        "slide_scores": [score_budget(d) | {"slide": d["slide"]} for d in decisions],
        "method_scores": method_scores,
    }
    (TAB / "st_graph_audit_harm_budget.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_budget_md(decisions: list[dict], method_scores: list[dict], path: Path) -> None:
    lines = [
        "# ST-Graph Audit · Harm Budget PASS/FAIL（v0.9）",
        "",
        "> **Frozen budget**：推荐 audit 臂上 look/remote/rga=0；CytAssist 允许 spatial_look≤1。",
        "> 配置模板 → `config/st_graph_audit_config.yaml`",
        "",
        "## 切片级（推荐臂）",
        "",
        "| slide | profile | status | L/R/rga/sL | violations | recommended arm |",
        "|-------|---------|--------|------------|------------|-----------------|",
    ]
    for d in decisions:
        b = score_budget(d)
        m = b["metrics"]
        viol = "; ".join(b["violations"]) if b["violations"] else "—"
        lines.append(
            f"| {d['slide']} | {b['profile']} | **{b['status']}** | "
            f"{m['look']}/{m['remote']}/{m['rga']}/{m['spatial_look']} | {viol} | "
            f"{d['final_audit_arm']} |"
        )

    lines += [
        "",
        "## 方法臂级（strict breast budget；skip OFF 切片 N/A）",
        "",
        "| slide | method | status | look | remote | rga | spatial_look |",
        "|-------|--------|--------|------|--------|-----|--------------|",
    ]
    for s in sorted(method_scores, key=lambda x: (x["slide"], x["status"] != "PASS", x["method"])):
        if s["status"] == "N/A":
            lines.append(f"| {s['slide']} | {s['method']} | N/A | — | — | — | — |")
            continue
        met = s.get("metrics", {})
        lines.append(
            f"| {s['slide']} | {s['method']} | {s['status']} | "
            f"{met.get('look', '—')} | {met.get('remote', '—')} | {met.get('rga', '—')} | "
            f"{met.get('spatial_look', '—')} |"
        )

    n_pass = sum(1 for d in decisions if score_budget(d)["status"] == "PASS")
    n_fail = sum(1 for d in decisions if score_budget(d)["status"] == "FAIL")
    n_report = sum(1 for d in decisions if score_budget(d)["status"] == "REPORT")
    lines += [
        "",
        f"**汇总**：PASS={n_pass} · FAIL={n_fail} · REPORT={n_report}",
        "",
        "### Budget profiles",
        "",
        "```yaml",
        yaml.safe_dump({"harm_budgets": HARM_BUDGETS}, sort_keys=False, default_flow_style=False).strip(),
        "```",
        "",
        f"_generated {_now_iso()}_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_budget_summary(decisions: list[dict], out: Path) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    slides = [d["slide"] for d in decisions]
    colors = {"PASS": "#10b981", "FAIL": "#ef4444", "REPORT": "#64748b"}
    statuses = [score_budget(d)["status"] for d in decisions]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(slides)), 3.2))
    ax.bar(range(len(slides)), [1] * len(slides), color=[colors.get(s, "#94a3b8") for s in statuses])
    ax.set_xticks(range(len(slides)))
    ax.set_xticklabels([s.replace("_", "\n") for s in slides], fontsize=8)
    ax.set_yticks([])
    ax.set_title("ST-Graph Audit v0.9 · harm budget (recommended arm)", fontsize=10)
    for i, (st, d) in enumerate(zip(statuses, decisions)):
        ax.text(i, 0.5, st, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
