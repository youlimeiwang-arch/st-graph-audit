"""ST-Graph Audit v1.0: certify command + community harm report template."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from stamp.edges import add_fm_skip_edges
from stamp.graph_audit.core import score_graph
from stamp.graph_audit_benchmark import (
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
from stamp.graph_audit_budget import CONFIG_PATH, score_budget, score_budget_metrics

CERT_PATH = TAB / "st_graph_audit_certificate.md"
REPORT_PATH = TAB / "st_graph_audit_harm_report.md"


def load_audit_config(path: Path | None = None) -> dict[str, Any]:
    p = path or CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"missing audit config: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _registry_map() -> dict[str, dict]:
    return {s["id"]: s for s in load_slide_registry()}


def build_adj_from_graph_config(ref, fk: str, graph_config: dict) -> np.ndarray:
    """Build adjacency for recommended audit arm from yaml graph_config."""
    g0 = np.array(ref.obsm["graph_neigh"], dtype=np.float64, copy=True)
    ep0 = np.array(ref.obsm["edge_probabilities"], dtype=np.float64, copy=True)
    if not graph_config.get("fm_skip"):
        return g0
    if not fk:
        raise ValueError("fm_skip requested but no Phikon embedding on slide")
    ad = ref.copy()
    ad.obsm["graph_neigh"] = g0.copy()
    ad.obsm["edge_probabilities"] = ep0.copy()
    gg = graph_config.get("gene_gate") or {}
    add_fm_skip_edges(
        ad,
        fk,
        k=1,
        drop_prob=0.05,
        pca_dim=16,
        gene_key=str(gg.get("gene_key", "feat")),
        gene_k=int(gg.get("gene_k", 10)),
        gene_mutual=bool(gg.get("mutual", True)),
        max_hops=int(graph_config.get("max_hops") or 0),
    )
    return np.asarray(ad.obsm["graph_neigh"])


def _summary_to_metrics(summary: dict) -> dict[str, int]:
    return {
        "collide": int(summary.get("primary_collide") or 0),
        "look": int(summary.get("primary_lookalike") or 0),
        "remote": int(summary.get("primary_remote") or 0),
        "rga": int(summary.get("primary_remote_gene_agree") or 0),
        "spatial_look": int(summary.get("primary_spatial_lookalike") or 0),
    }


def certify_slide_live(slide_id: str, slide_cfg: dict, reg: dict) -> dict[str, Any]:
    """Re-score recommended arm from yaml graph_config (independent verification)."""
    meta = reg.get(slide_id, {})
    path = ROOT / slide_cfg.get("path", meta.get("path", ""))
    kind = meta.get("kind", "visium")
    truth = meta.get("truth_file", "truth.txt")
    ref = load_staig(path, kind, truth)
    ensure_feat(ref)
    fk = fm_key(ref)
    gc = slide_cfg["graph_config"]
    adj = build_adj_from_graph_config(ref, fk or "", gc)
    method = slide_cfg.get("recommended_arm", "certified")
    summary, _ = score_graph(
        ref,
        adj,
        method,
        img_key="img_emb" if "img_emb" in ref.obsm else None,
        fm_key=fk,
        notes="certify --live",
    )
    spec = {"primary": slide_cfg.get("primary", meta.get("primary", "idc_dcis"))}
    _attach_primary(summary, spec)
    metrics = _summary_to_metrics(summary)
    budget = score_budget_metrics(
        slide_id,
        slide_cfg.get("harm_budget_profile", ""),
        metrics,
        skip_rule_on=slide_cfg.get("skip_rule_on"),
        primary=slide_cfg.get("primary", "idc_dcis"),
    )
    cached = slide_cfg.get("harm_metrics") or {}
    drift = any(metrics.get(k) != cached.get(k) for k in ("look", "remote", "rga", "spatial_look"))
    return {
        "slide": slide_id,
        "mode": "live",
        "recommended_arm": method,
        "metrics": metrics,
        "cached_metrics": cached,
        "drift": drift,
        "budget": budget,
        "certified": budget["status"] in ("PASS", "REPORT"),
    }


def certify_from_artifacts(config: dict) -> list[dict]:
    """Fast certificate from v0.9 budget + yaml (no re-load)."""
    budget_path = TAB / "st_graph_audit_harm_budget.json"
    if budget_path.is_file():
        slide_scores = json.loads(budget_path.read_text(encoding="utf-8")).get("slide_scores", [])
        by_slide = {s["slide"]: s for s in slide_scores}
    else:
        by_slide = {}

    out = []
    for slide_id, slide_cfg in config.get("slides", {}).items():
        b = by_slide.get(slide_id)
        if b:
            status = b["status"]
            metrics = b.get("metrics", slide_cfg.get("harm_metrics", {}))
        else:
            status = slide_cfg.get("harm_budget_status", "UNKNOWN")
            metrics = slide_cfg.get("harm_metrics", {})
        out.append(
            {
                "slide": slide_id,
                "mode": "artifact",
                "recommended_arm": slide_cfg.get("recommended_arm"),
                "metrics": metrics,
                "budget": {"status": status, "profile": slide_cfg.get("harm_budget_profile"), "violations": slide_cfg.get("violations", [])},
                "certified": status in ("PASS", "REPORT"),
            }
        )
    return out


def run_certify_v10(
    *,
    live: bool = False,
    slides: list[str] | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    setup_env()
    config = load_audit_config(config_path)
    reg = _registry_map()
    target = slides or list(config.get("slides", {}).keys())

    results: list[dict] = []
    if live:
        for sid in target:
            if sid not in config.get("slides", {}):
                continue
            results.append(certify_slide_live(sid, config["slides"][sid], reg))
    else:
        all_res = certify_from_artifacts(config)
        results = [r for r in all_res if r["slide"] in target]

    n_cert = sum(1 for r in results if r["certified"])
    n_pass = sum(1 for r in results if r["budget"].get("status") == "PASS")
    n_report = sum(1 for r in results if r["budget"].get("status") == "REPORT")
    n_fail = sum(1 for r in results if r["budget"].get("status") == "FAIL")
    overall = "PASS" if n_fail == 0 and n_cert == len(results) else "FAIL"

    payload = {
        "version": "1.0",
        "time": _now_iso(),
        "mode": "live" if live else "artifact",
        "overall": overall,
        "summary": {"certified": n_cert, "pass": n_pass, "report": n_report, "fail": n_fail, "total": len(results)},
        "freeze_note": config.get("freeze_training"),
        "results": results,
    }
    (TAB / "st_graph_audit_certificate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_certificate_md(payload, CERT_PATH)
    _write_harm_report(config, payload, REPORT_PATH)
    return payload


def _write_certificate_md(payload: dict, path: Path) -> None:
    s = payload["summary"]
    lines = [
        "# ST-Graph Audit Certificate（v1.0）",
        "",
        f"> **Overall: {payload['overall']}** · certified {s['certified']}/{s['total']} "
        f"(PASS={s['pass']} · REPORT={s['report']} · FAIL={s['fail']})",
        f"> Mode: `{payload['mode']}` · {payload.get('freeze_note', '')}",
        "",
        "| slide | status | look | remote | rga | spatial_look | recommended arm | drift |",
        "|-------|--------|------|--------|-----|--------------|-----------------|-------|",
    ]
    for r in payload["results"]:
        m = r.get("metrics", {})
        drift = "yes" if r.get("drift") else ("—" if r.get("mode") == "artifact" else "no")
        lines.append(
            f"| {r['slide']} | **{r['budget'].get('status')}** | {m.get('look', '—')} | "
            f"{m.get('remote', '—')} | {m.get('rga', '—')} | {m.get('spatial_look', '—')} | "
            f"{r.get('recommended_arm', '')} | {drift} |"
        )
    lines += [
        "",
        "```bash",
        "# Re-verify independently (re-scores recommended arm from yaml):",
        "python -m stamp.graph_audit certify --live",
        "```",
        "",
        f"_issued {payload['time']}_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_harm_report(config: dict, cert: dict, path: Path) -> None:
    """Community-facing harm report template (Methods-ready paragraph + tables)."""
    thr = config.get("thresholds", {})
    skip = config.get("skip_rule", {})
    lines = [
        "# ST-Graph Audit · Community Harm Report（v1.0 模板）",
        "",
        "> 复制以下 **Methods 段** 到论文/附录；表格为本次 certify 快照。",
        "",
        "## Methods（可直接引用）",
        "",
        (
            "We audited histology-augmented spatial graphs using the ST-Graph Audit Benchmark "
            "(spec v1.0). For each labeled slide we report primary-tissue collide edges, "
            "lookalike pairs (gene cosine ≤ "
            f"{thr.get('tau_gene', 0.2)} with appearance cosine ≥ "
            f"{thr.get('tau_fm', 0.8)}), remote pairs (spatial distance ≥ "
            f"{thr.get('tau_remote', 3.0)}× median nearest-neighbor spacing), "
            "remote-gene-agreement pairs (remote cross-class edges with gene cosine above threshold), "
            "and spatial lookalike pairs (lookalike but not remote). "
            "Breast slides enable FM skip only when ungated skip satisfies a frozen rule "
            f"(≥{skip.get('n_collide_min', 10)} IDC–DCIS collide edges, median nn-multiple ≥ "
            f"{skip.get('nn_mult_min', 3.0)}, median collide gene cosine ≤ "
            f"{skip.get('gene_median_max', 0.2)}); otherwise we use the spatial graph only. "
            "When gene-gated skip retains remote harm, we report an optional "
            "max_hops=2 extension arm. Graph harm metrics are independent of clustering ARI."
        ),
        "",
        "## Certification snapshot",
        "",
        f"**Overall certificate: {cert['overall']}** ({cert['time']})",
        "",
        "| slide | budget | look | remote | rga | spatial_look | arm |",
        "|-------|--------|------|--------|-----|--------------|-----|",
    ]
    for r in cert["results"]:
        m = r.get("metrics", {})
        lines.append(
            f"| {r['slide']} | {r['budget'].get('status')} | {m.get('look', '—')} | "
            f"{m.get('remote', '—')} | {m.get('rga', '—')} | {m.get('spatial_look', '—')} | "
            f"{r.get('recommended_arm', '')} |"
        )
    lines += [
        "",
        "## 必报清单（社区复现）",
        "",
        "1. 四通道 harm on **recommended audit arm** (not training graph unless stated)",
        "2. `skip_rule_on` ON/OFF with ungated skip collide stats",
        "3. Leaderboard row for STAIG+ungated skip vs spatial baselines",
        "4. Optional: 5-seed graph-harm IQR on edge dropout",
        "",
        "## 一键复现",
        "",
        "```bash",
        "python -m stamp.graph_audit suite      # full Track A stack",
        "python -m stamp.graph_audit certify    # issue certificate (artifact mode)",
        "python -m stamp.graph_audit certify --live  # independent re-score",
        "```",
        "",
        f"_generated {cert['time']}_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
