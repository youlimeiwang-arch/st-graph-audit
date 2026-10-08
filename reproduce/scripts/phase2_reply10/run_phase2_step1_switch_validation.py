"""Reply-10 Phase 2 Step 1: run the existing skip rule, unchanged, on every dataset.

Rule (frozen): ON iff n_collide >= 10 AND nn_mult_median >= 3.0 AND gene_cos_median <= 0.20,
computed on primary-collide edges of the ungated FM skip graph (Phikon-v2, k=1).
Graph-only (no training). ARI direction is read from existing result tables, not re-run.

Outputs: lab/static/tables/reply10_phase2_step1_switch.{md,json}
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
TAB = ROOT / "lab/static/tables"

SLIDES = [
    ("BlockA", "Block A"),
    ("CytAssist_Patho", "CytAssist (FFPE)"),
    ("Wu", "Wu CID44971"),
    ("DLPFC151673", "DLPFC 151673"),
    ("DLPFC151507", "DLPFC 151507"),
    ("Liver_Yu_L5", "Liver-Yu L5"),
    ("Kidney_DKD1", "Kidney DKD1"),
    ("Xenium_Rep1_IDC", "Xenium Rep1"),
    ("Hartman_BCSA_A1", "Hartman A1"),
    ("CRC_Valdeolivas", "CRC Valdeolivas Rep1"),
    ("CRC_Valdeolivas_Rep2", "CRC Valdeolivas Rep2"),
    ("HPR_Prostate", "HPR Prostate"),
]

FLOOR_ARI = 0.05
FLAT_DELTA = 0.01


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _phase1_pairs(fname: str) -> list[tuple[float, float, int]]:
    rows = json.loads((TAB / fname).read_text(encoding="utf-8")).get("ari", [])
    by_seed: dict[int, dict[str, float]] = {}
    for r in rows:
        if r.get("ok") is False or r.get("ari") is None:
            continue
        by_seed.setdefault(int(r["seed"]), {})[r["arm"]] = float(r["ari"])
    return [(v["ungated"], v["gated"], s) for s, v in by_seed.items() if {"ungated", "gated"} <= v.keys()]


def _crc_pairs(fname: str) -> list[tuple[float, float, int]]:
    rows = json.loads((TAB / fname).read_text(encoding="utf-8")).get("ari_rows", [])
    by_seed: dict[int, dict[str, float]] = {}
    for r in rows:
        by_seed.setdefault(int(r.get("seed", 39788)), {})[r["arm"]] = float(r["ari"])
    return [
        (v["ungated_gk0"], v["gated_gk10"], s)
        for s, v in by_seed.items()
        if {"ungated_gk0", "gated_gk10"} <= v.keys()
    ]


def _dlpfc_pairs(slide: str) -> list[tuple[float, float, int]]:
    rows = json.loads((TAB / "dlpfc_seed_stability_reply8.json").read_text(encoding="utf-8"))
    return [(float(r["ari_ungated"]), float(r["ari_gated"]), int(r["seed"])) for r in rows if r["slide"] == slide]


def ari_sources() -> dict[str, dict]:
    return {
        "BlockA": {"pairs": [(0.562, 0.617, 39788)], "src": "Block A freeze"},
        "CytAssist_Patho": {"pairs": [(0.3430, 0.3370, 39788)], "src": "extension_results_live.md"},
        "Wu": {"pairs": [(0.4143, 0.4681, 39788)], "src": "extension_results_live.md"},
        "DLPFC151673": {"pairs": _dlpfc_pairs("DLPFC151673"), "src": "dlpfc_seed_stability_reply8.json"},
        "DLPFC151507": {"pairs": _dlpfc_pairs("DLPFC151507"), "src": "dlpfc_seed_stability_reply8.json"},
        "Liver_Yu_L5": {"pairs": _phase1_pairs("reply9_phase1b_liver_yu_l5.json"), "src": "reply9_phase1b_liver_yu_l5.json"},
        "Kidney_DKD1": {"pairs": _phase1_pairs("reply9_phase1c_kidney_dkd1.json"), "src": "reply9_phase1c_kidney_dkd1.json"},
        "Xenium_Rep1_IDC": {"pairs": _phase1_pairs("reply9_phase1a_xenium_rep1.json"), "src": "reply9_phase1a_xenium_rep1.json"},
        "Hartman_BCSA_A1": {"pairs": _phase1_pairs("reply9_phase1a_hartman_bcsa_a1.json"), "src": "reply9_phase1a_hartman_bcsa_a1.json"},
        "CRC_Valdeolivas": {"pairs": _crc_pairs("valdeolivas_strategy_a_results.json"), "src": "valdeolivas_strategy_a_results.json"},
        "CRC_Valdeolivas_Rep2": {"pairs": _crc_pairs("valdeolivas_rep2_strategy_a_results.json"), "src": "valdeolivas_rep2_strategy_a_results.json"},
        "HPR_Prostate": {"pairs": [(0.1727, 0.2003, 39788)], "src": "extension_results_crc_hpr.md"},
    }


def ari_direction(pairs: list[tuple[float, float, int]]) -> dict:
    if not pairs:
        return {"label": "n/a", "n_up": 0, "n": 0, "median_delta": None, "floor": None}
    deltas = [g - u for u, g, _ in pairs]
    n_up = sum(d > 0 for d in deltas)
    med = statistics.median(deltas)
    floor = max(u for u, _, _ in pairs) < FLOOR_ARI
    if abs(med) < FLAT_DELTA:
        label = "flat"
    elif n_up == len(deltas):
        label = "up"
    elif n_up == 0:
        label = "down"
    else:
        label = "mixed"
    if floor:
        label += " (floor)"
    return {"label": label, "n_up": n_up, "n": len(deltas), "median_delta": med, "floor": floor}


def consistency(switch: str, direction: str) -> str:
    if switch == "N/A":
        return "rule undefined"
    base = direction.split(" ")[0]
    if "(floor)" in direction:
        return "uninterpretable (ARI at floor)"
    if switch == "ON":
        return "consistent" if base == "up" else "INCONSISTENT"
    return "consistent" if base in {"down", "flat", "mixed"} else "INCONSISTENT"


def components(n, nn, gene, rule) -> dict:
    return {
        "n_collide": n,
        "nn_mult_median": nn,
        "gene_cos_median": gene,
        "pass_n": n is not None and n >= rule.SKIP_RULE_N_MIN,
        "pass_nn": nn is not None and nn >= rule.SKIP_RULE_NN_MIN,
        "pass_gene": gene is not None and gene <= rule.SKIP_RULE_GENE_MAX,
    }


def main() -> None:
    if "stamp" not in sys.modules:
        pkg = types.ModuleType("stamp")
        pkg.__path__ = [str(ROOT / "stamp")]
        sys.modules["stamp"] = pkg
    _load("stamp.graph_audit", ROOT / "stamp" / "graph_audit" / "__init__.py")
    rule = _load("stamp.graph_audit.core", ROOT / "stamp" / "graph_audit" / "core.py")
    _load("stamp.edges", ROOT / "stamp" / "edges.py")
    bench = _load("stamp.graph_audit_benchmark", ROOT / "stamp" / "graph_audit_benchmark.py")
    bench.setup_env()
    from scripts.audit.run_strategy_a_extreme_harm import audit_fm_only

    registry = {s["id"]: s for s in bench.load_slide_registry()}
    ari = ari_sources()
    out_rows = []
    for sid, name in SLIDES:
        spec = registry[sid]
        print(f"==> {sid}", flush=True)
        try:
            rows = audit_fm_only(bench, spec)
        except Exception as e:
            print(f"  ERR {type(e).__name__}: {e}", flush=True)
            out_rows.append({"slide": sid, "name": name, "ok": False, "error": f"{type(e).__name__}: {e}"})
            continue
        u = next(r for r in rows if "ungated skip" in str(r.get("method", "")))
        g = next(r for r in rows if str(r.get("method", "")).startswith("STAMP gene-gated skip"))
        if spec["primary"] == "cross":
            switch = "N/A"
            comp = components(u.get("n_idc_dcis"), u.get("nn_mult_idc_dcis_median"), u.get("gene_cos_idc_dcis_median"), rule)
            cross = components(
                u.get("n_cross_coarse"), u.get("nn_mult_cross_median"), u.get("gene_cos_cross_median"), rule
            )
            cross["would_be_on"] = bool(cross["pass_n"] and cross["pass_nn"] and cross["pass_gene"])
        else:
            n = int(u.get("n_idc_dcis") or 0)
            nn = u.get("nn_mult_idc_dcis_median")
            gene = u.get("gene_cos_idc_dcis_median")
            on = rule.skip_rule_on(n, float(nn or 0), float(gene if gene is not None else float("nan")))
            switch = "ON" if on else "OFF"
            comp = components(n, nn, gene, rule)
            cross = None
        a = ari[sid]
        d = ari_direction(a["pairs"])
        row = {
            "slide": sid,
            "name": name,
            "ok": True,
            "switch": switch,
            "components": comp,
            "cross_layer_components": cross,
            "ungated_LR": [u.get("primary_lookalike"), u.get("primary_remote")],
            "gated_LR": [g.get("primary_lookalike"), g.get("primary_remote")],
            "ari_pairs": [{"seed": s, "ungated": uu, "gated": gg} for uu, gg, s in a["pairs"]],
            "ari_source": a["src"],
            "ari_direction": d,
            "consistency": consistency(switch, d["label"]),
        }
        out_rows.append(row)
        print(
            f"  switch={switch} n={comp['n_collide']} nn={comp['nn_mult_median']} gene={comp['gene_cos_median']} "
            f"ARI={d['label']} -> {row['consistency']}",
            flush=True,
        )

    report = {
        "phase": "Reply-10 Phase 2 Step 1",
        "rule": f"n_collide>={rule.SKIP_RULE_N_MIN} AND nn_mult>={rule.SKIP_RULE_NN_MIN} AND gene_cos<={rule.SKIP_RULE_GENE_MAX}",
        "embedding": "local Phikon-v2 (phikonv2_emb.npy)",
        "ari_floor": FLOOR_ARI,
        "flat_delta": FLAT_DELTA,
        "rows": out_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (TAB / "reply10_phase2_step1_switch.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_md(report)


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def write_md(report: dict) -> None:
    lines = [
        "# Reply-10 Phase 2 Step 1 — existing skip rule on all datasets",
        "",
        f"> Rule (unchanged): **{report['rule']}** on primary-collide edges of the ungated FM skip graph.",
        f"> Embedding: {report['embedding']}. Graph-only; ARI read from existing tables (no retraining).",
        f"> ARI direction: gated − ungated per seed; `flat` if |median Δ| < {report['flat_delta']}; "
        f"`(floor)` if ungated ARI < {report['ari_floor']}.",
        "",
        "| Dataset | Switch | n_collide | nn_mult | gene_cos | Ungated L/R | Gated L/R | ARI ungated→gated (seeds) | Direction | Consistent? |",
        "|---------|--------|----------:|--------:|---------:|------------:|----------:|---------------------------|-----------|-------------|",
    ]
    for r in report["rows"]:
        if not r.get("ok"):
            lines.append(f"| {r['name']} | ERR | — | — | — | — | — | — | — | {r.get('error')} |")
            continue
        c = r["components"]
        mark = lambda ok, v, nd=3: (f"**{_f(v, nd)}**" if ok else _f(v, nd)) if v is not None else "—"
        ari_txt = " · ".join(f"{p['ungated']:.3f}→{p['gated']:.3f}" for p in r["ari_pairs"]) or "—"
        d = r["ari_direction"]
        lines.append(
            f"| {r['name']} | **{r['switch']}** | {c['n_collide'] if c['n_collide'] is not None else '—'} | "
            f"{mark(c['pass_nn'], c['nn_mult_median'], 2)} | {mark(c['pass_gene'], c['gene_cos_median'])} | "
            f"{r['ungated_LR'][0]} / {r['ungated_LR'][1]} | {r['gated_LR'][0]} / {r['gated_LR'][1]} | "
            f"{ari_txt} | {d['label']} ({d['n_up']}/{d['n']}↑) | {r['consistency']} |"
        )
    lines += ["", "Bold component = passes its threshold.", ""]
    cross_rows = [r for r in report["rows"] if r.get("ok") and r.get("cross_layer_components")]
    if cross_rows:
        lines += [
            "## DLPFC: rule undefined as-is (no primary collide pair)",
            "",
            "Descriptive only — the same three thresholds applied to all cross-layer edges:",
            "",
            "| Dataset | n_cross | nn_mult | gene_cos | Would be ON? |",
            "|---------|--------:|--------:|---------:|--------------|",
        ]
        for r in cross_rows:
            c = r["cross_layer_components"]
            lines.append(
                f"| {r['name']} | {c['n_collide']} | {_f(c['nn_mult_median'], 2)} | {_f(c['gene_cos_median'])} | "
                f"{'yes' if c['would_be_on'] else 'no'} |"
            )
        lines.append("")
    lines += ["ARI sources: " + "; ".join(f"{r['name']}: `{r['ari_source']}`" for r in report["rows"] if r.get("ok")), ""]
    (TAB / "reply10_phase2_step1_switch.md").write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", TAB / "reply10_phase2_step1_switch.md", flush=True)


if __name__ == "__main__":
    main()
