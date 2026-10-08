"""Strategy A extreme harm — FM skip arms only (no GraphST/STAGATE baselines).

Bypasses stamp/__init__ and skips third-party graph builders that fail locally.
Fixed thresholds: τ_gene=0.20, τ_fm=0.80, τ_remote=3.0, gene_k=10.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
TAB = ROOT / "lab/static/tables"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def audit_fm_only(bench, spec: dict) -> list[dict]:
    """Score ungated + gene-gated FM skip only (Strategy A need)."""
    from stamp.edges import add_fm_skip_edges
    from stamp.graph_audit.core import score_graph, skip_rule_on, ensure_feat

    path = spec["path"]
    truth_file = spec.get("truth_file", "truth.txt")
    ref = bench.load_staig(path, spec["kind"], truth_file)
    raw = bench.load_raw(path, spec["kind"], truth_file)
    names = [b for b in ref.obs_names if b in raw.obs_names]
    ref = ref[names].copy()
    raw = raw[names].copy()
    ensure_feat(raw)
    if "feat" not in ref.obsm:
        ref.obsm["feat"] = raw.obsm["feat"]
    fk = bench.fm_key(ref)
    if not fk:
        raise RuntimeError("no phikonv2_emb / phikon_emb on slide")

    rows: list[dict] = []
    g0 = np.array(ref.obsm["graph_neigh"], dtype=np.float64, copy=True)
    ep0 = np.array(ref.obsm["edge_probabilities"], dtype=np.float64, copy=True)

    def add(method: str, adj, notes: str):
        summary, _ = score_graph(
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
        bench._attach_primary(summary, spec)
        rows.append(summary)

    ad_u = ref.copy()
    ad_u.obsm["graph_neigh"] = g0.copy()
    ad_u.obsm["edge_probabilities"] = ep0.copy()
    add_fm_skip_edges(ad_u, fk, k=1, drop_prob=0.05, pca_dim=16)
    add(
        f"STAIG+ungated skip ({fk})",
        np.asarray(ad_u.obsm["graph_neigh"]),
        f"skip k=1 {fk}; Strategy A extreme FM-only",
    )

    ungated = next((r for r in rows if "ungated skip" in r.get("method", "")), None)
    if ungated and spec["primary"] == "idc_dcis":
        ungated["skip_rule_on"] = skip_rule_on(
            int(ungated.get("n_idc_dcis") or 0),
            float(ungated.get("nn_mult_idc_dcis_median") or 0),
            float(ungated.get("gene_cos_idc_dcis_median") or float("nan")),
        )

    ad_g = ref.copy()
    ad_g.obsm["graph_neigh"] = g0.copy()
    ad_g.obsm["edge_probabilities"] = ep0.copy()
    add_fm_skip_edges(
        ad_g, fk, k=1, drop_prob=0.05, pca_dim=16, gene_key="feat", gene_k=10, gene_mutual=True
    )
    note = "gene_k=10 mutual"
    if spec["primary"] == "cross":
        note += " (cross-layer)"
    add("STAMP gene-gated skip", np.asarray(ad_g.obsm["graph_neigh"]), note)
    return rows


def main():
    if "stamp" not in sys.modules:
        pkg = types.ModuleType("stamp")
        pkg.__path__ = [str(ROOT / "stamp")]
        sys.modules["stamp"] = pkg

    _load("stamp.graph_audit", ROOT / "stamp" / "graph_audit" / "__init__.py")
    _load("stamp.graph_audit.core", ROOT / "stamp" / "graph_audit" / "core.py")
    try:
        _load("stamp.edges", ROOT / "stamp" / "edges.py")
    except Exception as e:
        print("WARN stamp.edges:", e, flush=True)
        raise

    bench = _load("stamp.graph_audit_benchmark", ROOT / "stamp" / "graph_audit_benchmark.py")
    bench.setup_env()
    wanted = {"HPR_Prostate"}
    specs = [s for s in bench.load_slide_registry() if s["id"] in wanted]
    rows_out = []
    for spec in specs:
        print(f"==> audit FM-only {spec['id']}", flush=True)
        path = spec["path"]
        print(
            f"  phikonv2={(path / 'phikonv2_emb.npy').is_file()} "
            f"phikon={(path / 'phikon_emb.npy').is_file()}",
            flush=True,
        )
        try:
            rows = audit_fm_only(bench, spec)
            u = next((r for r in rows if "ungated skip" in str(r.get("method", ""))), None)
            g = next(
                (
                    r
                    for r in rows
                    if str(r.get("method", "")).startswith("STAMP gene-gated skip")
                    and "max_hops" not in str(r.get("method", ""))
                ),
                None,
            )
            # HPR / non-breast: breast skip rule stays OFF when no IDC–DCIS/Tumor–Stroma primary
            if u is not None and spec.get("primary") == "cross":
                u["skip_rule_on"] = False
            rows_out.append(
                {
                    "slide": spec["id"],
                    "ok": True,
                    "skip_rule_on": None if u is None else u.get("skip_rule_on"),
                    "ungated_lookalike": None if u is None else u.get("primary_lookalike"),
                    "ungated_remote": None if u is None else u.get("primary_remote"),
                    "gated_lookalike": None if g is None else g.get("primary_lookalike"),
                    "gated_remote": None if g is None else g.get("primary_remote"),
                    "ungated_method": None if u is None else u.get("method"),
                    "fm_key": None if u is None else u.get("notes", ""),
                    "role": spec.get("role", ""),
                    "error": None if (u or g) else "no FM skip arms",
                }
            )
            print(
                f"  skip={rows_out[-1]['skip_rule_on']} "
                f"U={rows_out[-1]['ungated_lookalike']}/{rows_out[-1]['ungated_remote']} "
                f"G={rows_out[-1]['gated_lookalike']}/{rows_out[-1]['gated_remote']}",
                flush=True,
            )
        except Exception as e:
            rows_out.append({"slide": spec["id"], "ok": False, "error": f"{type(e).__name__}: {e}"})
            print(f"  ERR {e}", flush=True)

    out = {
        "threshold_strategy": "A_fixed",
        "tau_gene": 0.20,
        "tau_fm": 0.80,
        "tau_remote": 3.0,
        "gene_k": 10,
        "mode": "fm_skip_arms_only_crc_hpr",
        "slides": rows_out,
    }
    (TAB / "strategy_a_crc_hpr_harm.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    # also merge into extreme harm if present
    extreme_path = TAB / "strategy_a_extreme_harm.json"
    if extreme_path.is_file():
        extreme = json.loads(extreme_path.read_text(encoding="utf-8"))
        by = {s["slide"]: s for s in extreme.get("slides", [])}
        for r in rows_out:
            by[r["slide"]] = r
        extreme["slides"] = list(by.values())
        extreme_path.write_text(json.dumps(extreme, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Strategy A — CRC + HPR harm (fixed thresholds)",
        "",
        "> τ_gene=0.20, τ_fm=0.80, τ_remote=3.0, gene_k=10. No retune.",
        "> CRC: 10x Parent Visium CRC; primary collide = **Tumor–Stroma** (graphclust→marker remap).",
        "> HPR: 10x Normal Prostate FFPE; primary = cross-domain; expected Skip-OFF.",
        "",
        "| slide | skip_rule | ungated look/remote | gated look/remote | role |",
        "|-------|-----------|--------------------:|------------------:|------|",
    ]
    for r in rows_out:
        if not r.get("ok"):
            lines.append(f"| {r['slide']} | ERR | — | — | {r.get('error')} |")
            continue
        lines.append(
            f"| {r['slide']} | {r.get('skip_rule_on')} | "
            f"{r.get('ungated_lookalike')} / {r.get('ungated_remote')} | "
            f"{r.get('gated_lookalike')} / {r.get('gated_remote')} | "
            f"{r.get('role') or ''} |"
        )
    (TAB / "strategy_a_crc_hpr_harm.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE", TAB / "strategy_a_crc_hpr_harm.md", flush=True)
    print(json.dumps(rows_out, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
