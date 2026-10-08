"""Part 3 stage A — edge-level audit and switch decision per dataset x FM. Graph only, no training.

Exactly the Step 1 audit (run_phase2_step1_switch_validation -> run_strategy_a_extreme_harm.audit_fm_only, with the
review-fix Hartman label patch), with phikonv2_emb.npy redirected in memory to <fm>_prereg_emb.npy. The switch rule
and all thresholds are unchanged. --fm phikonv2 reproduces the Step 1 switch table (sanity check).

The switch rule uses n_collide, nn_mult and gene_cos only; tau_fm enters the lookalike counts, not the switch. The
percentile-matched tau_fm (secondary) is therefore reported as extra lookalike counts: per slide, tau_fm' is the
quantile of this FM's pairwise cosine (PCA-16, all spot pairs) at the fraction of Phikon-v2 pairs with cosine >= 0.80.

Output (new): logs/prereg/part3_stageA/<fm>.json

  python scripts/prereg/run_part3_stageA_audit.py --fm phikon
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--fm", default="phikonv2")
_fm = _ap.parse_known_args()[0].fm
os.environ["STAIG_FIX2_FM"] = "phikonv2_fix2_emb.npy" if _fm == "phikonv2" else f"{_fm}_prereg_emb.npy"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _prereg_common as C  # noqa: E402

import json  # noqa: E402
import types  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

OUT_DIR = C.LOG / "part3_stageA"
ORDER = ["BlockA", "CytAssist_Patho", "Wu", "DLPFC151673", "DLPFC151507", "Liver_Yu_L5", "Kidney_DKD1",
         "Xenium_Rep1_IDC", "Hartman_BCSA_A1", "CRC_Valdeolivas", "CRC_Valdeolivas_Rep2", "HPR_Prostate"]


def _pair_cos(path: Path, n: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = np.load(path).reshape(n, -1).astype(np.float64)
    x = StandardScaler().fit_transform(x)
    if x.shape[1] > 16:
        x = StandardScaler().fit_transform(PCA(n_components=16, random_state=42).fit_transform(x))
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    s = x @ x.T
    return s[np.triu_indices(n, 1)]


def matched_tau(ds: str) -> dict:
    sd = C.slide_dir(ds)
    ref = sd / ("phikonv2_fix2_emb.npy" if (sd / "phikonv2_fix2_emb.npy").is_file() else "phikonv2_xy_emb.npy")
    new = sd / C.expected_fm(ds)
    n = np.load(ref, mmap_mode="r").shape[0]
    frac = float((_pair_cos(ref, n) >= 0.80).mean())
    tau = float(np.quantile(_pair_cos(new, n), 1.0 - frac)) if frac > 0 else float("inf")
    return {"phikonv2_frac_ge_0.80": frac, "tau_fm_matched": tau}


def lookalike_at(df, tau: float, relation: str) -> int:
    rel = (df["relation"] != "same") if relation == "cross" else (df["relation"] == "idc_dcis")
    look = (df["gene_cos"] <= 0.20) & (df["fm_cos"] >= tau)
    return int((look & rel).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fm", default="phikonv2", choices=["phikonv2", "phikon", "plip", "uni", "conch"])
    ap.add_argument("--check", action="store_true", help="only check that the FM file exists on every slide")
    args = ap.parse_args()
    for ds in ORDER:
        print(f"{ds:22s} -> {C.expected_fm(ds)}", flush=True)
    if args.check:
        return

    if "stamp" not in sys.modules:
        pkg = types.ModuleType("stamp")
        pkg.__path__ = [str(C.ROOT / "stamp")]
        sys.modules["stamp"] = pkg
    C.load_module("stamp.graph_audit", C.ROOT / "stamp/graph_audit/__init__.py")
    rule = C.load_module("stamp.graph_audit.core", C.ROOT / "stamp/graph_audit/core.py")
    C.load_module("stamp.edges", C.ROOT / "stamp/edges.py")
    bench = C.load_module("stamp.graph_audit_benchmark", C.ROOT / "stamp/graph_audit_benchmark.py")
    bench.setup_env()
    swf2 = C.load_module("swf2", C.ROOT / "scripts/review_fix/run_phase2_step1_switch_fix2.py")
    swf2._audit_patch()
    import scripts.audit.run_strategy_a_extreme_harm as xh

    captured = {}
    rule_score = rule.score_graph

    def score_graph_capture(adata, adj, method, **kw):
        s, df = rule_score(adata, adj, method, **kw)
        captured[method] = df
        return s, df

    registry = {s["id"]: s for s in bench.load_slide_registry()}
    rows = []
    for ds in ORDER:
        spec = registry[ds]
        print(f"==> {ds}", flush=True)
        captured.clear()
        try:
            rule.score_graph = score_graph_capture
            try:
                res = xh.audit_fm_only(bench, spec)
            finally:
                rule.score_graph = rule_score
            used = C.check_redirect(ds)
            u = next(r for r in res if "ungated skip" in str(r.get("method", "")))
            g = next(r for r in res if str(r.get("method", "")).startswith("STAMP gene-gated skip"))
            if spec["primary"] == "cross":
                switch = "N/A"
                n, nn, gene = u.get("n_cross_coarse"), u.get("nn_mult_cross_median"), u.get("gene_cos_cross_median")
            else:
                n, nn, gene = int(u.get("n_idc_dcis") or 0), u.get("nn_mult_idc_dcis_median"), u.get("gene_cos_idc_dcis_median")
                on = rule.skip_rule_on(n, float(nn or 0), float(gene if gene is not None else float("nan")))
                switch = "ON" if on else "OFF"
            row = {"dataset": ds, "fm": args.fm, "ok": True, "switch": switch, "primary": spec["primary"],
                   "n_collide": n, "nn_mult_median": nn, "gene_cos_median": gene,
                   "ungated_lookalike": u.get("primary_lookalike"), "ungated_remote": u.get("primary_remote"),
                   "gated_lookalike": g.get("primary_lookalike"), "gated_remote": g.get("primary_remote"),
                   "n_edges_ungated": u.get("n_edges"), "n_edges_gated": g.get("n_edges"), "fm_redirects": used}
            if args.fm != "phikonv2":
                mt = matched_tau(ds)
                row.update(mt)
                du = next((d for m, d in captured.items() if "ungated skip" in m), None)
                dg = next((d for m, d in captured.items() if m.startswith("STAMP gene-gated skip")), None)
                if du is not None and dg is not None:
                    row["ungated_lookalike_matched"] = lookalike_at(du, mt["tau_fm_matched"], spec["primary"])
                    row["gated_lookalike_matched"] = lookalike_at(dg, mt["tau_fm_matched"], spec["primary"])
            print(f"  switch={switch} n={n} nn={nn} gene={gene} LR {row['ungated_lookalike']}/{row['ungated_remote']}"
                  f" -> {row['gated_lookalike']}/{row['gated_remote']}", flush=True)
        except Exception as e:
            row = {"dataset": ds, "fm": args.fm, "ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"  ERR {row['error']}", flush=True)
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = C.new_path(OUT_DIR / f"{args.fm}.json")
    out.write_text(json.dumps({"fm": args.fm, "fm_file": os.environ["STAIG_FIX2_FM"],
                               "rule": f"n_collide>={rule.SKIP_RULE_N_MIN} AND nn_mult>={rule.SKIP_RULE_NN_MIN} "
                                       f"AND gene_cos<={rule.SKIP_RULE_GENE_MAX}",
                               "rows": rows, "time": datetime.now(timezone.utc).isoformat()}, indent=2, default=str),
                   encoding="utf-8")
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
