"""Reply-10 Phase 2 Step 1 — frozen switch rule + ARI direction after the code-review fixes.

Same rule, audit, direction and consistency logic as scripts/xyfix/run_phase2_step1_switch_xy.py. Differences,
all in memory (scripts/review_fix/_fix2_env.py):
  Xenium_Rep1_IDC / CRC_Valdeolivas / _Rep2  Phikon-v2 = phikonv2_fix2_emb.npy
  Hartman_BCSA_A1                            labels = truth_filled.txt, Phikon-v2 = phikonv2_fix2_emb.npy
ARI pairs for those four slides come from logs/review_fix/reply10_phase2_multiseed_fix2.jsonl; all other slides
keep the corrected-coordinate (xy) rows. Block A is reported on both seed sets: the 5 seeds pre-specified for
Step 1 (0, 1, 42, 123, 39788) and all 10 seeds.
Outputs (new): lab/static/tables/review_fix/reply10_phase2_step1_switch_fix2.{md,json},
               logs/review_fix/reply10_phase2_multiseed_merged_fix2.jsonl
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fix2_env  # noqa: E402

spec = importlib.util.spec_from_file_location("swxy", ROOT / "scripts/xyfix/run_phase2_step1_switch_xy.py")
swxy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(swxy)
sv, tbl = swxy.sv, swxy.tbl

OUT = ROOT / "lab/static/tables/review_fix"
STEM = "reply10_phase2_step1_switch_fix2"
XY_MERGED = ROOT / "logs/xyfix/reply10_phase2_multiseed_merged.jsonl"
FIX2_JSONL = ROOT / "logs/review_fix/reply10_phase2_multiseed_fix2.jsonl"
MERGED = ROOT / "logs/review_fix/reply10_phase2_multiseed_merged_fix2.jsonl"
PRESPEC_SEEDS = (0, 1, 42, 123, 39788)


def merged_rows() -> list[dict]:
    keep = [r for r in swxy._rows(XY_MERGED) if r["dataset"] not in _fix2_env.FIX2_SLIDES]
    new = [r for r in swxy._rows(FIX2_JSONL) if r.get("ok", True)]
    MERGED.parent.mkdir(parents=True, exist_ok=True)
    MERGED.write_text("\n".join(json.dumps(r, default=str) for r in keep + new) + "\n", encoding="utf-8")
    return keep + new


def ari_sources() -> dict[str, dict]:
    tab = sv.TAB
    sv.TAB = swxy.TAB_XY
    try:
        out = swxy.ari_sources()
    finally:
        sv.TAB = tab
    for a in out.values():
        if a["src"] == "logs/xyfix/reply10_phase2_multiseed_merged.jsonl":
            a["src"] = str(MERGED.relative_to(ROOT)).replace("\\", "/")
    return out


def _audit_patch() -> None:
    import scripts.audit.run_strategy_a_extreme_harm as xh

    afo = xh.audit_fm_only

    def audit_fm_only(bench, spec):
        spec = dict(spec)
        name = _fix2_env.TRUTH_OVERRIDE.get(Path(spec["path"]).name)
        if name is None:
            return afo(bench, spec)
        spec["truth_file"] = name
        load_staig = bench.load_staig
        bench.load_staig = lambda path, kind, truth_file="truth.txt": load_staig(path, kind, "truth.txt")
        try:
            return afo(bench, spec)
        finally:
            bench.load_staig = load_staig

    xh.audit_fm_only = audit_fm_only


def blocka_seed_sets() -> list[dict]:
    by = {}
    for r in swxy._rows(MERGED):
        if r["dataset"] == "BlockA":
            by.setdefault(int(r["seed"]), {})[r["arm"]] = float(r["ari"])
    out = []
    for label, seeds in (("pre-specified 5 seeds", PRESPEC_SEEDS), ("all 10 seeds", tuple(sorted(by)))):
        pairs = [(by[s]["ungated"], by[s]["gated"], s) for s in seeds if {"ungated", "gated"} <= by.get(s, {}).keys()]
        d = swxy.ari_direction(pairs)
        out.append({"seed_set": label, "seeds": [s for _, _, s in pairs], "direction": d,
                    "consistency": swxy.consistency("ON", d["label"])})
    return out


def main() -> None:
    _fix2_env.check_fix2_files()
    _audit_patch()
    swxy.merged_rows = merged_rows
    sv.SLIDES = sv.SLIDES + [("Liver_H36", "Liver H36"), ("CRC_Parent", "CRC Parent (10x)")]
    sv.ari_sources = ari_sources
    sv.ari_direction = swxy.ari_direction
    sv.consistency = swxy.consistency
    OUT.mkdir(parents=True, exist_ok=True)
    sv.TAB = OUT
    sv.main()

    src_json, src_md = OUT / "reply10_phase2_step1_switch.json", OUT / "reply10_phase2_step1_switch.md"
    report = json.loads(src_json.read_text(encoding="utf-8"))
    report["embedding"] = ("Phikon-v2: phikonv2_fix2_emb.npy on Xenium / Hartman / CRC Valdeolivas Rep1+Rep2, "
                           "phikonv2_xy_emb.npy where the original was transposed, else phikonv2_emb.npy")
    report["labels"] = "Hartman A1: truth_filled.txt (filled pathologist outlines); others unchanged"
    report["redirects"] = _fix2_env.redirects()
    report["direction_rule"] = f">= {int(tbl.AGREE_FRAC * 100)} % of seeds same sign (all seeds when n < 5)"
    report["ari_floor_rule"] = f"median ungated or gated ARI < {tbl.FLOOR}"
    report["blocka_seed_sets"] = blocka_seed_sets()
    report.pop("flat_delta", None)
    (OUT / f"{STEM}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    report["flat_delta"] = None
    sv.write_md(report)
    lines = src_md.read_text(encoding="utf-8").splitlines()
    lines[0] = "# Reply-10 Phase 2 Step 1 — existing skip rule on all datasets (after code-review fixes)"
    for i, ln in enumerate(lines):
        if ln.startswith("> Embedding:"):
            lines[i] = (f"> Embedding: {report['embedding']}. {report['labels']}. Switch is graph-only; "
                        "ARI from the reruns listed under sources.")
        elif ln.startswith("> ARI direction:"):
            lines[i] = (f"> ARI direction: gated − ungated per seed; `up`/`down` if {report['direction_rule']}, "
                        f"else `mixed`; `(floor)` if {report['ari_floor_rule']}. Same rule as the multi-seed table.")
    lines += ["", "## Block A on both seed sets", "",
              "| Seed set | Seeds | ARI direction | Consistent? |", "|---|---|---|---|"]
    for b in report["blocka_seed_sets"]:
        d = b["direction"]
        lines.append(f"| {b['seed_set']} | {', '.join(map(str, b['seeds']))} | {d['label']} ({d['n_up']}/{d['n']}↑, "
                     f"median Δ {d['median_delta']:+.3f}) | {b['consistency']} |")
    lines += ["", "The 5-seed set was fixed before Step 1 was run; the 10-seed set adds 2, 3, 7, 99, 2023.", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines), encoding="utf-8")
    src_json.unlink()
    src_md.unlink()
    print("WROTE", OUT / f"{STEM}.md", flush=True)


if __name__ == "__main__":
    main()
