"""Reply-10 Phase 2 Step 1 — frozen switch rule + ARI direction on corrected-coordinate Phikon-v2.

Reuses run_phase2_step1_switch_validation (rule, audit and consistency logic) unchanged.
Differences: phikonv2_emb.npy redirected in memory to phikonv2_xy_emb.npy; ARI pairs come from the
corrected reruns (merged multiseed log, DLPFC xy, Liver H36 xy, CRC Parent xy); Liver H36 and CRC Parent
added; ARI direction uses the multi-seed table's pre-specified rule (>=80 % of seeds same sign, floor =
median ARI < 0.05) so both Step 1 tables agree (identical to the original rule when n = 1).
Also writes the merged multiseed log used by the table builder.
Outputs (new): lab/static/tables/xyfix/reply10_phase2_step1_switch.{md,json},
               logs/xyfix/reply10_phase2_multiseed_merged.jsonl
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xyfix"))
import _xy_fm_redirect  # noqa: E402

_xy_fm_redirect.install()
spec = importlib.util.spec_from_file_location("sv_orig", ROOT / "scripts/phase2_reply10/run_phase2_step1_switch_validation.py")
sv = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sv)
spec = importlib.util.spec_from_file_location("tbl_orig", ROOT / "scripts/phase2_reply10/build_phase2_step1_multiseed_table.py")
tbl = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tbl)

TAB_XY = ROOT / "lab/static/tables/xyfix"
CRC_PARENT_XY = ROOT / "logs/xyfix/strategy_a_crc_parent_ari_xy.jsonl"
OLD_JSONL = ROOT / "logs/reply10_phase2_multiseed.jsonl"
XY_JSONL = ROOT / "logs/reply10_phase2_multiseed_xy.jsonl"
MERGED = ROOT / "logs/xyfix/reply10_phase2_multiseed_merged.jsonl"
DLPFC_XY = TAB_XY.parent / "dlpfc_seed_stability_xy.json"
# embeddings already at the correct location, or arms that never read Phikon
KEEP_OLD = {("Hartman_BCSA_A1", None), ("Xenium_Rep1_IDC", None), ("BlockA", "none")}


def _rows(fp: Path) -> list[dict]:
    if not fp.is_file():
        return []
    return [json.loads(x) for x in fp.read_text(encoding="utf-8").splitlines() if x.strip()]


def merged_rows() -> list[dict]:
    keep = [r for r in _rows(OLD_JSONL) if r.get("ok", True)
            and ((r["dataset"], None) in KEEP_OLD or (r["dataset"], r["arm"]) in KEEP_OLD)]
    new = [r for r in _rows(XY_JSONL) if r.get("ok", True)]
    for r in keep:
        r["source"] = "original run (embedding unaffected)"
    for r in new:
        r["source"] = "xy rerun"
    MERGED.parent.mkdir(parents=True, exist_ok=True)
    MERGED.write_text("\n".join(json.dumps(r, default=str) for r in keep + new) + "\n", encoding="utf-8")
    return keep + new


def ari_sources() -> dict[str, dict]:
    by: dict[str, dict[int, dict[str, float]]] = {}
    for r in merged_rows():
        by.setdefault(r["dataset"], {}).setdefault(int(r["seed"]), {})[r["arm"]] = float(r["ari"])
    out = {}
    for sid, _ in sv.SLIDES:
        seeds = by.get(sid, {})
        pairs = [(v["ungated"], v["gated"], s) for s, v in sorted(seeds.items()) if {"ungated", "gated"} <= v.keys()]
        out[sid] = {"pairs": pairs, "src": "logs/xyfix/reply10_phase2_multiseed_merged.jsonl"}
    if DLPFC_XY.is_file():
        rows = json.loads(DLPFC_XY.read_text(encoding="utf-8"))
        for sid in ("DLPFC151673", "DLPFC151507"):
            out[sid] = {"pairs": [(r["ari_ungated"], r["ari_gated"], r["seed"]) for r in rows if r["slide"] == sid],
                        "src": "dlpfc_seed_stability_xy.json"}
    out.setdefault("Liver_H36", {"pairs": [], "src": "—"})
    out.setdefault("CRC_Parent", {"pairs": [], "src": "—"})
    h36 = TAB_XY / "reply9_phase1b_liver_h36.json"
    if h36.is_file():
        out["Liver_H36"] = {"pairs": sv._phase1_pairs("reply9_phase1b_liver_h36.json"),
                            "src": "xyfix/reply9_phase1b_liver_h36.json"}
    crc = {r["arm"]: r for r in _rows(CRC_PARENT_XY)}
    if {"ungated_gk0", "gated_gk10"} <= crc.keys():
        out["CRC_Parent"] = {
            "pairs": [(float(crc["ungated_gk0"]["ari"]), float(crc["gated_gk10"]["ari"]),
                       int(crc["gated_gk10"].get("seed", 39788)))],
            "src": "logs/xyfix/strategy_a_crc_parent_ari_xy.jsonl",
        }
    return out


def ari_direction(pairs: list[tuple[float, float, int]]) -> dict:
    if not pairs:
        return {"label": "n/a", "n_up": 0, "n": 0, "median_delta": None, "floor": None}
    d = tbl.direction([g - u for u, g, _ in pairs])
    label = {"up": "up", "down": "down"}.get(d["label"], "mixed")
    floor = min(sv.statistics.median(u for u, _, _ in pairs),
                sv.statistics.median(g for _, g, _ in pairs)) < tbl.FLOOR
    if floor:
        label += " (floor)"
    return {"label": label, "n_up": d["n_up"], "n": d["n"], "need": d["need"],
            "median_delta": d["median"], "wilcoxon_p": d["wilcoxon_p"], "floor": floor}


_consistency_orig = sv.consistency


def consistency(switch: str, direction: str) -> str:
    if direction.startswith("n/a") and switch != "N/A":
        return "no ARI yet"
    return _consistency_orig(switch, direction)


def main() -> None:
    sv.SLIDES = sv.SLIDES + [("Liver_H36", "Liver H36"), ("CRC_Parent", "CRC Parent (10x)")]
    sv.ari_sources = ari_sources
    sv.ari_direction = ari_direction
    sv.consistency = consistency
    TAB_XY.mkdir(parents=True, exist_ok=True)
    sv.TAB = TAB_XY
    sv.main()
    fp = TAB_XY / "reply10_phase2_step1_switch.json"
    report = json.loads(fp.read_text(encoding="utf-8"))
    report["embedding"] = "corrected-coordinate Phikon-v2 (phikonv2_xy_emb.npy where the original was transposed)"
    report["redirects"] = _xy_fm_redirect.redirects()
    report["direction_rule"] = f">= {int(tbl.AGREE_FRAC * 100)} % of seeds same sign (all seeds when n < 5)"
    report["ari_floor_rule"] = f"median ungated or gated ARI < {tbl.FLOOR}"
    report.pop("flat_delta", None)
    fp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["flat_delta"] = None
    sv.write_md(report)
    md = TAB_XY / "reply10_phase2_step1_switch.md"
    lines = md.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("> Embedding:"):
            lines[i] = (f"> Embedding: {report['embedding']}. Switch is graph-only; ARI from the "
                        "corrected-coordinate reruns (see sources below).")
        elif ln.startswith("> ARI direction:"):
            lines[i] = (f"> ARI direction: gated − ungated per seed; `up`/`down` if {report['direction_rule']}, "
                        f"else `mixed`; `(floor)` if {report['ari_floor_rule']}. Same rule as the multi-seed table.")
    md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
