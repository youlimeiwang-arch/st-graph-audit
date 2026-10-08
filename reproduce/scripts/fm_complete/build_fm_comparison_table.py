"""Complete 12-slide x 5-FM comparison table (edge audit + ARI), with the Phikon-v2 gaps filled.

Reads only finished logs; no training. Direction rule and floor are the ones used by the pre-registered summary.
The pre-registered decision is NOT recomputed here: it stays as in lab/static/tables/prereg/prereg_results_v2.md.

ARI sources
  phikon / plip / uni / conch    logs/prereg/part3_stageB.jsonl
  phikonv2 (non-DLPFC)           Step 1 log (merged fix2) + logs/fm_complete/phikonv2_fill.jsonl (extra seeds)
  phikonv2 DLPFC151507           logs/prereg/part1_local.jsonl (same Runner, 10 seeds)
  phikonv2 DLPFC151673           logs/fm_complete/phikonv2_fill.jsonl
Edge audit: logs/prereg/part3_stageA/<fm>.json

Outputs (new; never overwritten): lab/static/tables/fm_complete/fm_comparison_12x5.{md,csv} and
fm_comparison_runs.csv (one row per training run).
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "prereg"))
import _prereg_common as C  # noqa: E402

FMS = ["phikonv2", "phikon", "plip", "uni", "conch"]
ORDER = ["BlockA", "CytAssist_Patho", "Wu", "Kidney_DKD1", "HPR_Prostate", "CRC_Valdeolivas", "CRC_Valdeolivas_Rep2",
         "Liver_Yu_L5", "Xenium_Rep1_IDC", "Hartman_BCSA_A1", "DLPFC151507", "DLPFC151673"]
STEP1 = ROOT / "logs/review_fix/reply10_phase2_multiseed_merged_fix2.jsonl"
FILL = ROOT / "logs/fm_complete/phikonv2_fill.jsonl"
P1 = C.LOG / "part1_local.jsonl"
P3A = C.LOG / "part3_stageA"
P3B = C.LOG / "part3_stageB.jsonl"
OUT = ROOT / "lab/static/tables/fm_complete"
FLOOR = 0.05
ARROW = {"up": "↑", "down": "↓", "no stable change": "·"}


def runs() -> list[dict]:
    """One row per (dataset, fm, arm, seed); the first source listed wins on duplicates."""
    out: dict[tuple, dict] = {}

    def add(rows, fm_of, source):
        for r in rows:
            if not r.get("ok") or r.get("arm") not in ("ungated", "gated"):
                continue
            key = (r["dataset"], fm_of(r), r["arm"], int(r["seed"]))
            out.setdefault(key, {"dataset": key[0], "fm": key[1], "arm": key[2], "seed": key[3],
                                 "ari": float(r["ari"]), "source": source})

    add(C.read_jsonl(STEP1), lambda r: "phikonv2", "step1")
    add([r for r in C.read_jsonl(P1) if r["dataset"] in C.DLPFC], lambda r: "phikonv2", "prereg_part1")
    add(C.read_jsonl(FILL), lambda r: "phikonv2", "fm_complete_fill")
    add(C.read_jsonl(P3B), lambda r: r["fm"], "prereg_part3B")
    return sorted(out.values(), key=lambda r: (ORDER.index(r["dataset"]), FMS.index(r["fm"]), r["seed"], r["arm"]))


def audit() -> dict[tuple[str, str], dict]:
    out = {}
    for fm in FMS:
        p = P3A / f"{fm}.json"
        for r in json.loads(p.read_text(encoding="utf-8"))["rows"] if p.is_file() else []:
            if r.get("ok"):
                out[(r["dataset"], fm)] = r
    return out


def combos(rr: list[dict], aud: dict) -> list[dict]:
    by: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    src: dict[tuple[str, str], set[str]] = {}
    for r in rr:
        by.setdefault((r["dataset"], r["fm"]), {}).setdefault(r["seed"], {})[r["arm"]] = r["ari"]
        src.setdefault((r["dataset"], r["fm"]), set()).add(r["source"])
    out = []
    for ds in ORDER:
        for fm in FMS:
            seeds = {s: v for s, v in by.get((ds, fm), {}).items() if {"ungated", "gated"} <= v.keys()}
            a = aud.get((ds, fm), {})
            row = {"dataset": ds, "fm": fm, "switch": a.get("switch"), "n_collide": a.get("n_collide"),
                   "nn_mult_median": a.get("nn_mult_median"), "gene_cos_median": a.get("gene_cos_median"),
                   "lookalike_ungated": a.get("ungated_lookalike"), "lookalike_gated": a.get("gated_lookalike"),
                   "remote_ungated": a.get("ungated_remote"), "remote_gated": a.get("gated_remote"),
                   "tau_fm_matched": a.get("tau_fm_matched"),
                   "lookalike_matched_ungated": a.get("ungated_lookalike_matched"),
                   "lookalike_matched_gated": a.get("gated_lookalike_matched"),
                   "n_seeds": len(seeds), "ari_sources": "+".join(sorted(src.get((ds, fm), set())))}
            if seeds:
                u = [v["ungated"] for v in seeds.values()]
                g = [v["gated"] for v in seeds.values()]
                d = C.direction([v["gated"] - v["ungated"] for v in seeds.values()])
                row.update(median_ari_ungated=statistics.median(u), median_ari_gated=statistics.median(g),
                           median_delta=d["median"], n_up=d["n_up"], n_down=d["n_down"], direction=d["label"],
                           wilcoxon_p=d["wilcoxon_p"], floor=min(statistics.median(u), statistics.median(g)) < FLOOR)
            out.append(row)
    return out


def f(x, nd=3) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def markdown(cc: list[dict], rr: list[dict]) -> str:
    get = {(c["dataset"], c["fm"]): c for c in cc}
    hdr = "| Dataset | " + " | ".join(FMS) + " |\n|---|" + "---|" * len(FMS)
    L = ["# FM comparison — 12 slides × 5 encoders (complete)", "",
         f"> Generated {datetime.now(timezone.utc).isoformat()} by `scripts/fm_complete/build_fm_comparison_table.py`.",
         "> The pre-registered decision is not recomputed here; it stays as in "
         "`lab/static/tables/prereg/prereg_results_v2.md` (Harm met, Benefit not met → Application Note).",
         f"> {len(rr)} training runs. Phikon-v2 for Liver, Xenium, Hartman (seeds 2, 3, 7, 99, 2023) and DLPFC151673 "
         "come from the gap fill (`logs/fm_complete/`), which reproduced the Step 1 seed-0 ARI exactly; "
         "these runs were not part of the pre-registered package.", "",
         "Direction: ≥ 80% of seeds with the same sign (↑ up, ↓ down, · no stable change). "
         f"Floor (median ARI < {FLOOR}) marked with *.", "",
         "## 1. Switch decision", "", hdr]
    for ds in ORDER:
        L.append(f"| {ds} | " + " | ".join(f(get[(ds, fm)]["switch"]) for fm in FMS) + " |")
    L += ["", "## 2. Median Δ ARI (gated − ungated), direction", "", hdr]
    for ds in ORDER:
        cells = []
        for fm in FMS:
            c = get[(ds, fm)]
            cells.append("—" if not c["n_seeds"] else
                         f"{c['median_delta']:+.3f} {ARROW[c['direction']]}{'*' if c['floor'] else ''}")
        L.append(f"| {ds} | " + " | ".join(cells) + " |")
    L += ["", "## 3. Lookalike edges, ungated → gated (fixed τ_fm = 0.80)", "", hdr]
    for ds in ORDER:
        L.append(f"| {ds} | " + " | ".join(f"{f(get[(ds, fm)]['lookalike_ungated'])} → {f(get[(ds, fm)]['lookalike_gated'])}"
                                         for fm in FMS) + " |")
    L += ["", "## 4. All combinations", "",
          "| Dataset | FM | Switch | n_collide | nn_mult | gene_cos | Lookalike u→g | Remote u→g | "
          "Lookalike u→g (matched τ) | Seeds | ARI ungated | ARI gated | Δ ARI | ↑/↓ | Direction | Floor |",
          "|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|---|---|"]
    for c in cc:
        L.append(f"| {c['dataset']} | {c['fm']} | {f(c['switch'])} | {f(c['n_collide'])} | {f(c['nn_mult_median'], 2)} | "
                 f"{f(c['gene_cos_median'])} | {f(c['lookalike_ungated'])}→{f(c['lookalike_gated'])} | "
                 f"{f(c['remote_ungated'])}→{f(c['remote_gated'])} | "
                 f"{f(c['lookalike_matched_ungated'])}→{f(c['lookalike_matched_gated'])} | {c['n_seeds']} | "
                 f"{f(c.get('median_ari_ungated'))} | {f(c.get('median_ari_gated'))} | {f(c.get('median_delta'))} | "
                 f"{c.get('n_up', '—')}/{c.get('n_down', '—')} | {c.get('direction', '—')} | {f(c.get('floor'))} |")
    return "\n".join(L) + "\n"


def main() -> None:
    rr = runs()
    cc = combos(rr, audit())
    missing = [(c["dataset"], c["fm"]) for c in cc if c["n_seeds"] < 10 or c["switch"] is None]
    OUT.mkdir(parents=True, exist_ok=True)
    md = C.new_path(OUT / "fm_comparison_12x5.md")
    md.write_text(markdown(cc, rr), encoding="utf-8")
    write_csv(C.new_path(OUT / "fm_comparison_12x5.csv"), cc)
    write_csv(C.new_path(OUT / "fm_comparison_runs.csv"), rr)
    print(f"{len(rr)} runs, {len(cc)} combinations -> {md.relative_to(ROOT)}")
    print("incomplete combinations:", missing or "none")


if __name__ == "__main__":
    main()
