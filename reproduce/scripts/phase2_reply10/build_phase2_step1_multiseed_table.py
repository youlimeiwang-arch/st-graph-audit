"""Reply-10 Phase 2 Step 1 — aggregate multi-seed three-arm ARI against the unchanged switch.

Direction rule (fixed before results were read): a paired difference has a stable direction
only if at least 80 % of seeds agree in sign (4/5, 8/10); otherwise "no stable change".
Wilcoxon signed-rank p is reported alongside.
Floor: arms with median ARI < 0.05 are not interpretable.

Consistency (PI Reply-10):
  primary   ON  -> gated vs ungated stable up;  OFF -> gated vs ungated NOT stable up
  deployment OFF -> spatial-only (none) not stably worse than gated

Inputs : logs/reply10_phase2_multiseed.jsonl, lab/static/tables/reply10_phase2_step1_switch.json
Outputs: lab/static/tables/reply10_phase2_step1_multiseed.{md,json}
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "lab/static/tables"
JSONL = ROOT / "logs/reply10_phase2_multiseed.jsonl"
SWITCH = TAB / "reply10_phase2_step1_switch.json"

FLOOR = 0.05
AGREE_FRAC = 0.8
ORDER = [
    "BlockA", "CytAssist_Patho", "Wu", "Kidney_DKD1", "HPR_Prostate", "Hartman_BCSA_A1",
    "CRC_Valdeolivas", "CRC_Valdeolivas_Rep2", "Liver_Yu_L5", "Xenium_Rep1_IDC",
]


def direction(deltas: list[float]) -> dict:
    n = len(deltas)
    up = sum(d > 0 for d in deltas)
    down = sum(d < 0 for d in deltas)
    need = math.ceil(AGREE_FRAC * n) if n >= 5 else n
    if n and up >= need:
        lab = "up"
    elif n and down >= need:
        lab = "down"
    else:
        lab = "no stable change"
    p = None
    if n >= 5 and any(d != 0 for d in deltas):
        p = float(wilcoxon(deltas).pvalue)
    return {
        "label": lab, "n": n, "n_up": up, "n_down": down, "need": need, "wilcoxon_p": p,
        "median": statistics.median(deltas) if deltas else None,
        "min": min(deltas) if deltas else None, "max": max(deltas) if deltas else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="39788,42,0,1,123")
    ap.add_argument("--out", default="reply10_phase2_step1_multiseed")
    args = ap.parse_args()
    use_seeds = [int(s) for s in args.seeds.split(",")]
    n_seeds = len(use_seeds)

    runs: dict[tuple[str, str, int], float] = {}
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("ok") and int(r["seed"]) in use_seeds:
            runs[(r["dataset"], r["arm"], int(r["seed"]))] = float(r["ari"])
    sw = {r["slide"]: r for r in json.loads(SWITCH.read_text(encoding="utf-8"))["rows"]}
    need = math.ceil(AGREE_FRAC * n_seeds)

    out_rows = []
    for ds in ORDER:
        seeds = sorted({s for (d, _, s) in runs if d == ds})
        if not seeds:
            continue
        arm = {a: {s: runs[(ds, a, s)] for s in seeds if (ds, a, s) in runs} for a in ("none", "ungated", "gated")}
        common_ug = [s for s in seeds if s in arm["ungated"] and s in arm["gated"]]
        common_ng = [s for s in seeds if s in arm["none"] and s in arm["gated"]]
        d_gu = direction([arm["gated"][s] - arm["ungated"][s] for s in common_ug])
        d_gn = direction([arm["gated"][s] - arm["none"][s] for s in common_ng])
        med = {a: (statistics.median(v.values()) if v else None) for a, v in arm.items()}
        sd = {a: (statistics.pstdev(v.values()) if len(v) > 1 else None) for a, v in arm.items()}
        floor = any(m is not None and m < FLOOR for m in (med["ungated"], med["gated"]))
        s = sw.get(ds, {})
        switch = s.get("switch")
        if floor:
            primary = "uninterpretable (ARI floor)"
        elif d_gu["n"] < n_seeds:
            primary = f"incomplete ({d_gu['n']}/{n_seeds} seeds)"
        elif switch == "ON":
            primary = "consistent" if d_gu["label"] == "up" else "INCONSISTENT"
        else:
            primary = "INCONSISTENT" if d_gu["label"] == "up" else "consistent"
        if switch == "OFF" and not floor and d_gn["n"] >= n_seeds:
            deploy = "OK (none not worse)" if d_gn["label"] != "up" else "gated beats none"
        else:
            deploy = "—"
        out_rows.append({
            "dataset": ds, "name": s.get("name", ds), "switch": switch,
            "components": s.get("components"), "seeds": seeds, "arms": arm,
            "median": med, "sd": sd, "gated_minus_ungated": d_gu, "gated_minus_none": d_gn,
            "floor": floor, "primary_consistency": primary, "deployment_check": deploy,
        })

    TAB.joinpath(f"{args.out}.json").write_text(
        json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(), "seeds": use_seeds,
            "direction_rule": f">= {need}/{n_seeds} seeds same sign", "floor": FLOOR, "rows": out_rows,
        }, indent=2), encoding="utf-8")

    def f(v, d=3):
        return "—" if v is None else f"{v:.{d}f}"

    def dcell(x):
        if not x["n"]:
            return "—"
        p = "" if x["wilcoxon_p"] is None else f", p={x['wilcoxon_p']:.3f}"
        return f"{x['label']} ({x['n_up']}/{x['n']}↑, med {x['median']:+.3f}{p})"

    L = [
        "# Reply-10 Phase 2 Step 1 — multi-seed ARI vs the unchanged switch",
        "",
        "> Switch: unchanged rule (n_collide≥10 ∧ nn_mult≥3 ∧ gene_cos≤0.20), local Phikon-v2, from `reply10_phase2_step1_switch.md`.",
        f"> ARI: {n_seeds} GNN seeds ({', '.join(map(str, use_seeds))}) × 3 arms, each dataset on its original recipe; only seed/arm vary.",
        "> Arms: **none** = spatial graph only (Skip=OFF deployment); **ungated** = FM skip gene_k=0; **gated** = FM skip gene_k=10 mutual.",
        f"> Stable direction = ≥{need}/{n_seeds} seeds same sign (fixed before results); p = Wilcoxon signed-rank (two-sided). "
        f"Floor = median ARI < {FLOOR}.",
        "",
        "| Dataset | Switch | ARI none (median ± sd) | ARI ungated | ARI gated | gated − ungated | gated − none | Consistent? | OFF deployment |",
        "|---------|--------|-----------------------:|------------:|----------:|-----------------|--------------|-------------|----------------|",
    ]
    for r in out_rows:
        m, sd = r["median"], r["sd"]
        L.append(
            f"| {r['name']} | **{r['switch']}** | {f(m['none'])} ± {f(sd['none'])} | {f(m['ungated'])} ± {f(sd['ungated'])} | "
            f"{f(m['gated'])} ± {f(sd['gated'])} | {dcell(r['gated_minus_ungated'])} | {dcell(r['gated_minus_none'])} | "
            f"{r['primary_consistency']} | {r['deployment_check']} |"
        )
    L += ["", "## Per-seed ARI", "", "| Dataset | seed | none | ungated | gated |", "|---------|-----:|-----:|--------:|------:|"]
    for r in out_rows:
        for s in r["seeds"]:
            L.append(
                f"| {r['name']} | {s} | {f(r['arms']['none'].get(s))} | {f(r['arms']['ungated'].get(s))} | {f(r['arms']['gated'].get(s))} |"
            )
    L += [
        "",
        "DLPFC 151673 / 151507: switch undefined (no primary collide pair) — not re-run; existing seed grids in `dlpfc_seed_stability_reply8.json`.",
        "Block A: local re-run with local Phikon-v2; does not reproduce the AutoDL freeze (0.617) and does not replace it.",
        "",
    ]
    TAB.joinpath(f"{args.out}.md").write_text("\n".join(L), encoding="utf-8")
    print("WROTE", TAB / f"{args.out}.md")


if __name__ == "__main__":
    main()
