"""Fill the Phikon-v2 gaps in the 12-slide x 5-FM comparison. Not part of the pre-registered decision.

The pre-registered summary takes Phikon-v2 from the Step 1 logs, which have 5 seeds for Liver_Yu_L5,
Xenium_Rep1_IDC and Hartman_BCSA_A1 and no DLPFC rows. DLPFC151507 already has 10 Phikon-v2 seeds in
logs/prereg/part1_local.jsonl (same Runner). This runs the rest with the Part 3 stage B Runner and the default
Phikon-v2 file (fix2 where present, else xy). The three slides are at the ARI floor and DLPFC has no switch,
so none of these runs can enter the primary test.

Seed 0 of each Step 1 slide is re-run first and must match the Step 1 log; otherwise that slide stops.

Output (new, resumable): logs/fm_complete/phikonv2_fill.jsonl
  python scripts/fm_complete/run_phikonv2_fill.py          # dry run
  python scripts/fm_complete/run_phikonv2_fill.py --run
"""
from __future__ import annotations

import os

os.environ.pop("STAIG_FIX2_FM", None)

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "prereg"))
sys.path.insert(0, str(ROOT / "scripts" / "maintenance"))
import _prereg_common as C  # noqa: E402

JSONL = ROOT / "logs" / "fm_complete" / "phikonv2_fill.jsonl"
STEP1 = ROOT / "logs" / "review_fix" / "reply10_phase2_multiseed_merged_fix2.jsonl"
ARMS = ["ungated", "gated"]
CHECK_SEED = 0
MISSING = [2, 3, 7, 99, 2023]
PLAN = {
    "Liver_Yu_L5": [CHECK_SEED] + MISSING,
    "Xenium_Rep1_IDC": [CHECK_SEED] + MISSING,
    "Hartman_BCSA_A1": [CHECK_SEED] + MISSING,
    "DLPFC151673": C.SEEDS10,
}
TOL = 1e-6


def step1_ari(ds: str, arm: str, seed: int) -> float | None:
    for r in C.read_jsonl(STEP1):
        if r.get("dataset") == ds and r.get("arm") == arm and r.get("seed") == seed:
            return r["ari"]
    return None


def run_dataset(ds: str) -> None:
    r = C.Runner(ds)
    done = C.done_keys(JSONL, ("dataset", "arm", "seed"))
    for seed in PLAN[ds]:
        for arm in ARMS:
            if (ds, arm, seed) in done:
                print(f"SKIP done {ds} {arm} {seed}", flush=True)
                continue
            try:
                row = r.train(arm, seed)
                row.update(ok=True, fm="phikonv2", time=datetime.now(timezone.utc).isoformat())
                ref = step1_ari(ds, arm, seed) if seed == CHECK_SEED and ds not in C.DLPFC else None
                if ref is not None:
                    row.update(check_ref=ref, check_diff=abs(row["ari"] - ref))
                print(f"DONE phikonv2 {ds} {arm} seed={seed} ARI={row['ari']:.4f}"
                      + (f" step1={ref:.4f}" if ref is not None else ""), flush=True)
            except Exception as e:
                row = {"fm": "phikonv2", "dataset": ds, "arm": arm, "seed": seed, "ok": False,
                       "error": f"{type(e).__name__}: {e}"}
                print(f"ERR {ds} {arm} {seed}: {row['error']}", flush=True)
            C.append_jsonl(JSONL, row)
            C.free_gpu()
            if row.get("check_diff", 0) > TOL:
                raise SystemExit(f"{ds} {arm} seed {seed}: ARI {row['ari']} != Step 1 {row['check_ref']}; stopping")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="all", help="dataset id or 'all'")
    C.add_run_flag(ap)
    args = ap.parse_args()
    datasets = list(PLAN) if args.only == "all" else [args.only]
    if any(d not in PLAN for d in datasets):
        raise SystemExit(f"--only must be one of {list(PLAN)}")
    done = C.done_keys(JSONL, ("dataset", "arm", "seed"))
    todo = [(d, a, s) for d in datasets for s in PLAN[d] for a in ARMS if (d, a, s) not in done]
    if not args.run:
        for d in datasets:
            print(f"{d:22s} {C.expected_fm(d):24s} jobs={sum(1 for x in todo if x[0] == d)}")
        print(f"DRY RUN: {len(todo)} training runs to do -> {JSONL.relative_to(ROOT)} (add --run to train)")
        return
    if len(datasets) == 1:
        run_dataset(datasets[0])
        return
    from notify import send

    for d in datasets:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--only", d], check=False)
    rows = [r for r in C.read_jsonl(JSONL) if r.get("dataset") in PLAN]
    ok = [r for r in rows if r.get("ok")]
    checks = [r for r in ok if "check_diff" in r]
    bad = [r for r in checks if r["check_diff"] > TOL]
    send("Phikon-v2 补全跑完",
         f"成功 {len(ok)} 次，失败 {len(rows) - len(ok)} 次。"
         f"复现核对 {len(checks) - len(bad)}/{len(checks)} 一致。这批不进入事先定好的判定。")


if __name__ == "__main__":
    main()
