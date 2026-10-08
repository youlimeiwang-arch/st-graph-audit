"""Part 3 stage B — ARI, ungated vs gated, 10 seeds, per dataset x FM.

Same training as Step 1 (recipes, arms, seed handling); the only change is that phikonv2_emb.npy is redirected in
memory to <fm>_prereg_emb.npy, so the FM drives both the skip edges and (on the fm_pca recipes) the image
drop-edge probabilities, as Phikon-v2 did in Step 1. Every run checks that the redirect hit the expected file.
Phikon-v2 combinations are not re-run: they come from the Step 1 logs.

Output (new, resumable): logs/prereg/part3_stageB.jsonl

  python scripts/prereg/run_part3_stageB_ari.py --fm phikon             # dry run
  python scripts/prereg/run_part3_stageB_ari.py --fm phikon --run       # all datasets, one process each
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--fm", default="")
_fm = _ap.parse_known_args()[0].fm
if _fm:
    os.environ["STAIG_FIX2_FM"] = f"{_fm}_prereg_emb.npy"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _prereg_common as C  # noqa: E402

import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

FMS = ["phikon", "plip", "uni", "conch"]
ORDER = ["BlockA", "CytAssist_Patho", "Wu", "Kidney_DKD1", "HPR_Prostate", "CRC_Valdeolivas", "CRC_Valdeolivas_Rep2",
         "Liver_Yu_L5", "Xenium_Rep1_IDC", "Hartman_BCSA_A1", "DLPFC151507", "DLPFC151673"]
ARMS = ["ungated", "gated"]
JSONL = C.LOG / "part3_stageB.jsonl"


def run_dataset(fm: str, ds: str, seeds: list[int], log: Path) -> None:
    r = C.Runner(ds)
    done = C.done_keys(log, ("fm", "dataset", "arm", "seed"))
    for seed in seeds:
        for arm in ARMS:
            if (fm, ds, arm, seed) in done:
                print(f"SKIP done {fm} {ds} {arm} {seed}", flush=True)
                continue
            try:
                row = r.train(arm, seed)
                row.update(ok=True, fm=fm, time=datetime.now(timezone.utc).isoformat())
                print(f"DONE {fm} {ds} {arm} seed={seed} ARI={row['ari']:.4f}", flush=True)
            except Exception as e:
                row = {"fm": fm, "dataset": ds, "arm": arm, "seed": seed, "ok": False,
                       "error": f"{type(e).__name__}: {e}"}
                print(f"ERR {fm} {ds} {arm} {seed}: {row['error']}", flush=True)
            C.append_jsonl(log, row)
            C.free_gpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fm", required=True, choices=FMS)
    ap.add_argument("--only", default="all", help="dataset id or 'all'")
    ap.add_argument("--seeds", default="", help="override the 10 seeds (smoke tests)")
    ap.add_argument("--log", default="", help="override the JSONL path (smoke tests)")
    C.add_run_flag(ap)
    args = ap.parse_args()
    datasets = ORDER if args.only == "all" else [args.only]
    if any(d not in ORDER for d in datasets):
        raise SystemExit(f"--only must be one of {ORDER}")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else C.SEEDS10
    log = C.ROOT / args.log if args.log else JSONL
    missing = [d for d in datasets if not (C.slide_dir(d) / C.fm_file(args.fm)).is_file()]
    if not args.run:
        done = C.done_keys(log, ("fm", "dataset", "arm", "seed"))
        todo = [(d, a, s) for d in datasets for s in seeds for a in ARMS if (args.fm, d, a, s) not in done]
        for d in datasets:
            print(f"{d:22s} {C.fm_file(args.fm):24s} {'MISSING' if d in missing else 'ok':8s} "
                  f"jobs={sum(1 for x in todo if x[0] == d)}")
        print(f"DRY RUN: {len(todo)} training runs to do -> {log.relative_to(C.ROOT)} (add --run to train)")
        return
    if missing:
        raise SystemExit(f"{C.fm_file(args.fm)} missing for {missing}; run extract_fm_prereg.py first")
    if len(datasets) > 1:
        for d in datasets:
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--fm", args.fm, "--run", "--only", d]
            if args.seeds:
                cmd += ["--seeds", args.seeds]
            if args.log:
                cmd += ["--log", args.log]
            subprocess.run(cmd, check=False)
        return
    run_dataset(args.fm, datasets[0], seeds, log)


if __name__ == "__main__":
    main()
