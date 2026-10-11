"""Part 3 stage B (ARI, ungated vs gated, 10 seeds) for Xenium_Rep1_IDC with Janesick labels, 5 FMs.

Same runner as scripts/prereg/run_part3_stageB_ari.py (_prereg_common.Runner); in memory only:
  - labels: _fix2_env.TRUTH_OVERRIDE["Xenium_Rep1_IDC"] = truth_janesick_full.txt
  - FM:     STAIG_FIX2_FM = <fm>_janesick_emb.npy (rows of the labeled bins; scripts/data_prep/xenium_janesick_train_inputs.py)
Before training, one old-label run (phikon, ungated, seed 0) must reproduce the stored stage B ARI exactly.
Per-run cluster assignments are saved so ARI can be re-scored later without retraining.

  python scripts/review_fix/run_xenium_janesick_stageB.py          # dry run
  python scripts/review_fix/run_xenium_janesick_stageB.py --run    # sanity run, then 100 training runs

Outputs (new, resumable): logs/xenium_relabel/stageB_janesick.jsonl, stageB_sanity.jsonl, preds/, stageB_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLIDE = ROOT / "Dataset/visium/Xenium_Rep1_IDC"
OUT = ROOT / "logs/xenium_relabel"
JSONL = OUT / "stageB_janesick.jsonl"
SANITY = OUT / "stageB_sanity.jsonl"
PREDS = OUT / "preds"
FMS = ("phikonv2", "phikon", "plip", "uni", "conch")
ARMS = ("ungated", "gated")
DS = "Xenium_Rep1_IDC"
TRUTH = "truth_janesick_full.txt"
SANITY_JOB = ("phikon", "ungated", 0)


def stored_sanity_ari() -> float:
    fm, arm, seed = SANITY_JOB
    for line in (ROOT / "logs/prereg/part3_stageB.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if (r.get("dataset"), r.get("fm"), r.get("arm"), r.get("seed"), r.get("ok")) == (DS, fm, arm, seed, True):
            return r["ari"]
    raise SystemExit("stored sanity row not found")


def worker(fm: str, sanity: bool) -> None:
    os.environ["STAIG_FIX2_FM"] = f"{fm}_prereg_emb.npy" if sanity else f"{fm}_janesick_emb.npy"
    sys.path.insert(0, str(ROOT / "scripts" / "prereg"))
    import _prereg_common as C

    if not sanity:
        C._fix2_env.TRUTH_OVERRIDE[DS] = TRUTH
    runner = C.Runner(DS)
    C.install_cluster_capture()
    log = SANITY if sanity else JSONL
    jobs = [SANITY_JOB[1:]] if sanity else [(arm, seed) for seed in C.SEEDS10 for arm in ARMS]
    done = C.done_keys(log, ("fm", "arm", "seed"))
    PREDS.mkdir(parents=True, exist_ok=True)
    for arm, seed in jobs:
        if (fm, arm, seed) in done:
            print(f"SKIP done {fm} {arm} {seed}", flush=True)
            continue
        try:
            row = runner.train(arm, seed)
            if row.get("truth_file") != ("truth.txt" if sanity else TRUTH):
                raise RuntimeError(f"truth file used: {row.get('truth_file')}")
            row.update(ok=True, fm=fm, labels="truth.txt" if sanity else TRUTH)
            if not sanity:
                import pandas as pd
                pd.DataFrame({"barcode": C.CAPTURED["obs_names"], "pred": C.CAPTURED["pred"],
                              "truth": C.CAPTURED["truth"]}).to_csv(PREDS / f"{fm}_{arm}_seed{seed}.csv", index=False)
            print(f"DONE {fm} {arm} seed={seed} n={row['n']} ncl={row['ncl']} ARI={row['ari']:.4f}", flush=True)
        except Exception as e:
            row = {"fm": fm, "dataset": DS, "arm": arm, "seed": seed, "ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"ERR {fm} {arm} {seed}: {row['error']}", flush=True)
        C.append_jsonl(log, row)
        C.free_gpu()


def summarize() -> dict:
    sys.path.insert(0, str(ROOT / "scripts" / "prereg"))
    import _prereg_common as C

    rows = [r for r in C.read_jsonl(JSONL) if r.get("ok")]
    out = {}
    for fm in FMS:
        by = {(r["arm"], r["seed"]): r["ari"] for r in rows if r["fm"] == fm}
        seeds = [s for s in C.SEEDS10 if ("ungated", s) in by and ("gated", s) in by]
        deltas = [by[("gated", s)] - by[("ungated", s)] for s in seeds]
        out[fm] = {"n_seeds": len(seeds),
                   "ari_ungated_mean": sum(by[("ungated", s)] for s in seeds) / len(seeds) if seeds else None,
                   "ari_gated_mean": sum(by[("gated", s)] for s in seeds) / len(seeds) if seeds else None,
                   "delta_mean": sum(deltas) / len(deltas) if deltas else None,
                   "direction": C.direction(deltas) if deltas else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--worker", choices=FMS)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, args.sanity)
        return 0

    needed = [SLIDE / TRUTH] + [SLIDE / f"{fm}_janesick_emb.npy" for fm in FMS]
    missing = [p.name for p in needed if not p.is_file()]
    if missing:
        raise SystemExit(f"missing {missing}; run scripts/data_prep/xenium_janesick_train_inputs.py first")
    if not args.run:
        print(f"DRY RUN: sanity {SANITY_JOB} then {len(FMS) * len(ARMS) * 10} runs -> {JSONL.relative_to(ROOT)}")
        return 0

    sys.path.insert(0, str(ROOT / "scripts" / "maintenance"))
    from notify import send

    t0 = time.monotonic()
    me = [sys.executable, "-u", str(Path(__file__).resolve())]
    subprocess.run(me + ["--worker", SANITY_JOB[0], "--sanity"], check=False)
    got = [json.loads(x) for x in SANITY.read_text(encoding="utf-8").splitlines() if x.strip()] if SANITY.is_file() else []
    got = [r for r in got if r.get("ok")]
    want = stored_sanity_ari()
    if not got or got[-1]["ari"] != want:
        send("Xenium 重训：复现检查没通过", f"旧标签 seed0 应得 ARI {want:.6f}，这次 "
             f"{got[-1]['ari'] if got else '没跑成'}。已停，没开始正式训练。")
        print(f"SANITY FAILED: stored {want}, got {got[-1]['ari'] if got else None}", flush=True)
        return 1
    print(f"SANITY OK: ARI {want} reproduced", flush=True)

    for fm in FMS:
        subprocess.run(me + ["--worker", fm], check=False)

    summary = summarize()
    rows = [json.loads(x) for x in JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
    n_ok, n_err = sum(r.get("ok", False) for r in rows), sum(not r.get("ok", False) for r in rows)
    (OUT / "stageB_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    lines = [f"{fm}: 无门控 {s['ari_ungated_mean']:.3f} 门控 {s['ari_gated_mean']:.3f} {s['direction']['label']}"
             for fm, s in summary.items() if s["n_seeds"]]
    send("Xenium 重训跑完", f"成功 {n_ok} 次，失败 {n_err} 次，用时 {(time.monotonic() - t0) / 3600:.1f} 小时\n"
         + "\n".join(lines))
    print(json.dumps(summary, indent=2))
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
