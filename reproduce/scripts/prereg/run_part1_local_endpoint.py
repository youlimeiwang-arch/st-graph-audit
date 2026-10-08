"""Part 1 — local endpoint: clustering accuracy on the spots whose edges the gate removes.

Training is the Step 1 recipe (ungated vs gated, 10 seeds); nothing about it changes. Before the skip edges are
added, the hook computes, on a light copy of the training AnnData, the ungated and the gated skip pairs. Removed =
ungated − gated. Affected spots (primary) = endpoints of removed pairs that are lookalike or remote
(stamp.graph_audit.core.edge_table, frozen thresholds) on the slide's primary relation (collide pairs; cross-label
for DLPFC). Secondary: the same without the relation restriction. The hook consumes no random numbers.

Accuracy: per-spot correctness after a Hungarian cluster->label mapping fixed on all spots. Control: the same number
of unaffected spots, drawn once per dataset with numpy seed 0.

Outputs (new, resumable): logs/prereg/part1_local.jsonl, logs/prereg/part1_preds/<ds>_<arm>_<seed>.npz

  python scripts/prereg/run_part1_local_endpoint.py             # dry run: list jobs
  python scripts/prereg/run_part1_local_endpoint.py --run       # all datasets, one process each
  python scripts/prereg/run_part1_local_endpoint.py --run --only BlockA [--seeds 0 --log logs/prereg/smoke.jsonl]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _prereg_common as C  # noqa: E402
import numpy as np  # noqa: E402

PLAN = {
    "BlockA": "ON",
    "CytAssist_Patho": "ON",
    "Wu": "OFF",
    "Kidney_DKD1": "OFF",
    "CRC_Valdeolivas": "OFF",
    "CRC_Valdeolivas_Rep2": "OFF",
    "DLPFC151507": "separate",
}
ARMS = ["ungated", "gated"]
JSONL = C.LOG / "part1_local.jsonl"
PRED_DIR = C.LOG / "part1_preds"
CONTROL_SEED = 0
STATE: dict = {}


def endpoints(pairs, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for i, j in pairs:
        m[i] = m[j] = True
    return m


def hook(ds: str):
    def run(data, key, orig, kw):
        ung = C.skip_pairs(data, key, orig, **C.UNGATED_KW)
        gat = C.skip_pairs(data, key, orig, **C.GATED_KW)
        removed = sorted(ung - gat)
        info = {"n_ungated_pairs": len(ung), "n_gated_pairs": len(gat), "n_removed": len(removed),
                "n_added_by_gate": len(gat - ung)}
        n = data.n_obs
        if removed:
            df = C.edge_flags(data, removed, key)
            flag = (df["lookalike"] == 1) | (df["remote"] == 1)
            rel = (df["relation"] != "same") if C.primary_relation(ds) == "cross" else (df["relation"] == "idc_dcis")
            prim = [(int(i), int(j)) for i, j in df.loc[flag & rel, ["i", "j"]].to_numpy()]
            anyr = [(int(i), int(j)) for i, j in df.loc[flag, ["i", "j"]].to_numpy()]
            info.update(n_removed_flagged_primary=len(prim), n_removed_flagged_any=len(anyr),
                        n_removed_lookalike=int(df["lookalike"].sum()), n_removed_remote=int(df["remote"].sum()))
        else:
            prim, anyr = [], []
            info.update(n_removed_flagged_primary=0, n_removed_flagged_any=0, n_removed_lookalike=0, n_removed_remote=0)
        aff_p, aff_a = endpoints(prim, n), endpoints(anyr, n)
        rng = np.random.default_rng(CONTROL_SEED)
        pool = np.flatnonzero(~aff_p)
        ctrl = np.zeros(n, dtype=bool)
        ctrl[rng.choice(pool, size=min(int(aff_p.sum()), pool.size), replace=False)] = True
        STATE.clear()
        STATE.update(info=info, obs_names=np.asarray(data.obs_names).astype(str), affected_primary=aff_p,
                     affected_any=aff_a, control=ctrl)
        return orig(data, key, **kw)
    return run


def acc(correct: np.ndarray, mask: np.ndarray):
    return float(correct[mask].mean()) if mask.any() else None


def run_dataset(ds: str, seeds: list[int], arms: list[str], log: Path) -> None:
    r = C.Runner(ds)
    C.install_skip_hook(r.owner)
    C.install_cluster_capture()
    C.SKIP_HOOK = hook(ds)
    done = C.done_keys(log, ("dataset", "arm", "seed"))
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for arm in arms:
            if (ds, arm, seed) in done:
                print(f"SKIP done {ds} {arm} {seed}", flush=True)
                continue
            try:
                STATE.clear()
                C.CAPTURED.clear()
                row = r.train(arm, seed)
                if not STATE or not C.CAPTURED:
                    raise RuntimeError("hook or cluster capture did not run")
                if not np.array_equal(STATE["obs_names"], C.CAPTURED["obs_names"]):
                    raise RuntimeError("spot order changed between graph build and clustering")
                truth, pred = C.CAPTURED["truth"], C.CAPTURED["pred"]
                ok = C.hungarian_correct(truth, pred)
                aff_p, aff_a, ctrl = STATE["affected_primary"], STATE["affected_any"], STATE["control"]
                npz = PRED_DIR / f"{ds}_{arm}_{seed}.npz"
                np.savez_compressed(C.new_path(npz), obs_names=STATE["obs_names"], truth=truth, pred=pred,
                                    affected_primary=aff_p, affected_any=aff_a, control=ctrl)
                row.update(ok=True, switch_group=PLAN[ds], **STATE["info"],
                           n_affected_primary=int(aff_p.sum()), n_affected_any=int(aff_a.sum()),
                           n_control=int(ctrl.sum()), acc_all=acc(ok, np.ones_like(ok)),
                           acc_affected_primary=acc(ok, aff_p), acc_affected_any=acc(ok, aff_a),
                           acc_control=acc(ok, ctrl), mapping="hungarian on all spots",
                           time=datetime.now(timezone.utc).isoformat())
                print(f"DONE {ds} {arm} seed={seed} ARI={row['ari']:.4f} acc_aff={row['acc_affected_primary']} "
                      f"n_aff={row['n_affected_primary']}", flush=True)
            except Exception as e:
                row = {"dataset": ds, "arm": arm, "seed": seed, "ok": False, "error": f"{type(e).__name__}: {e}"}
                print(f"ERR {ds} {arm} {seed}: {row['error']}", flush=True)
            C.append_jsonl(log, row)
            C.free_gpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="all", help="dataset id from PLAN, or 'all'")
    ap.add_argument("--seeds", default="", help="override the 10 seeds (smoke tests)")
    ap.add_argument("--log", default="", help="override the JSONL path (smoke tests)")
    C.add_run_flag(ap)
    args = ap.parse_args()
    datasets = list(PLAN) if args.only == "all" else [args.only]
    if any(d not in PLAN for d in datasets):
        raise SystemExit(f"--only must be one of {list(PLAN)}")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else C.SEEDS10
    log = C.ROOT / args.log if args.log else JSONL
    if not args.run:
        done = C.done_keys(log, ("dataset", "arm", "seed"))
        todo = [(d, a, s) for d in datasets for s in seeds for a in ARMS if (d, a, s) not in done]
        for d in datasets:
            print(f"{d:22s} group={PLAN[d]:8s} FM={C.expected_fm(d)}  jobs={sum(1 for x in todo if x[0] == d)}")
        print(f"DRY RUN: {len(todo)} training runs to do -> {log.relative_to(C.ROOT)} (add --run to train)")
        return
    if len(datasets) > 1:
        for d in datasets:
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--only", d]
            if args.seeds:
                cmd += ["--seeds", args.seeds]
            if args.log:
                cmd += ["--log", args.log]
            subprocess.run(cmd, check=False)
        return
    run_dataset(datasets[0], seeds, ARMS, log)


if __name__ == "__main__":
    main()
