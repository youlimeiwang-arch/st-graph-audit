"""Reply-10 Phase 2 Step 1 — multi-seed ARI for the four slides fixed in the code review.

Identical to run_phase2_step1_multiseed.py (recipes, seeds, arms, thresholds) except, in memory only:
  Xenium_Rep1_IDC        Phikon-v2 re-extracted at the corrected H&E location (phikonv2_fix2_emb.npy)
  Hartman_BCSA_A1        labels = filled outlines (truth_filled.txt); Phikon-v2 for those spots (fix2)
  CRC_Valdeolivas / Rep2 Phikon-v2 re-extracted with a 288 um patch (fix2)
Seeds and arms as in the runs they replace (Xenium / Hartman 5 seeds, Valdeolivas 10 seeds; none/ungated/gated).
Log: logs/review_fix/reply10_phase2_multiseed_fix2.jsonl (resumable). Nothing on disk is overwritten.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fix2_env  # noqa: E402

spec = importlib.util.spec_from_file_location("ms_orig", ROOT / "scripts/phase2_reply10/run_phase2_step1_multiseed.py")
ms = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ms)

ms.JSONL = ROOT / "logs/review_fix/reply10_phase2_multiseed_fix2.jsonl"
SEEDS5 = "0,1,42,123,39788"
SEEDS10 = "0,1,2,3,7,42,99,123,2023,39788"
ARMS = "none,ungated,gated"
PLAN = {
    "Xenium_Rep1_IDC": (SEEDS5, ARMS),
    "Hartman_BCSA_A1": (SEEDS5, ARMS),
    "CRC_Valdeolivas": (SEEDS10, ARMS),
    "CRC_Valdeolivas_Rep2": (SEEDS10, ARMS),
}
for ds, name in (("Hartman_BCSA_A1", "truth_filled.txt"),):
    ms.RECIPES[ds]["truth"] = name


class _NoStage:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


ms.StagedTruth = _NoStage
ms.n_truth_classes = (lambda f: (lambda p: f(_fix2_env.truth_path(Path(p).parent))))(ms.n_truth_classes)
_run_one = ms.run_one


def run_one(ds: str, arm: str, seed: int) -> dict:
    row = _run_one(ds, arm, seed)
    red = _fix2_env.redirects()
    slide = ms.RECIPES[ds]["slide"]
    used = {Path(k).name: Path(v).name for k, v in red.items() if Path(k).parent.name == slide}
    if used.get("phikonv2_emb.npy") != _fix2_env.FIX2_EMB:
        raise RuntimeError(f"{ds}: phikonv2_emb.npy not redirected to {_fix2_env.FIX2_EMB} ({used})")
    row.update(fm_redirects=used, truth_file=_fix2_env.truth_path(ROOT / "Dataset/visium" / slide).name,
               source="review-fix rerun (fix2)")
    return row


ms.run_one = run_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    _fix2_env.check_fix2_files()
    ms.JSONL.parent.mkdir(parents=True, exist_ok=True)
    if args.only == "all":
        for ds, (seeds, arms) in PLAN.items():
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--only", ds,
                            "--seeds", seeds, "--arms", arms], check=False)
        return
    ds = args.only
    seeds, arms = PLAN[ds]
    sys.argv = [sys.argv[0], "--only", ds, "--seeds", args.seeds or seeds, "--arms", args.arms or arms]
    ms.main()


if __name__ == "__main__":
    main()
