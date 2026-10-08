"""Check the Part 1 / Part 2 hooks on the real training data without training anything.

STAMP.train is replaced by a function that raises right after the graph is built, so each call runs the normal
data loading, the Step 1 recipe up to the skip edges, and the hook — then stops. Nothing is written to logs/prereg;
Part 2 selections go to a temporary folder.

  python scripts/prereg/check_hooks_no_training.py --part 1 --only BlockA
  python scripts/prereg/check_hooks_no_training.py --part 2 --only DLPFC151507
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _prereg_common as C  # noqa: E402


class StopBeforeTraining(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["1", "2"], required=True)
    ap.add_argument("--only", required=True)
    args = ap.parse_args()

    r = C.Runner(args.only)
    from stamp.model import STAMP

    def no_train(self, *a, **k):
        raise StopBeforeTraining

    STAMP.train = no_train
    C.install_skip_hook(r.owner)
    if args.part == "1":
        import run_part1_local_endpoint as P1
        C.SKIP_HOOK = P1.hook(args.only)
        for arm in ("ungated", "gated"):
            try:
                r.train(arm, 0)
            except StopBeforeTraining:
                pass
            info = P1.STATE["info"]
            print(f"[check] {args.only} {arm}: {info} affected_primary={int(P1.STATE['affected_primary'].sum())} "
                  f"affected_any={int(P1.STATE['affected_any'].sum())} control={int(P1.STATE['control'].sum())} "
                  f"FM={C.check_redirect(args.only)}", flush=True)
    else:
        import run_part2_injection as P2
        P2.SEL_DIR = Path(tempfile.mkdtemp(prefix="prereg_part2_"))
        C.SKIP_HOOK = P2.hook(args.only)
        for cond, rate in P2.JOBS:
            P2.CURRENT.clear()
            P2.CURRENT.update(cond=cond, rate=rate)
            try:
                r.train("gated", 0)
            except StopBeforeTraining:
                pass
            print(f"[check] {args.only} {cond} rate={rate}: injected={P2.CURRENT.get('n_injected')}", flush=True)
        print(f"[check] selection meta: {P2.SEL['meta']}", flush=True)
        print(f"[check] temp selections: {P2.SEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
