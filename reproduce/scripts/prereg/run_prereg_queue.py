"""Queue for the pre-registered package, with a phone push after every job.

Refuses to start unless every script in scripts/prereg matches the SHA-256 recorded in logs/prereg/plan_freeze.json
(written with the plan freeze, before any run). Order: Part 1 (one job per dataset), Part 2 (one per slide), then for
each public FM: patch check (--verify, once), extraction, stage A, stage B; finally the summary table.

  python scripts/prereg/run_prereg_queue.py [--fms phikon,plip] [--skip part1,part2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / "logs/prereg/plan_freeze.json"
sys.path.insert(0, str(ROOT / "scripts/maintenance"))
from notify import send  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_freeze() -> None:
    if not FREEZE.is_file():
        raise SystemExit(f"missing {FREEZE}; freeze the plan first")
    want = json.loads(FREEZE.read_text(encoding="utf-8"))["sha256"]
    bad = [n for n, h in want.items() if not (HERE / n).is_file() or sha(HERE / n) != h]
    if bad:
        raise SystemExit(f"scripts changed after the freeze: {bad}")


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(args: list[str], capture: bool = False) -> tuple[int, float, str]:
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, text=True,
                       capture_output=capture, encoding="utf-8", errors="replace")
    out = (p.stdout or "") if capture else ""
    if capture:
        print(out, flush=True)
    return p.returncode, (time.monotonic() - t0) / 60, out


def med(v):
    return f"{st.median(v):+.3f}" if v else "NA"


def part1() -> None:
    import run_part1_local_endpoint as P1

    for i, ds in enumerate(P1.PLAN, 1):
        rc, mins, _ = run([str(HERE / "run_part1_local_endpoint.py"), "--run", "--only", ds])
        rr = [r for r in rows(P1.JSONL) if r["dataset"] == ds]
        ok = [r for r in rr if r.get("ok")]
        by = {}
        for r in ok:
            by.setdefault(r["seed"], {})[r["arm"]] = r
        d = [v["gated"]["acc_affected_primary"] - v["ungated"]["acc_affected_primary"] for v in by.values()
             if {"ungated", "gated"} <= v.keys() and v["gated"].get("acc_affected_primary") is not None
             and v["ungated"].get("acc_affected_primary") is not None]
        n_aff = ok[0]["n_affected_primary"] if ok else "NA"
        send(f"第1部分 {ds} 跑完（{i}/{len(P1.PLAN)}）",
             f"成功 {len(ok)} 次，失败 {len(rr) - len(ok)} 次，用时 {mins:.0f} 分钟。\n"
             f"受影响 spot {n_aff} 个；门控后准确率变化中位数 {med(d)}（{sum(x > 0 for x in d)}/{len(d)} 个种子上升）")


def part2() -> None:
    import run_part2_injection as P2

    for i, ds in enumerate(P2.SLIDES, 1):
        rc, mins, _ = run([str(HERE / "run_part2_injection.py"), "--run", "--only", ds])
        rr = [r for r in rows(P2.JSONL) if r["dataset"] == ds]
        ok = [r for r in rr if r.get("ok")]
        m = lambda c, x: med([r["ari"] for r in ok if r["cond"] == c and r["rate"] == x]).lstrip("+")
        send(f"第2部分 {ds} 跑完（{i}/{len(P2.SLIDES)}）",
             f"成功 {len(ok)} 次，失败 {len(rr) - len(ok)} 次，用时 {mins:.0f} 分钟。\n"
             f"ARI 中位数：基准 {m('base', 0.0)}；4倍注入 {m('injected', 4.0)}，"
             f"4倍注入后门控 {m('injected_gated', 4.0)}，4倍随机 {m('random', 4.0)}")


def part3(fms: list[str]) -> None:
    import run_part3_stageB_ari as PB

    rc, mins, out = run([str(HERE / "extract_fm_prereg.py"), "--verify"], capture=True)
    n_ok = sum(1 for ln in out.splitlines() if ln.startswith("OK "))
    if rc != 0 or "MISMATCH" in out or n_ok < 12:
        send("第3部分停止：图块核对没通过", f"通过 {n_ok}/12 张切片（rc={rc}）。没有提取新特征，请在电脑上看日志。")
        return
    send("第3部分：图块核对通过", "12 张切片的图块和 Step 1 的 Phikon-v2 特征一致，开始提取新模型特征。")
    for fm in fms:
        rc, mins, out = run([str(HERE / "extract_fm_prereg.py"), "--encoder", fm], capture=True)
        if rc != 0:
            send(f"第3部分 {fm}：提取特征失败", f"rc={rc}，用时 {mins:.0f} 分钟。跳过这个模型，请在电脑上看日志。")
            continue
        rc, mins, out = run([str(HERE / "run_part3_stageA_audit.py"), "--fm", fm], capture=True)
        sw = [ln for ln in out.splitlines() if "switch=" in ln]
        n_on = sum("switch=ON" in ln for ln in sw)
        send(f"第3部分 {fm}：只看边的审计完成",
             f"rc={rc}，{len(sw)} 个数据集有结果，判 ON 的 {n_on} 个。用时 {mins:.0f} 分钟。")
        rc, mins, _ = run([str(HERE / "run_part3_stageB_ari.py"), "--fm", fm, "--run"])
        rr = [r for r in rows(PB.JSONL) if r.get("fm") == fm]
        ok = [r for r in rr if r.get("ok")]
        send(f"第3部分 {fm}：ARI 训练完成", f"成功 {len(ok)} 次，失败 {len(rr) - len(ok)} 次，用时 {mins:.0f} 分钟。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fms", default="phikon,plip")
    ap.add_argument("--skip", default="", help="comma list of part1,part2,part3")
    args = ap.parse_args()
    check_freeze()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    sys.path.insert(0, str(HERE))
    t0 = time.monotonic()
    try:
        send("事先定好的实验：开始", f"顺序：{'第1部分 → ' if 'part1' not in skip else ''}"
             f"{'第2部分 → ' if 'part2' not in skip else ''}{'第3部分（' + args.fms + '）' if 'part3' not in skip else ''}")
        if "part1" not in skip:
            part1()
        if "part2" not in skip:
            part2()
        if "part3" not in skip:
            part3([f.strip() for f in args.fms.split(",") if f.strip()])
        rc, _, out = run([str(HERE / "build_prereg_summary.py")], capture=True)
        send("事先定好的实验：队列完成", f"总用时 {(time.monotonic() - t0) / 60:.0f} 分钟。汇总表已生成（rc={rc}），"
             "UNI/CONCH 等权限批下来再跑。")
    except Exception as e:
        send("事先定好的实验：队列出错停止", f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
