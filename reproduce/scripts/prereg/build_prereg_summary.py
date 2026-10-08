"""Results of the pre-registered package against the fixed decision rule (reply12_response_v2_EN.md).

Reads whatever is finished; parts with no rows are shown as "not run". Never overwrites: writes
lab/static/tables/prereg/prereg_results.md (or _v2, _v3, ...) and the same stem .json.

Decision rule: full paper only if
  Harm    = Part 2, on at least one slide: ARI decreases with injection rate (Spearman rho < 0, one-sided p < 0.05)
            AND the injected_gated condition does not decrease (one-sided p >= 0.05)
  Benefit = Part 3 primary test p < 0.05 (fixed tau_fm; the switch itself does not use tau_fm)
Part 1 is supporting evidence only.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _prereg_common as C  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

P1 = C.LOG / "part1_local.jsonl"
P2 = C.LOG / "part2_injection.jsonl"
P3A = C.LOG / "part3_stageA"
P3B = C.LOG / "part3_stageB.jsonl"
STEP1_ARI = C.ROOT / "logs/review_fix/reply10_phase2_multiseed_merged_fix2.jsonl"
STEP1_SWITCH = C.ROOT / "lab/static/tables/review_fix/reply10_phase2_step1_switch_fix2.json"
FLOOR = 0.05
ALPHA = 0.05
N_PERM = 10000
PERM_SEED = 0


def _ok(rows):
    return [r for r in rows if r.get("ok")]


def _w_greater(d):
    return float(wilcoxon(d, alternative="greater").pvalue) if len(d) >= 5 and any(x != 0 for x in d) else None


def part1() -> dict:
    rows = _ok(C.read_jsonl(P1))
    step1 = {(r["dataset"], r["arm"], int(r["seed"])): float(r["ari"]) for r in _ok(C.read_jsonl(STEP1_ARI))
             if "ari" in r} if STEP1_ARI.is_file() else {}
    out = []
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        by = {}
        for r in rows:
            if r["dataset"] == ds:
                by.setdefault(int(r["seed"]), {})[r["arm"]] = r
        seeds = sorted(s for s, v in by.items() if {"ungated", "gated"} <= v.keys())
        res = {"dataset": ds, "group": by[seeds[0]]["gated"]["switch_group"] if seeds else None, "n_seeds": len(seeds)}
        for key in ("acc_affected_primary", "acc_affected_any", "acc_control", "acc_all", "ari"):
            d = [by[s]["gated"][key] - by[s]["ungated"][key] for s in seeds
                 if by[s]["gated"].get(key) is not None and by[s]["ungated"].get(key) is not None]
            res[key] = C.direction(d) if d else None
            if d:
                res[key]["wilcoxon_p_greater"] = _w_greater(d)
        res["n_affected_primary"] = by[seeds[0]]["gated"]["n_affected_primary"] if seeds else None
        diffs = [abs(float(v[a]["ari"]) - step1[(ds, a, s)]) for s, v in by.items() for a in v if (ds, a, s) in step1]
        res["step1_ari_max_abs_diff"] = max(diffs) if diffs else None
        g = res["group"]
        lab = (res["acc_affected_primary"] or {}).get("label")
        res["prediction_met"] = None if lab is None else (lab == "up" if g == "ON" else lab != "up" if g == "OFF" else None)
        out.append(res)
    return {"rows": out}


def part2() -> dict:
    rows = _ok(C.read_jsonl(P2))
    out = []
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        rr = [r for r in rows if r["dataset"] == ds]
        base = {int(r["seed"]): float(r["ari"]) for r in rr if r["cond"] == "base"}
        res = {"dataset": ds}
        for cond in ("injected", "injected_gated", "random"):
            pts = [(0.0, a) for a in base.values()] + [(float(r["rate"]), float(r["ari"])) for r in rr if r["cond"] == cond]
            if len({x for x, _ in pts}) >= 2:
                rho, p = spearmanr([x for x, _ in pts], [y for _, y in pts], alternative="less")
                res[cond] = {"rho": float(rho), "p_less": float(p), "n": len(pts),
                             "median_by_rate": {x: statistics.median(y for xx, y in pts if xx == x)
                                                for x in sorted({x for x, _ in pts})}}
            else:
                res[cond] = None
        inj = {(float(r["rate"]), int(r["seed"])): float(r["ari"]) for r in rr if r["cond"] == "injected"}
        rnd = {(float(r["rate"]), int(r["seed"])): float(r["ari"]) for r in rr if r["cond"] == "random"}
        d = [rnd[k] - inj[k] for k in inj if k in rnd]
        res["specificity"] = {"n": len(d), "median_random_minus_injected": statistics.median(d) if d else None,
                              "wilcoxon_p_greater": _w_greater(d)}
        i, g = res["injected"], res["injected_gated"]
        res["harm"] = bool(i and i["rho"] < 0 and i["p_less"] < ALPHA)
        res["recovery"] = bool(g and g["p_less"] >= ALPHA)
        out.append(res)
    return {"rows": out, "harm_criterion": any(r["harm"] and r["recovery"] for r in out) if out else None}


def _latest(fm: str) -> Path | None:
    c = sorted(P3A.glob(f"{fm}*.json"), key=lambda p: (len(p.stem), p.stem))
    c = [p for p in c if p.stem == fm or p.stem.startswith(f"{fm}_v")]
    return c[-1] if c else None


def part3() -> dict:
    switch: dict[tuple[str, str], str] = {}
    for fm in ("phikonv2", "phikon", "plip", "uni", "conch"):
        p = _latest(fm)
        if p:
            for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
                if r.get("ok"):
                    switch[(r["dataset"], fm)] = r["switch"]
    if not any(k[1] == "phikonv2" for k in switch) and STEP1_SWITCH.is_file():
        for r in json.loads(STEP1_SWITCH.read_text(encoding="utf-8"))["rows"]:
            if r.get("ok"):
                switch[(r["slide"], "phikonv2")] = r["switch"]
    ari: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for r in _ok(C.read_jsonl(STEP1_ARI)):
        if r["arm"] in ("ungated", "gated"):
            ari.setdefault((r["dataset"], "phikonv2"), {}).setdefault(int(r["seed"]), {})[r["arm"]] = float(r["ari"])
    for r in _ok(C.read_jsonl(P3B)):
        ari.setdefault((r["dataset"], r["fm"]), {}).setdefault(int(r["seed"]), {})[r["arm"]] = float(r["ari"])
    combos = []
    for (ds, fm), by in ari.items():
        pairs = [(v["ungated"], v["gated"]) for v in by.values() if {"ungated", "gated"} <= v.keys()]
        if not pairs:
            continue
        d = C.direction([g - u for u, g in pairs])
        floor = min(statistics.median(u for u, _ in pairs), statistics.median(g for _, g in pairs)) < FLOOR
        combos.append({"dataset": ds, "fm": fm, "switch": switch.get((ds, fm)), "n_seeds": len(pairs),
                       "median_delta": d["median"], "direction": d["label"], "floor": floor})
    usable = [c for c in combos if c["switch"] in ("ON", "OFF") and not c["floor"]]
    on = [c["median_delta"] for c in usable if c["switch"] == "ON"]
    off = [c["median_delta"] for c in usable if c["switch"] == "OFF"]
    test = {"n_on": len(on), "n_off": len(off), "stat": None, "p": None}
    if on and off:
        obs = statistics.median(on) - statistics.median(off)
        rng = np.random.default_rng(PERM_SEED)
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(usable):
            groups.setdefault(c["dataset"], []).append(i)
        lab = np.array([c["switch"] == "ON" for c in usable])
        delta = np.array([c["median_delta"] for c in usable])
        hits = 0
        for _ in range(N_PERM):
            perm = lab.copy()
            for idx in groups.values():
                perm[idx] = rng.permutation(lab[idx])
            if perm.any() and (~perm).any() and np.median(delta[perm]) - np.median(delta[~perm]) >= obs:
                hits += 1
        test.update(stat=obs, p=(1 + hits) / (1 + N_PERM))
    test["n_on_stable_up"] = sum(1 for c in usable if c["switch"] == "ON" and c["direction"] == "up")
    return {"combos": combos, "primary_test": test,
            "benefit_criterion": None if test["p"] is None else test["p"] < ALPHA}


def f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def dcell(d):
    if not d:
        return "—"
    p = "" if d.get("wilcoxon_p") is None else f", p={d['wilcoxon_p']:.3f}"
    return f"{d['label']} ({d['n_up']}/{d['n']}↑, med {d['median']:+.3f}{p})"


def main() -> None:
    r1, r2, r3 = part1(), part2(), part3()
    harm, benefit = r2["harm_criterion"], r3["benefit_criterion"]
    if harm is None or benefit is None:
        decision = "incomplete (a deciding part has no result yet)"
    else:
        decision = "full paper" if harm and benefit else "Application Note"
    L = ["# Pre-registered package — results against the decision rule", "",
         "> Plan and rule: `lab/static/replies/reply12_response_v2_EN.md` (written before any run).",
         f"> Generated {datetime.now(timezone.utc).isoformat()}.", "",
         "## Decision", "",
         f"- **Harm (Part 2, harm + recovery on ≥ 1 slide):** {harm}",
         f"- **Benefit (Part 3 primary test, p < 0.05):** {benefit}",
         f"- **Outcome under the rule:** {decision} (final call: PI)", "",
         "## Part 1 — local endpoint (supporting only)", "",
         "| Dataset | Group | Affected spots | Δ acc affected (gated − ungated) | Δ acc control | Δ ARI | Prediction met | "
         "max |ARI − Step 1| |", "|---|---|---:|---|---|---|---|---:|"]
    for r in r1["rows"]:
        L.append(f"| {r['dataset']} | {r['group']} | {r['n_affected_primary']} | {dcell(r['acc_affected_primary'])} | "
                 f"{dcell(r['acc_control'])} | {dcell(r['ari'])} | {r['prediction_met']} | {f(r['step1_ari_max_abs_diff'], 4)} |")
    if not r1["rows"]:
        L.append("| not run | | | | | | | |")
    L += ["", "## Part 2 — controlled injection", "",
          "| Slide | injected ρ (p<) | injected_gated ρ (p<) | random ρ (p<) | random − injected (p>) | Harm | Recovery |",
          "|---|---|---|---|---|---|---|"]
    for r in r2["rows"]:
        cell = (lambda c: "—" if not c else f"{c['rho']:+.3f} ({c['p_less']:.3f})")
        s = r["specificity"]
        L.append(f"| {r['dataset']} | {cell(r['injected'])} | {cell(r['injected_gated'])} | {cell(r['random'])} | "
                 f"{f(s['median_random_minus_injected'])} ({f(s['wilcoxon_p_greater'])}) | {r['harm']} | {r['recovery']} |")
    if not r2["rows"]:
        L.append("| not run | | | | | | |")
    t = r3["primary_test"]
    L += ["", "## Part 3 — FM comparison", "",
          f"Primary test (switch-ON combos gain more than OFF; within-dataset permutation, {N_PERM} permutations): "
          f"ON n={t['n_on']}, OFF n={t['n_off']}, median(ON) − median(OFF) = {f(t['stat'])}, p = {f(t['p'])}. "
          f"ON combos with a stable 'up': {t['n_on_stable_up']}.", "",
          "| Dataset | FM | Switch | Seeds | median Δ ARI | Direction | Floor |", "|---|---|---|---:|---:|---|---|"]
    for c in sorted(r3["combos"], key=lambda c: (c["dataset"], c["fm"])):
        L.append(f"| {c['dataset']} | {c['fm']} | {c['switch']} | {c['n_seeds']} | {f(c['median_delta'])} | "
                 f"{c['direction']} | {c['floor']} |")
    C.TAB.mkdir(parents=True, exist_ok=True)
    md = C.new_path(C.TAB / "prereg_results.md")
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    md.with_suffix(".json").write_text(json.dumps({"part1": r1, "part2": r2, "part3": r3, "decision": decision},
                                                  indent=2, default=str), encoding="utf-8")
    print("WROTE", md, flush=True)


if __name__ == "__main__":
    main()
