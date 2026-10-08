"""Part 2 — controlled injection: are lookalike-remote edges harmful, and does the gate restore clustering?

Base graph = the Step 1 gated graph (gated arm, unchanged recipe). The hook builds it, then adds injected edges
(drop probability 0.05, as for skip edges). Candidates: pairs with different ground-truth labels, spatial distance
>= 3x the median nearest-neighbour distance, not an edge of the gated graph. Edges are spread over spots: one per
spot first, then two, ... up to MAX_PER_SPOT (greedy_spread), so low rates use one edge per spot.
  injected        candidates by descending FM cosine (Phikon-v2, PCA-16, as in the audit) — realistic lookalikes
  injected_gated  the injected set, keeping only pairs that pass the same gene gate (mutual gene 10-NN, PCA-32,
                  spatial neighbours excluded, as add_fm_skip_edges)
  random          candidates in a random order (numpy seed 0), no FM similarity
Rates 0.5x, 1x, 2x, 4x of the slide's natural remote-edge count (remote pairs among the ungated skip pairs). Sets are
nested across rates (the 1x set is the first half of the 2x set). Rate 0 = the base graph, run once per seed and
shared by all three conditions. Selections are saved to logs/prereg/part2_selections/<ds>.npz.

Outputs (new, resumable): logs/prereg/part2_injection.jsonl

  python scripts/prereg/run_part2_injection.py             # dry run
  python scripts/prereg/run_part2_injection.py --run       # both slides, one process each
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

SLIDES = ["DLPFC151507", "BlockA"]
RATES = [0.5, 1.0, 2.0, 4.0]
CONDS = ["injected", "injected_gated", "random"]
JOBS = [("base", 0.0)] + [(c, r) for r in RATES for c in CONDS]
JSONL = C.LOG / "part2_injection.jsonl"
SEL_DIR = C.LOG / "part2_selections"
SELECT_SEED = 0
TAU_REMOTE = 3.0
MAX_PER_SPOT = 8
CURRENT: dict = {}
SEL: dict = {}


def greedy_spread(ii: np.ndarray, jj: np.ndarray, order: np.ndarray, limit: int) -> list[tuple[int, int]]:
    """Walk the order in passes; pass c admits a pair only if both spots have < c injected edges (c = 1..MAX_PER_SPOT).
    Low rates therefore use at most one edge per spot, and every rate is a prefix of the same list."""
    order = order[: max(limit * 400, 2_000_000)]
    count = np.zeros(int(max(ii.max(), jj.max())) + 1, dtype=np.int32)
    taken = np.zeros(order.size, dtype=bool)
    out: list[tuple[int, int]] = []
    for cap in range(1, MAX_PER_SPOT + 1):
        for pos, t in enumerate(order):
            if taken[pos]:
                continue
            i, j = int(ii[t]), int(jj[t])
            if count[i] >= cap or count[j] >= cap:
                continue
            out.append((i, j))
            taken[pos] = True
            count[i] += 1
            count[j] += 1
            if len(out) >= limit:
                return out
    return out


def gene_gate_ok(data, spatial_graph: np.ndarray) -> np.ndarray:
    """Mutual gene 10-NN among non-spatial pairs — the gate of add_fm_skip_edges(gene_k=10, gene_mutual=True)."""
    from stamp.edges import _fm_matrix

    kw = C.GATED_KW
    g = _fm_matrix(data, kw["gene_key"], pca_dim=min(32, np.asarray(data.obsm[kw["gene_key"]]).shape[1]))
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
    gsim = g @ g.T
    np.fill_diagonal(gsim, -np.inf)
    gsim[spatial_graph > 0] = -np.inf
    k = kw["gene_k"]
    nn = np.argpartition(-gsim, kth=min(k, data.n_obs - 1) - 1, axis=1)[:, :k]
    ok = np.zeros((data.n_obs, data.n_obs), dtype=bool)
    ok[np.repeat(np.arange(data.n_obs), k), nn.reshape(-1)] = True
    return ok & ok.T


def build_selections(ds: str, data, key: str, ung: list[tuple[int, int]], spatial_graph: np.ndarray,
                     base_graph: np.ndarray) -> dict:
    from scipy.spatial.distance import cdist
    from stamp.edges import _fm_matrix
    from stamp.graph_audit.core import median_nn_dist

    df = C.edge_flags(data, ung, key)
    n_remote = int(df["remote"].sum())
    xy = np.asarray(data.obsm["spatial"], dtype=np.float64)
    nn = median_nn_dist(xy)
    lab = data.obs["ground_truth"].astype(str).to_numpy()
    fm = _fm_matrix(data, key, pca_dim=16)
    fm = fm / (np.linalg.norm(fm, axis=1, keepdims=True) + 1e-8)
    cand = (lab[:, None] != lab[None, :]) & (cdist(xy, xy) >= TAU_REMOTE * nn) & (base_graph <= 0)
    ii, jj = np.nonzero(np.triu(cand, 1))
    target = {r: int(round(r * n_remote)) for r in RATES}
    top = max(target.values())
    sims = np.einsum("ij,ij->i", fm[ii], fm[jj])
    look = greedy_spread(ii, jj, np.argsort(-sims, kind="stable"), top)
    rnd = greedy_spread(ii, jj, np.random.default_rng(SELECT_SEED).permutation(ii.size), top)
    if len(look) < top or len(rnd) < top:
        raise RuntimeError(f"{ds}: only {len(look)} lookalike / {len(rnd)} random pairs for the 4x target {top}")
    gate = gene_gate_ok(data, spatial_graph)
    sel = {}
    for r in RATES:
        lk = look[:target[r]]
        sel[("injected", r)] = lk
        sel[("injected_gated", r)] = [p for p in lk if gate[p]]
        sel[("random", r)] = rnd[:target[r]]
    meta = {"n_remote_natural": n_remote, "median_nn": nn, "n_candidates": int(ii.size), "target": target,
            "n_lookalike_available": len(look), "n_random_available": len(rnd),
            "fm_cos_injected_median": {r: float(np.median([fm[i] @ fm[j] for i, j in sel[("injected", r)]]))
                                       if sel[("injected", r)] else None for r in RATES},
            "fm_cos_random_median": {r: float(np.median([fm[i] @ fm[j] for i, j in sel[("random", r)]]))
                                     if sel[("random", r)] else None for r in RATES},
            "kept_by_gate": {r: len(sel[("injected_gated", r)]) for r in RATES}}
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    out = SEL_DIR / f"{ds}.npz"
    if not out.exists():
        np.savez_compressed(out, obs_names=np.asarray(data.obs_names).astype(str),
                            **{f"{c}_{r}": np.asarray(p, dtype=np.int64).reshape(-1, 2) for (c, r), p in sel.items()})
    return {"sel": sel, "meta": meta, "obs_names": np.asarray(data.obs_names).astype(str)}


def hook(ds: str):
    def run(data, key, orig, kw):
        if kw.get("gene_k") != C.GATED_KW["gene_k"]:
            raise RuntimeError("Part 2 runs on the gated arm only")
        spatial = np.array(data.obsm["graph_neigh"], dtype=np.float64, copy=True)
        ung = sorted(C.skip_pairs(data, key, orig, **C.UNGATED_KW)) if "sel" not in SEL else None
        orig(data, key, **kw)
        if "sel" not in SEL:
            SEL.update(build_selections(ds, data, key, ung, spatial, np.asarray(data.obsm["graph_neigh"])))
        elif not np.array_equal(SEL["obs_names"], np.asarray(data.obs_names).astype(str)):
            raise RuntimeError("spot order differs from the saved selection")
        cond, rate = CURRENT["cond"], CURRENT["rate"]
        pairs = [] if cond == "base" else SEL["sel"][(cond, rate)]
        graph = np.array(data.obsm["graph_neigh"], dtype=np.float64, copy=True)
        probs = np.array(data.obsm["edge_probabilities"], dtype=np.float64, copy=True)
        for i, j in pairs:
            if graph[i, j] > 0:
                raise RuntimeError(f"injected pair ({i},{j}) is already an edge")
            graph[i, j] = graph[j, i] = 1.0
            probs[i, j] = probs[j, i] = kw["drop_prob"]
        data.obsm["graph_neigh"] = graph
        data.obsm["edge_probabilities"] = probs
        CURRENT["n_injected"] = len(pairs)
        print(f"[part2] {ds} {cond} rate={rate} injected={len(pairs)}", flush=True)
    return run


def run_dataset(ds: str, seeds: list[int], log: Path) -> None:
    r = C.Runner(ds)
    C.install_skip_hook(r.owner)
    C.SKIP_HOOK = hook(ds)
    done = C.done_keys(log, ("dataset", "cond", "rate", "seed"))
    for seed in seeds:
        for cond, rate in JOBS:
            if (ds, cond, rate, seed) in done:
                print(f"SKIP done {ds} {cond} {rate} {seed}", flush=True)
                continue
            try:
                CURRENT.clear()
                CURRENT.update(cond=cond, rate=rate)
                row = r.train("gated", seed)
                if "n_injected" not in CURRENT:
                    raise RuntimeError("injection hook did not run")
                row.update(ok=True, cond=cond, rate=rate, n_injected=CURRENT["n_injected"],
                           selection=SEL["meta"], time=datetime.now(timezone.utc).isoformat())
                print(f"DONE {ds} {cond} rate={rate} seed={seed} ARI={row['ari']:.4f}", flush=True)
            except Exception as e:
                row = {"dataset": ds, "cond": cond, "rate": rate, "seed": seed, "ok": False,
                       "error": f"{type(e).__name__}: {e}"}
                print(f"ERR {ds} {cond} {rate} {seed}: {row['error']}", flush=True)
            C.append_jsonl(log, row)
            C.free_gpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="all", help="slide id from SLIDES, or 'all'")
    ap.add_argument("--seeds", default="", help="override the 10 seeds (smoke tests)")
    ap.add_argument("--log", default="", help="override the JSONL path (smoke tests)")
    C.add_run_flag(ap)
    args = ap.parse_args()
    datasets = SLIDES if args.only == "all" else [args.only]
    if any(d not in SLIDES for d in datasets):
        raise SystemExit(f"--only must be one of {SLIDES}")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else C.SEEDS10
    log = C.ROOT / args.log if args.log else JSONL
    if not args.run:
        done = C.done_keys(log, ("dataset", "cond", "rate", "seed"))
        todo = [(d, c, r, s) for d in datasets for s in seeds for c, r in JOBS if (d, c, r, s) not in done]
        for d in datasets:
            print(f"{d:14s} FM={C.expected_fm(d)}  jobs={sum(1 for x in todo if x[0] == d)}")
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
    run_dataset(datasets[0], seeds, log)


if __name__ == "__main__":
    main()
