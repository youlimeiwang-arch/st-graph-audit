"""Exploratory gene-kNN-only graph baseline, independent of audit and switch decisions.

For each spot, select its highest-expression-similarity mutual 10-NN among spatial
non-neighbours, then symmetrize the selected pairs. No FM ranking, audit score,
annotation label, or switch decision is used to construct this skip-edge set.
The existing model recipe and matched ten-seed list are reused unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Keep Numba's import-time cache probe on a fast, local workspace path.  The
# user's default Windows temporary directory can stall inside tempfile.isdir
# before Scanpy or the training code starts.
_RUNTIME_TMP = ROOT / "tmp" / "runtime_tmp"
_NUMBA_CACHE = ROOT / "tmp" / "numba_cache"
_RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
_NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(_RUNTIME_TMP)
os.environ["TMP"] = str(_RUNTIME_TMP)
os.environ["TMPDIR"] = str(_RUNTIME_TMP)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE))
sys.path.insert(0, str(ROOT / "scripts" / "prereg"))
import _prereg_common as C  # noqa: E402

DATASETS = [
    "BlockA", "CytAssist_Patho", "Wu", "Kidney_DKD1", "HPR_Prostate",
    "CRC_Valdeolivas", "CRC_Valdeolivas_Rep2", "Liver_Yu_L5",
    "Xenium_Rep1_IDC", "Hartman_BCSA_A1", "DLPFC151507", "DLPFC151673",
]
GENE_K = 10
LOG = C.ROOT / "logs" / "gene_knn_only_baseline.jsonl"
STATE: dict = {}


def build_gene_only_graph(data, key: str, _orig, _kw) -> None:
    """Add expression-only skip edges to the existing spatial graph."""
    from stamp.edges import _fm_matrix

    n = data.n_obs
    graph = np.asarray(data.obsm["graph_neigh"], dtype=np.float64).copy()
    probs = np.asarray(data.obsm["edge_probabilities"], dtype=np.float64).copy()
    gene = _fm_matrix(data, "feat", pca_dim=min(32, data.obsm["feat"].shape[1]))
    gene /= np.linalg.norm(gene, axis=1, keepdims=True) + 1e-8
    sim = gene @ gene.T
    np.fill_diagonal(sim, -np.inf)
    sim[graph > 0] = -np.inf

    k = min(GENE_K, n - 1)
    nn = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    directed = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    directed[rows, nn] = True
    mutual = directed & directed.T

    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        candidates = np.flatnonzero(mutual[i])
        if candidates.size:
            j = int(candidates[np.argmax(sim[i, candidates])])
            pairs.add((min(i, j), max(i, j)))

    for i, j in pairs:
        graph[i, j] = graph[j, i] = 1.0
        probs[i, j] = probs[j, i] = 0.05
    data.obsm["graph_neigh"] = graph
    data.obsm["edge_probabilities"] = probs

    from stamp.graph_audit.core import edge_table

    audit = edge_table(data, sorted(pairs), fm_key=key)
    primary = audit["relation"].ne("same") if C.primary_relation(STATE["dataset"]) == "cross" else audit["relation"].eq("idc_dcis")
    primary_edges = audit.loc[primary]
    STATE.update(
        n_skip_pairs=len(pairs),
        n_primary_collide=int(primary.sum()),
        primary_lookalike=int(primary_edges["lookalike"].sum()),
        primary_remote=int(primary_edges["remote"].sum()),
        n_skip_lookalike=int(audit["lookalike"].sum()),
        n_skip_remote=int(audit["remote"].sum()),
    )
    print(
        f"gene-kNN-only mutual={GENE_K} skip_pairs={len(pairs)} "
        f"primary_collide={int(primary.sum())} "
        f"lookalike={int(primary_edges['lookalike'].sum())} "
        f"remote={int(primary_edges['remote'].sum())}",
        flush=True,
    )


def run_dataset(ds: str, seeds: list[int], log: Path) -> None:
    runner = C.Runner(ds)
    C.install_skip_hook(runner.owner)
    C.install_cluster_capture()
    done = C.done_keys(log, ("dataset", "arm", "seed"))
    C.SKIP_HOOK = build_gene_only_graph
    for seed in seeds:
        if (ds, "gene_knn_only", seed) in done:
            print(f"SKIP done {ds} seed={seed}", flush=True)
            continue
        STATE.clear()
        STATE["dataset"] = ds
        C.CAPTURED.clear()
        started = datetime.now(timezone.utc).isoformat()
        try:
            row = runner.train("ungated", seed)
            if not C.CAPTURED or "n_skip_pairs" not in STATE:
                raise RuntimeError("gene-only graph hook or cluster capture did not run")
            row.update(
                ok=True,
                dataset=ds,
                arm="gene_knn_only",
                seed=seed,
                graph_rule="mutual_expression_top10_nonspatial; per-node_best; symmetrized",
                construction_uses_fm=False,
                construction_uses_labels=False,
                construction_uses_audit_or_switch=False,
                graph_metrics={k: v for k, v in STATE.items() if k != "dataset"},
                started=started,
                time=datetime.now(timezone.utc).isoformat(),
            )
            print(f"DONE {ds} seed={seed} ARI={row['ari']:.4f} "
                  f"skip_pairs={STATE['n_skip_pairs']}", flush=True)
        except Exception as exc:
            row = {
                "ok": False, "dataset": ds, "arm": "gene_knn_only", "seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
                "time": datetime.now(timezone.utc).isoformat(),
            }
            print(f"ERR {ds} seed={seed}: {row['error']}", flush=True)
        C.append_jsonl(log, row)
        C.free_gpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="all", choices=["all", *DATASETS])
    parser.add_argument("--seeds", default="", help="override matched seed list for smoke tests")
    parser.add_argument("--log", default="", help="write to a separate JSONL path")
    parser.add_argument("--run", action="store_true", help="train; without this flag only list jobs")
    args = parser.parse_args()
    datasets = DATASETS if args.only == "all" else [args.only]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed] if args.seeds else C.SEEDS10
    log = C.ROOT / args.log if args.log else LOG
    # Some frozen runs use the corrected XY Phikon-v2 file when a fix2 file
    # is not present (e.g. Block A); use the same redirect resolution as the
    # existing runner rather than requiring a single hard-coded filename.
    missing = [ds for ds in datasets if not (C.slide_dir(ds) / C.expected_fm(ds)).is_file()]
    if missing:
        raise SystemExit(f"missing fixed Phikon-v2 model inputs for: {missing}")
    done = C.done_keys(log, ("dataset", "arm", "seed"))
    todo = [(ds, seed) for ds in datasets for seed in seeds if (ds, "gene_knn_only", seed) not in done]
    print(f"Baseline: mutual expression top-{GENE_K} among spatial non-neighbours; one best edge/node; symmetrized")
    print(f"This is a post hoc exploratory arm, separate from the frozen preregistered analyses. Jobs remaining: {len(todo)}")
    if not args.run:
        for ds in datasets:
            print(f"{ds:22s} input={C.expected_fm(ds)} jobs={sum(1 for d, _ in todo if d == ds)}")
        return
    for ds in datasets:
        run_dataset(ds, seeds, log)


if __name__ == "__main__":
    main()
