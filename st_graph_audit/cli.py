"""Command line: audit one section and write a summary plus per-edge tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLES = """examples:
  # Space Ranger folder + label file (barcode<TAB>label) + FM embedding (.npy)
  python -m st_graph_audit visium data/Human_Breast_Cancer --labels truth.txt --fm phikonv2_emb.npy --out out/blockA

  # AnnData: raw counts in X, coordinates in obsm['spatial']
  python -m st_graph_audit h5ad section.h5ad --fm-key phikonv2 --label-key domain --out out/sec1

  # cortex: every cross-layer edge is primary
  python -m st_graph_audit visium data/151507 --labels truth.txt --fm emb.npy --primary cross --out out/151507

  # custom label pairs instead of the built-in ones
  python -m st_graph_audit h5ad s.h5ad --fm-key uni --label-key zone --collide Periportal:Pericentral --out out/liver

outputs (in --out; existing output files are never overwritten):
  summary.json         switch, harm counts on both graphs, settings
  edges_ungated.csv    one row per edge of the ungated graph (spatial + FM skip edges)
  edges_gated.csv      same for the gated graph
  skip_gated.csv       skip edges kept by the gene gate (spot barcodes)
exit codes: 0 ok, 1 input error, 2 bad arguments
"""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="st_graph_audit",
        description="Audit FM skip edges on one labeled spatial transcriptomics section (no training).",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="source", required=True)
    v = sub.add_parser("visium", help="Space Ranger output folder")
    v.add_argument("folder")
    v.add_argument("--labels", required=True, help="tab-separated barcode<TAB>label, no header")
    v.add_argument("--fm", required=True, help=".npy FM embedding, one row per spot")
    h = sub.add_parser("h5ad", help="AnnData file")
    h.add_argument("path")
    h.add_argument("--fm-key", required=True, help="obsm key of the FM embedding")
    h.add_argument("--label-key", required=True, help="obs column with spot labels")
    h.add_argument("--gene-key", help="obsm key of ready expression features (default: compute from raw counts in X)")
    for s in (v, h):
        s.add_argument("--out", required=True, help="output folder")
        s.add_argument("--primary", choices=["collide", "cross"], default="collide",
                       help="edges that count: fixed collide label pairs (default) or every cross-label edge")
        s.add_argument("--collide", nargs="+", metavar="A:B", help="collide label pairs (default: built-in tissue pairs)")
        s.add_argument("--n-top-genes", type=int, default=3000, help="highly variable genes (default 3000)")
        s.add_argument("--n-neighbors", type=int, default=5, help="spatial kNN (default 5)")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    out = Path(args.out)
    output_paths = [out / name for name in
                    ("summary.json", "edges_ungated.csv", "edges_gated.csv", "skip_gated.csv")]
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        print(f"error: output file(s) already exist: {paths}; choose a new --out", file=sys.stderr)
        return 1

    from .graph import spatial_knn_graph
    from .io import load_h5ad, load_visium
    from .labels import DEFAULT_COLLIDE_PAIRS, parse_pairs
    from .pipeline import audit_section

    try:
        pairs = parse_pairs(args.collide) if args.collide else DEFAULT_COLLIDE_PAIRS
        if args.source == "visium":
            sec = load_visium(args.folder, args.labels, args.fm, n_top_genes=args.n_top_genes)
        else:
            sec = load_h5ad(args.path, fm_key=args.fm_key, label_key=args.label_key, gene_key=args.gene_key,
                            n_top_genes=args.n_top_genes)
    except (OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"{sec.n_spots} labeled spots; building graphs", flush=True)
    graph = spatial_knn_graph(sec.xy, args.n_neighbors)
    res = audit_section(graph, sec.xy, sec.gene, sec.fm, sec.labels, primary=args.primary, collide_pairs=pairs)

    out.mkdir(parents=True, exist_ok=True)
    summary = {"n_spots": sec.n_spots, "n_neighbors": args.n_neighbors, "n_top_genes": args.n_top_genes,
               **res.summary()}
    for name, df in (("edges_ungated", res.edges_ungated), ("edges_gated", res.edges_gated)):
        df = df.copy()
        df.insert(0, "barcode_j", sec.barcodes[df["j"].to_numpy(dtype=int)])
        df.insert(0, "barcode_i", sec.barcodes[df["i"].to_numpy(dtype=int)])
        df.to_csv(out / f"{name}.csv", index=False)
    sk = np.array(res.skip_gated, dtype=int).reshape(-1, 2)
    pd.DataFrame({"barcode_i": sec.barcodes[sk[:, 0]], "barcode_j": sec.barcodes[sk[:, 1]]}).to_csv(
        out / "skip_gated.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    u, g = res.ungated, res.gated
    print(f"switch: {res.switch}")
    print(f"ungated: lookalike {u['lookalike']}, remote {u['remote']} (of {u['n_primary']} {args.primary} edges)")
    print(f"gated:   lookalike {g['lookalike']}, remote {g['remote']} (of {g['n_primary']} {args.primary} edges)")
    print(f"wrote {out}")
    return 0
