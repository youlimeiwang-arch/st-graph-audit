"""Check that the st_graph_audit package reproduces the paper's edge-level audit (12 sections x 5 FMs).

Reads only raw inputs (Space Ranger folder, label file, FM embedding .npy) and compares the switch,
collide-edge count, lookalike/remote counts and edge totals with results/part3_stageA/<fm>.json,
which were produced by scripts/prereg/run_part3_stageA_audit.py.

  python reproduce/check_package_against_stageA.py --data-root /path/to/Dataset
  python reproduce/check_package_against_stageA.py --data-root /path/to/Dataset --only BlockA --fm phikonv2

--data-root holds visium/<slide>/ and DLPFC/<slide>/ as listed in SLIDES; each slide folder has the
Space Ranger files, its label file and the FM embedding named in the stored result (fm_redirects).
Exit code 0 if every checked combination matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from st_graph_audit import audit_section, spatial_knn_graph  # noqa: E402
from st_graph_audit.features import gene_features  # noqa: E402
from st_graph_audit.io import read_labels  # noqa: E402

SLIDES = {
    "BlockA": "visium/Human_Breast_Cancer",
    "CytAssist_Patho": "visium/CytAssist_FFPE_Human_Breast_Cancer",
    "Wu": "visium/Wu_CID44971",
    "HPR_Prostate": "visium/HPR_Normal_Prostate",
    "Kidney_DKD1": "visium/Kidney_DKD1",
    "Liver_Yu_L5": "visium/Liver_Yu_L5",
    "Xenium_Rep1_IDC": "visium/Xenium_Rep1_IDC",
    "Hartman_BCSA_A1": "visium/Hartman_BCSA_A1",
    "CRC_Valdeolivas": "visium/CRC_Valdeolivas",
    "CRC_Valdeolivas_Rep2": "visium/CRC_Valdeolivas_Rep2",
    "DLPFC151507": "DLPFC/151507",
    "DLPFC151673": "DLPFC/151673",
}
LABEL_FILE = {"Hartman_BCSA_A1": "truth_filled.txt"}
FMS = ("phikonv2", "phikon", "plip", "uni", "conch")
KEYS = ("switch", "n_collide", "ungated_lookalike", "ungated_remote", "gated_lookalike", "gated_remote",
        "n_edges_ungated", "n_edges_gated")


def load(folder: Path, label_file: str):
    import scanpy as sc

    ad = sc.read_visium(str(folder), count_file="filtered_feature_bc_matrix.h5", load_images=True)
    ad.var_names_make_unique()
    lab = read_labels(folder / label_file).reindex(ad.obs_names)
    kept = (lab.notna() & (lab.astype(str).str.len() > 0)).to_numpy()
    ad = ad[kept].copy()
    return kept, np.asarray(ad.obsm["spatial"]), gene_features(ad, 3000), lab[kept].astype(str).to_numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--results", type=Path, default=HERE / "results" / "part3_stageA")
    ap.add_argument("--only", nargs="+", choices=list(SLIDES), help="subset of sections")
    ap.add_argument("--fm", nargs="+", choices=FMS, default=list(FMS))
    args = ap.parse_args(argv)

    ref = {}
    for fm in args.fm:
        for r in json.loads((args.results / f"{fm}.json").read_text(encoding="utf-8"))["rows"]:
            ref[(r["dataset"], fm)] = r

    n = n_ok = 0
    for ds in args.only or SLIDES:
        folder = args.data_root / SLIDES[ds]
        kept, xy, gene, labels = load(folder, LABEL_FILE.get(ds, "truth.txt"))
        g0 = spatial_knn_graph(xy, 5)
        primary = "cross" if ds.startswith("DLPFC") else "collide"
        for fm in args.fm:
            r = ref[(ds, fm)]
            x = np.load(folder / r["fm_redirects"]["phikonv2_emb.npy"])
            x = x.reshape(x.shape[0], -1)
            x = x if x.shape[0] == kept.sum() else x[kept]
            res = audit_section(g0, xy, gene, x, labels, primary=primary)
            new = {"switch": res.switch, "n_collide": res.ungated["n_primary"],
                   "ungated_lookalike": res.ungated["lookalike"], "ungated_remote": res.ungated["remote"],
                   "gated_lookalike": res.gated["lookalike"], "gated_remote": res.gated["remote"],
                   "n_edges_ungated": res.ungated["n_edges"], "n_edges_gated": res.gated["n_edges"]}
            diff = {k: (r.get(k), new[k]) for k in KEYS if r.get(k) != new[k]}
            n += 1
            n_ok += not diff
            print(f"{ds:22s} {fm:9s} {'OK' if not diff else 'DIFF ' + str(diff)}", flush=True)
    print(f"{n_ok}/{n} match")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
