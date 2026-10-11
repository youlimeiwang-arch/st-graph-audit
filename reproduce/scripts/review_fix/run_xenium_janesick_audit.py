"""Edge-level audit (Part 3 stage A rule) of Xenium_Rep1_IDC with Janesick labels, 5 FMs. No training.

Uses the st_graph_audit package (60/60 equivalent to run_part3_stageA_audit.py). For each FM it first
re-audits with the old truth.txt and requires an exact match with the stored stage A row, then audits
with truth_janesick.txt (from scripts/data_prep/relabel_xenium_rep1_janesick.py).
Writes logs/xenium_relabel/stageA_janesick.json (new file; refuses to overwrite).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT.parent
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT / "scripts" / "maintenance"))

from notify import send  # noqa: E402
from st_graph_audit import audit_section, spatial_knn_graph  # noqa: E402
from st_graph_audit.features import gene_features  # noqa: E402
from st_graph_audit.io import read_labels  # noqa: E402

SLIDE = ROOT / "Dataset/visium/Xenium_Rep1_IDC"
STAGE_A = PKG / "reproduce/results/part3_stageA"
OUT = ROOT / "logs/xenium_relabel/stageA_janesick.json"
FMS = ("phikonv2", "phikon", "plip", "uni", "conch")
KEYS = ("switch", "n_collide", "ungated_lookalike", "ungated_remote", "gated_lookalike", "gated_remote",
        "n_edges_ungated", "n_edges_gated")


def load(label_file: str):
    import scanpy as sc

    ad = sc.read_visium(str(SLIDE), count_file="filtered_feature_bc_matrix.h5", load_images=True)
    ad.var_names_make_unique()
    lab = read_labels(SLIDE / label_file).reindex(ad.obs_names)
    kept = (lab.notna() & (lab.astype(str).str.len() > 0)).to_numpy()
    ad = ad[kept].copy()
    return kept, np.asarray(ad.obsm["spatial"]), gene_features(ad, 3000), lab[kept].astype(str).to_numpy()


def row(res) -> dict:
    u, g = res.ungated, res.gated
    return {"switch": res.switch, "n_collide": u["n_primary"],
            "nn_mult_median": u["nn_mult_median"], "gene_cos_median": u["gene_cos_median"],
            "ungated_lookalike": u["lookalike"], "ungated_remote": u["remote"],
            "gated_lookalike": g["lookalike"], "gated_remote": g["remote"],
            "n_edges_ungated": u["n_edges"], "n_edges_gated": g["n_edges"]}


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    t0 = time.monotonic()
    ref = {}
    for fm in FMS:
        for r in json.loads((STAGE_A / f"{fm}.json").read_text(encoding="utf-8"))["rows"]:
            if r["dataset"] == "Xenium_Rep1_IDC":
                ref[fm] = r

    data = {name: load(name) for name in ("truth.txt", "truth_janesick.txt")}
    rows, n_sanity_ok = [], 0
    for fm in FMS:
        x_all = np.load(SLIDE / ref[fm]["fm_redirects"]["phikonv2_emb.npy"])
        x_all = x_all.reshape(x_all.shape[0], -1)
        for name, (kept, xy, gene, labels) in data.items():
            x = x_all if x_all.shape[0] == kept.sum() else x_all[kept]
            res = audit_section(spatial_knn_graph(xy, 5), xy, gene, x, labels, primary="collide")
            r = {"fm": fm, "labels": name, "n_spots": int(kept.sum()), **row(res)}
            if name == "truth.txt":
                diff = {k: (ref[fm].get(k), r[k]) for k in KEYS if ref[fm].get(k) != r[k]}
                r["matches_stored_stageA"] = not diff
                n_sanity_ok += not diff
                if diff:
                    print(f"{fm} sanity DIFF {diff}", flush=True)
            rows.append(r)
            print(json.dumps(r), flush=True)

    sha = {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
           for p in ("scripts/review_fix/run_xenium_janesick_audit.py",
                     "scripts/data_prep/relabel_xenium_rep1_janesick.py")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rule": "part3 stage A (st_graph_audit 0.1.0)", "script_sha256_lf": sha,
                               "sanity_truth_txt_matches": f"{n_sanity_ok}/{len(FMS)}", "rows": rows},
                              indent=2), encoding="utf-8", newline="\n")
    new = [r for r in rows if r["labels"] == "truth_janesick.txt"]
    body = (f"旧标签复现 {n_sanity_ok}/{len(FMS)}；用时 {(time.monotonic() - t0) / 60:.1f} 分钟\n"
            + "\n".join(f"{r['fm']}: 开关{r['switch']} 碰撞边{r['n_collide']} "
                        f"假相似 {r['ungated_lookalike']}→{r['gated_lookalike']}" for r in new))
    send("Xenium 新标签审计跑完", body)
    print("WROTE", OUT)
    return 0 if n_sanity_ok == len(FMS) else 1


if __name__ == "__main__":
    sys.exit(main())
