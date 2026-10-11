"""Relabel the packed Xenium Rep1 bins with Janesick et al.'s own Rep1 cell annotations.

idc_pathologist.csv (Zenodo 15411357) annotates a different Xenium breast dataset (572k cells vs 167,780
in Rep1), so joining it on cell_id gives labels unrelated to the Rep1 cells. This script keeps the bins of
pack_xenium_rep1_phase1a.py unchanged (same cells, same 100 um grid, same order, so expression and FM
embeddings stay valid) and only replaces each bin's label by the majority of Janesick's supervised Rep1
cell types (Cell_Barcode_Type_Matrices.xlsx, sheet "Xenium R1 Fig1-5 (supervised)"), "Unlabeled" excluded.

Self-check: rebuilding the bin labels from idc_pathologist.csv must reproduce truth.txt byte for byte.
Writes new files only (truth_janesick.txt, LABEL_SOURCE_janesick.json); refuses to overwrite.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "Dataset/xenium/_downloads"
SLIDE = ROOT / "Dataset/visium/Xenium_Rep1_IDC"
OUT_TRUTH = SLIDE / "truth_janesick.txt"
OUT_META = SLIDE / "LABEL_SOURCE_janesick.json"

BIN_UM = 100.0
MIN_CELLS = 3

DOMAIN_MAP = {
    "Invasive Tumor": "IDC",
    "DCIS": "DCIS",
    "Stroma": "Stroma",
    "Immune cells": "Immune",
    "Adipose tissue (fat)": "Adipose",
    "Normal ducts": "Normal_duct",
    "Cyst": "Cyst",
    "Blood vessels": "Vessel",
}

JANESICK_MAP = {
    "Invasive_Tumor": "IDC",
    "Prolif_Invasive_Tumor": "IDC",
    "DCIS_1": "DCIS",
    "DCIS_2": "DCIS",
    "Stromal": "Stroma",
    "Perivascular-Like": "Stroma",
    "Endothelial": "Vessel",
    "Myoepi_ACTA2+": "Myoepi",
    "Myoepi_KRT15+": "Myoepi",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "B_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    "T_Cell_&_Tumor_Hybrid": "Hybrid",
    "Stromal_&_T_Cell_Hybrid": "Hybrid",
}


def _majority(labels: pd.Series) -> str:
    vc = labels.value_counts()
    for pref in ("IDC", "DCIS"):
        if pref in vc.index and vc[pref] == vc.iloc[0]:
            return pref
    return str(vc.index[0])


def main() -> None:
    for p in (OUT_TRUTH, OUT_META):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")

    raw = sc.read_10x_h5(DL / "Xenium_FFPE_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5")
    cells = pd.read_csv(DL / "Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz")
    cells["cell_id"] = cells["cell_id"].astype(str)
    idc = pd.read_csv(DL / "idc_pathologist.csv")
    idc["cell"] = idc["cell"].astype(str)
    idc["domain"] = idc["domain"].astype(str).map(lambda s: DOMAIN_MAP.get(s, s))

    obs = cells.set_index("cell_id")
    lab = idc.drop_duplicates("cell").set_index("cell")["domain"]
    common = raw.obs_names.astype(str).intersection(obs.index).intersection(lab.index)
    obs = obs.loc[common]
    lab = lab.loc[common]

    x0, y0 = float(obs["x_centroid"].min()), float(obs["y_centroid"].min())
    xi = ((obs["x_centroid"].to_numpy() - x0) / BIN_UM).astype(np.int32)
    yi = ((obs["y_centroid"].to_numpy() - y0) / BIN_UM).astype(np.int32)
    df = pd.DataFrame(
        {"bin": [f"{a}_{b}" for a, b in zip(xi, yi)], "lab": lab.to_numpy()},
        index=pd.Index(common, name="cell"),
    )
    sizes = df.groupby("bin").size()
    df = df[df["bin"].isin(sizes[sizes >= MIN_CELLS].index)]
    cats = pd.Categorical(df["bin"]).categories
    barcodes = [f"{i:016d}-1" for i in range(len(cats))]

    old = df.groupby("bin", sort=False)["lab"].agg(_majority).reindex(cats)
    rebuilt = "".join(f"{bc}\t{v}\n" for bc, v in zip(barcodes, old.astype(str)))
    if rebuilt.encode("utf-8") != (SLIDE / "truth.txt").read_bytes().replace(b"\r\n", b"\n"):
        raise SystemExit("self-check failed: rebuilt bins do not reproduce truth.txt")
    print(f"self-check OK: {len(cats)} bins reproduce truth.txt", flush=True)

    xl = pd.read_excel(DL / "Cell_Barcode_Type_Matrices.xlsx", sheet_name="Xenium R1 Fig1-5 (supervised)")
    jan = xl.assign(Barcode=xl["Barcode"].astype(str)).set_index("Barcode")["Cluster"].astype(str)
    unknown = sorted(set(jan) - set(JANESICK_MAP) - {"Unlabeled"})
    if unknown:
        raise SystemExit(f"unmapped Janesick types: {unknown}")
    df["jan"] = jan.reindex(df.index).map(JANESICK_MAP)
    voted = df.dropna(subset=["jan"])
    new = voted.groupby("bin", sort=False)["jan"].agg(_majority).reindex(cats)

    lines = [f"{bc}\t{v}" for bc, v in zip(barcodes, new) if isinstance(v, str)]
    OUT_TRUTH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    vc = new.value_counts().to_dict()
    cross = pd.crosstab(old.rename("idc_csv"), new.fillna("NA").rename("janesick"))
    meta = {
        "slide": "Xenium_Rep1_IDC",
        "label_source": "Janesick et al. 2023 Nat Commun, Cell_Barcode_Type_Matrices.xlsx, "
                        "sheet 'Xenium R1 Fig1-5 (supervised)'; bin label = majority mapped cell type, Unlabeled excluded",
        "replaces": "truth.txt (idc_pathologist.csv joined on cell_id; belongs to a different Xenium dataset)",
        "bins_total": int(len(cats)),
        "bins_labeled": int(len(lines)),
        "bins_without_labeled_cells": int(new.isna().sum()),
        "cells_in_bins": int(len(df)),
        "cells_voting": int(len(voted)),
        "label_counts": {str(k): int(v) for k, v in vc.items()},
        "n_IDC": int(vc.get("IDC", 0)),
        "n_DCIS": int(vc.get("DCIS", 0)),
        "janesick_map": JANESICK_MAP,
        "agreement_with_truth_txt": float((old.to_numpy() == new.to_numpy()).mean()),
        "crosstab_truth_txt_vs_janesick": {r: {c: int(cross.loc[r, c]) for c in cross.columns} for r in cross.index},
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: meta[k] for k in ("bins_total", "bins_labeled", "label_counts",
                                            "agreement_with_truth_txt")}, indent=2))
    print(cross.to_string())
    print("WROTE", OUT_TRUTH, OUT_META)


if __name__ == "__main__":
    main()
