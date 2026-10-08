"""Pack Janesick Xenium Rep1 as Visium-like slide for Reply-9 Phase 1-A.

Sources:
  - 10x Xenium FFPE Human Breast Cancer Rep1 (expression + cell centroids)
  - Zenodo pathologist domain labels (idc.csv; Invasive Tumor / DCIS / …)
  - HE ome.tif + he_imagealignment.csv (micron → HE pixel)

Cells are aggregated into ~100 µm square bins (Visium-scale) with majority
pathologist domain; primary collide = IDC–DCIS after rename.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import anndata as ad
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "Dataset/xenium/_downloads"
OUT = ROOT / "Dataset/visium/Xenium_Rep1_IDC"

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


def _write_10x_h5(adata: ad.AnnData, path: Path, library_id: str = "Xenium_Rep1_IDC") -> None:
    import h5py

    x = adata.X
    if not sparse.issparse(x):
        x = sparse.csc_matrix(x)
    x = x.T.tocsc()
    genes = adata.var_names.astype(str).to_numpy()
    barcodes = adata.obs_names.astype(str).to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["library_ids"] = np.array([library_id.encode("utf-8")])
        g = f.create_group("matrix")
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        g.create_dataset("data", data=np.asarray(x.data, dtype=np.float32))
        g.create_dataset("indices", data=np.asarray(x.indices, dtype=np.int32))
        g.create_dataset("indptr", data=np.asarray(x.indptr, dtype=np.int32))
        g.create_dataset("shape", data=np.array(x.shape, dtype=np.int32))
        feat = g.create_group("features")
        feat.create_dataset("id", data=np.array(genes, dtype="S"))
        feat.create_dataset("name", data=np.array(genes, dtype="S"))
        feat.create_dataset(
            "feature_type",
            data=np.array(["Gene Expression"] * len(genes), dtype="S"),
        )
        feat.create_dataset("genome", data=np.array(["GRCh38"] * len(genes), dtype="S"))


def _load_affine(path: Path) -> np.ndarray:
    """Return micron→HE affine.

    10x `he_imagealignment.csv` stores HE-pixel → micron; invert for clip coords.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        rows.append([float(x) for x in line.split(",")])
    M = np.asarray(rows, dtype=np.float64)
    if M.shape != (3, 3):
        raise SystemExit(f"bad affine shape {M.shape} in {path}")
    return np.linalg.inv(M)


def _um_to_he(xy_um: np.ndarray, M_um_to_he: np.ndarray) -> np.ndarray:
    ones = np.ones((xy_um.shape[0], 1), dtype=np.float64)
    hom = np.hstack([xy_um, ones])  # N×3
    out = (M_um_to_he @ hom.T).T
    return out[:, :2]


def _majority(labels: pd.Series) -> str:
    vc = labels.value_counts()
    # Prefer IDC/DCIS if tied with stroma-like
    for pref in ("IDC", "DCIS"):
        if pref in vc.index and vc[pref] == vc.iloc[0]:
            return pref
    return str(vc.index[0])


def _attach_he(spat: Path, he_path: Path, xy_he: np.ndarray) -> dict:
    """Crop tissue ROI from OME-TIFF at native res; positions become ROI-local."""
    import tifffile

    with tifffile.TiffFile(str(he_path)) as tf:
        im = tf.pages[0].asarray()
    if im.ndim == 3 and im.shape[0] in (1, 3, 4) and im.shape[0] < im.shape[-1]:
        im = np.moveaxis(im, 0, -1)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    H, W = im.shape[:2]

    pad = 600  # room for 504px patches at edges
    xmin = max(0, int(np.floor(xy_he[:, 0].min())) - pad)
    xmax = min(W, int(np.ceil(xy_he[:, 0].max())) + pad)
    ymin = max(0, int(np.floor(xy_he[:, 1].min())) - pad)
    ymax = min(H, int(np.ceil(xy_he[:, 1].max())) + pad)
    crop = im[ymin:ymax, xmin:xmax].copy()
    # rewrite positions into crop-local fullres pixels
    xy_local = xy_he.copy()
    xy_local[:, 0] -= xmin
    xy_local[:, 1] -= ymin
    print(
        f"HE full {W}x{H}; ROI [{xmin}:{xmax},{ymin}:{ymax}] → {crop.shape[1]}x{crop.shape[0]}",
        flush=True,
    )

    # If ROI still huge, downscale but keep ≥ ~1.0 px/µm equivalent (scale ≥ 0.4)
    max_side = 8000
    scale = min(1.0, max_side / max(crop.shape[0], crop.shape[1]))
    if scale < 1.0:
        hi = cv2.resize(
            crop,
            (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        hi = crop
    low = cv2.resize(hi, (max(1, hi.shape[1] // 4), max(1, hi.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(spat / "tissue_hires_image.png"), hi)
    cv2.imwrite(str(spat / "tissue_lowres_image.png"), low)
    (spat / "he_source.txt").write_text(str(he_path), encoding="utf-8")
    sf = {
        "spot_diameter_fullres": float(BIN_UM),
        "tissue_hires_scalef": float(scale),
        "fiducial_diameter_fullres": float(BIN_UM),
        "tissue_lowres_scalef": float(scale * 0.25),
        "roi_origin_xy": [xmin, ymin],
        "note": "positions rewritten to ROI-local fullres pixels before scale",
    }
    (spat / "scalefactors_json.json").write_text(json.dumps(sf, indent=2), encoding="utf-8")
    # persist local coords for caller to rewrite tissue_positions
    np.save(spat / "_xy_he_local.npy", xy_local)
    return {
        "he_w": W,
        "he_h": H,
        "roi": [xmin, ymin, xmax, ymax],
        "hires_scale": scale,
        "hires_wh": [hi.shape[1], hi.shape[0]],
        "xy_local_range": [xy_local.min(0).tolist(), xy_local.max(0).tolist()],
    }


def main() -> None:
    mat = DL / "Xenium_FFPE_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5"
    cells_fp = DL / "Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz"
    idc_fp = DL / "idc_pathologist.csv"
    aff_fp = DL / "he_imagealignment.csv"
    he_cands = [
        DL / "he_image.ome.tif",
        DL / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif",
        DL / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif",
    ]
    he_path = next((p for p in he_cands if p.is_file() and p.stat().st_size > 500_000_000), None)

    raw = sc.read_10x_h5(mat)
    raw.var_names_make_unique()
    cells = pd.read_csv(cells_fp)
    cells["cell_id"] = cells["cell_id"].astype(str)
    idc = pd.read_csv(idc_fp)
    idc["cell"] = idc["cell"].astype(str)
    idc["domain_raw"] = idc["domain"].astype(str)
    idc["domain"] = idc["domain_raw"].map(lambda s: DOMAIN_MAP.get(s, s))

    obs = cells.set_index("cell_id")
    lab = idc.drop_duplicates("cell").set_index("cell")["domain"]
    common = raw.obs_names.astype(str).intersection(obs.index).intersection(lab.index)
    print(f"cells with expr+label: {len(common)} / {raw.n_obs}", flush=True)
    raw = raw[common].copy()
    obs = obs.loc[common]
    lab = lab.loc[common]

    x0, y0 = float(obs["x_centroid"].min()), float(obs["y_centroid"].min())
    xi = ((obs["x_centroid"].to_numpy() - x0) / BIN_UM).astype(np.int32)
    yi = ((obs["y_centroid"].to_numpy() - y0) / BIN_UM).astype(np.int32)
    bin_id = pd.Index([f"{a}_{b}" for a, b in zip(xi, yi)], name="bin")

    X = raw.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    # Aggregate by bin
    df = pd.DataFrame(
        {
            "bin": bin_id,
            "x": obs["x_centroid"].to_numpy(),
            "y": obs["y_centroid"].to_numpy(),
            "lab": lab.to_numpy(),
        },
        index=raw.obs_names.astype(str),
    )
    sizes = df.groupby("bin").size()
    keep_bins = sizes[sizes >= MIN_CELLS].index
    df = df[df["bin"].isin(keep_bins)]
    print(f"bins >= {MIN_CELLS} cells: {len(keep_bins)}", flush=True)

    # Sum expression via groupby on CSR — build index map
    bin_codes = pd.Categorical(df["bin"])
    n_bins = bin_codes.categories.size
    name_to_i = {n: i for i, n in enumerate(raw.obs_names.astype(str))}
    row_idx = np.fromiter((name_to_i[n] for n in df.index), dtype=np.int64, count=len(df))
    Xk = X[row_idx]
    ind = sparse.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (np.arange(len(df)), bin_codes.codes)),
        shape=(len(df), n_bins),
    )
    Xb = ind.T @ Xk  # bins × genes
    xy = df.groupby("bin", sort=False)[["x", "y"]].mean().reindex(bin_codes.categories)
    labs = df.groupby("bin", sort=False)["lab"].agg(_majority).reindex(bin_codes.categories)

    adata = ad.AnnData(Xb.tocsr())
    adata.obs_names = [f"{i:016d}-1" for i in range(adata.n_obs)]
    adata.var_names = raw.var_names.astype(str)
    adata.obs["ground_truth"] = labs.to_numpy()
    adata.obs["n_cells"] = sizes.reindex(bin_codes.categories).to_numpy()
    adata.obsm["spatial_um"] = xy[["x", "y"]].to_numpy(dtype=np.float64)

    M = _load_affine(aff_fp)
    xy_he = _um_to_he(adata.obsm["spatial_um"], M)
    adata.obsm["spatial"] = xy_he  # HE fullres pixels (col, row) ≈ (x, y)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    spat = OUT / "spatial"
    spat.mkdir()

    truth_lines = [f"{bc}\t{lab}" for bc, lab in zip(adata.obs_names, adata.obs["ground_truth"].astype(str))]
    (OUT / "truth.txt").write_text("\n".join(truth_lines) + "\n", encoding="utf-8")
    (OUT / "truth_pathologist.txt").write_text("\n".join(truth_lines) + "\n", encoding="utf-8")

    # tissue_positions: barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
    # convention used by clip_her2st: spatial = (x,y) = (col, row) → positions col5=x, col4=y
    rows = []
    for i, bc in enumerate(adata.obs_names):
        x, y = xy_he[i]
        rows.append([bc, 1, int(i // 80), int(i % 80), int(round(y)), int(round(x))])
    pd.DataFrame(rows).to_csv(spat / "tissue_positions_list.csv", header=False, index=False)

    he_meta = {}
    if he_path is None:
        print("WARNING: HE ome.tif not ready (>500MB); writing placeholder scalefactors", flush=True)
        (spat / "scalefactors_json.json").write_text(
            json.dumps(
                {
                    "spot_diameter_fullres": BIN_UM,
                    "tissue_hires_scalef": 1.0,
                    "fiducial_diameter_fullres": BIN_UM,
                    "tissue_lowres_scalef": 1.0,
                    "pending_he": True,
                }
            ),
            encoding="utf-8",
        )
    else:
        he_meta = _attach_he(spat, he_path, xy_he)
        local_fp = spat / "_xy_he_local.npy"
        if local_fp.is_file():
            xy_local = np.load(local_fp)
            adata.obsm["spatial"] = xy_local
            rows = []
            for i, bc in enumerate(adata.obs_names):
                x, y = xy_local[i]
                rows.append([bc, 1, int(i // 80), int(i % 80), int(round(y)), int(round(x))])
            pd.DataFrame(rows).to_csv(spat / "tissue_positions_list.csv", header=False, index=False)
            local_fp.unlink()
            print(
                f"rewrote positions to ROI-local; "
                f"x={xy_local[:,0].min():.0f}..{xy_local[:,0].max():.0f} "
                f"y={xy_local[:,1].min():.0f}..{xy_local[:,1].max():.0f}",
                flush=True,
            )

    _write_10x_h5(adata, OUT / "filtered_feature_bc_matrix.h5")

    vc = adata.obs["ground_truth"].value_counts().to_dict()
    meta = {
        "slide": "Xenium_Rep1_IDC",
        "platform": "10x Xenium FFPE Human Breast Cancer Rep1 (Janesick et al.)",
        "bin_um": BIN_UM,
        "min_cells_per_bin": MIN_CELLS,
        "n_bins": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "label_counts": {str(k): int(v) for k, v in vc.items()},
        "primary_collide": "IDC–DCIS",
        "n_IDC": int(vc.get("IDC", 0)),
        "n_DCIS": int(vc.get("DCIS", 0)),
        "label_source": "Zenodo pathologist idc.csv (domain) joined on cell_id",
        "expr_source": "10x cell_feature_matrix.h5",
        "he_source": str(he_path) if he_path else None,
        "he_meta": he_meta,
        "phase1a_note": "Reply-9 Phase 1-A primary redo (Xenium + pathologist IDC/DCIS); HER2ST H1 demoted to side evidence",
        "join_cells_labeled": int(len(common)),
    }
    (OUT / "LABEL_SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "pack_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
