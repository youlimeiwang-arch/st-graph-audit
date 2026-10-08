"""Re-clip + Phikon-v2 for the two image-side bugs found in the code review. New files only:
  <slide>/clip_image_fix2/{i}.png, <slide>/phikonv2_fix2_emb.npy, <slide>/phikonv2_fix2_emb_meta.json
Use them in memory with STAIG_FM_PREFER=phikonv2_fix2_emb.npy (scripts/xyfix/_xy_fm_redirect.py).

Xenium_Rep1_IDC — location fix. pack_xenium_rep1_phase1a applied inv(he_imagealignment) to micron
  centroids; the matrix maps H&E pixels to Xenium morphology pixels (0.2125 um/px), so the bins landed in a
  4.7x-shrunk corner of the H&E. Bin centroids are recomputed exactly as the pack script does (checked
  against the stored positions under the old mapping), mapped with um / 0.2125, and clipped from the full
  H&E ome.tif with the original 504 px patch. Spot order = clip_xenium_rep1_fast (read_visium order,
  labelled bins only). Spatial coordinates on disk are not changed (the kNN graph and remote rule are
  scale-invariant, and old vs new H&E coordinates differ only by a scale about a fixed point).

CRC_Valdeolivas / _Rep2 — patch-size fix. The fixed 504 fullres px patch is 29.5 spot diameters here
  (spot_diameter_fullres 17.1). The patch is set to the median physical field of view of the other Step-1
  slides' 504 px patches, 288 um = 5.23 spot diameters (= DLPFC). Same image (hires), coordinates and spot
  order as run_local_extension_pipeline_xyfix.

Hartman_BCSA_A1 — label fix. phikonv2_emb.npy only covers the 609 spots labelled by the outline-only
  truth.txt; truth_filled.txt labels 3114. Same clip rule as clip_hartman_bcsa_a1_fast (hires, 504 fullres px,
  INTER_AREA), spots = labelled rows of truth_filled.txt in read_visium order.

  python scripts/review_fix/reextract_fix2.py [--only xenium|valdeolivas|hartman] [--stage clip|emb|all]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

KEY = "phikonv2_fix2_emb"
CLIP = "clip_image_fix2"
XENIUM_UM_PER_PX = 0.2125
PATCH_SPOTS = 504 / 96.4
VALDEOLIVAS = ("CRC_Valdeolivas", "CRC_Valdeolivas_Rep2")
XENIUM_EXTRA: dict = {}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clip_patches(im: np.ndarray, xy: np.ndarray, patch: int, out: Path, interp: int = cv2.INTER_LINEAR) -> None:
    out.mkdir(parents=True, exist_ok=True)
    h, w = im.shape[:2]
    for i, (cx, cy) in enumerate(tqdm(xy, desc=out.parent.name)):
        left, top = int(round(cx - patch / 2)), int(round(cy - patch / 2))
        right, bottom = left + patch, top + patch
        pad_l, pad_t = max(0, -left), max(0, -top)
        pad_r, pad_b = max(0, right - w), max(0, bottom - h)
        crop = im[max(0, top):min(h, bottom), max(0, left):min(w, right)]
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        cv2.imwrite(str(out / f"{i}.png"), cv2.resize(crop, (512, 512), interpolation=interp))


def xenium_coords() -> tuple[Path, np.ndarray, dict]:
    import scanpy as sc

    pk = _load("pack_xenium", ROOT / "scripts/data_prep/pack_xenium_rep1_phase1a.py")
    sd = ROOT / "Dataset/visium/Xenium_Rep1_IDC"
    raw = sc.read_10x_h5(pk.DL / "Xenium_FFPE_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5")
    raw.var_names_make_unique()
    cells = pd.read_csv(pk.DL / "Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz")
    cells["cell_id"] = cells["cell_id"].astype(str)
    idc = pd.read_csv(pk.DL / "idc_pathologist.csv")
    idc["cell"] = idc["cell"].astype(str)
    idc["domain"] = idc["domain"].astype(str).map(lambda s: pk.DOMAIN_MAP.get(s, s))
    obs = cells.set_index("cell_id")
    lab = idc.drop_duplicates("cell").set_index("cell")["domain"]
    common = raw.obs_names.astype(str).intersection(obs.index).intersection(lab.index)
    obs, lab = obs.loc[common], lab.loc[common]
    x0, y0 = float(obs["x_centroid"].min()), float(obs["y_centroid"].min())
    xi = ((obs["x_centroid"].to_numpy() - x0) / pk.BIN_UM).astype(np.int32)
    yi = ((obs["y_centroid"].to_numpy() - y0) / pk.BIN_UM).astype(np.int32)
    df = pd.DataFrame({"bin": [f"{a}_{b}" for a, b in zip(xi, yi)], "x": obs["x_centroid"].to_numpy(),
                       "y": obs["y_centroid"].to_numpy()}, index=common)
    sizes = df.groupby("bin").size()
    df = df[df["bin"].isin(sizes[sizes >= pk.MIN_CELLS].index)]
    cats = pd.Categorical(df["bin"]).categories
    um = df.groupby("bin", sort=False)[["x", "y"]].mean().reindex(cats).to_numpy(dtype=np.float64)
    n_cells = df.groupby("bin", sort=False).size().reindex(cats).to_numpy()
    names = [f"{i:016d}-1" for i in range(len(um))]

    Minv = np.linalg.inv(np.loadtxt(pk.DL / "he_imagealignment.csv", delimiter=","))
    to_he = lambda v: (np.c_[v, np.ones(len(v))] @ Minv.T)[:, :2]
    sf = json.loads((sd / "spatial/scalefactors_json.json").read_text(encoding="utf-8"))
    origin = np.asarray(sf["roi_origin_xy"], dtype=np.float64)
    pos = pd.read_csv(sd / "spatial/tissue_positions_list.csv", header=None).set_index(0)
    pos.index = pos.index.astype(str)
    stored = pos.loc[names, [5, 4]].to_numpy(dtype=np.float64)
    old = to_he(um) - origin
    err = float(np.abs(old - stored).max())
    if err > 0.51:
        raise RuntimeError(f"recomputed bins do not reproduce stored positions (max |d| = {err:.2f} px)")
    new = to_he(um / XENIUM_UM_PER_PX)

    ad = sc.read_visium(str(sd), count_file="filtered_feature_bc_matrix.h5", load_images=False)
    truth = pd.read_csv(sd / "truth.txt", sep="\t", header=None, dtype=str, keep_default_na=False)
    labs = truth.set_index(0)[1].reindex(ad.obs_names.astype(str))
    keep = labs.notna() & ~labs.astype(str).isin({"", "nan", "NaN"})
    order = [int(b[:16]) for b in ad.obs_names.astype(str)[keep.to_numpy()]]
    meta = {"check_old_mapping_max_abs_px": err, "um_per_xenium_px": XENIUM_UM_PER_PX, "patch_px_fullres": 504,
            "image_source": str(pk.DL / "he_image.ome.tif"), "coords": "inv(he_imagealignment) @ (um / 0.2125)",
            "new_he_range_x": [float(new[:, 0].min()), float(new[:, 0].max())],
            "new_he_range_y": [float(new[:, 1].min()), float(new[:, 1].max())]}
    XENIUM_EXTRA.update(um=um[order], n_cells=n_cells[order], old_he=old[order] + origin, to_he=to_he)
    return sd, new[order], meta


def read_ome(fp: Path) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(str(fp)) as tf:
        im = tf.pages[0].asarray()
    if im.ndim == 3 and im.shape[0] in (3, 4) and im.shape[0] < im.shape[-1]:
        im = np.moveaxis(im, 0, -1)
    return cv2.cvtColor(np.ascontiguousarray(im[..., :3]), cv2.COLOR_RGB2BGR)


def clip_xenium() -> tuple[Path, dict]:
    sd, xy, meta = xenium_coords()
    n_old = np.load(sd / "phikonv2_emb.npy", mmap_mode="r").shape[0]
    if len(xy) != n_old:
        raise RuntimeError(f"Xenium: {len(xy)} labelled bins != {n_old} rows in phikonv2_emb.npy")
    out = sd / CLIP
    if not (out.exists() and len(list(out.glob("*.png"))) == len(xy)):
        im = read_ome(Path(meta["image_source"]))
        clip_patches(im, xy, 504, out)
        thumb = cv2.resize(im, (im.shape[1] // 16, im.shape[0] // 16), interpolation=cv2.INTER_AREA)
        for x, y in xy[:: max(1, len(xy) // 800)]:
            cv2.circle(thumb, (int(x / 16), int(y / 16)), 2, (0, 0, 255), -1)
        fig = ROOT / "lab/static/figs/review_fix"
        fig.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fig / "xenium_fix2_bins_on_he.png"), thumb)
        del im
    meta["n"] = len(xy)
    return sd, meta


def clip_valdeolivas(slide: str) -> tuple[Path, dict]:
    xf = _load("xyfix_pipe", ROOT / "scripts/xyfix/run_local_extension_pipeline_xyfix.py")
    sd = ROOT / "Dataset/visium" / slide
    xy = xf.coords_xy(sd, "truth.txt")
    n_old = np.load(sd / "phikonv2_emb.npy", mmap_mode="r").shape[0]
    if len(xy) != n_old:
        raise RuntimeError(f"{slide}: {len(xy)} spots != {n_old} rows in phikonv2_emb.npy")
    sf = json.loads((sd / "spatial/scalefactors_json.json").read_text(encoding="utf-8"))
    hs, spot = float(sf["tissue_hires_scalef"]), float(sf["spot_diameter_fullres"])
    patch = max(8, int(round(PATCH_SPOTS * spot * hs)))
    out = sd / CLIP
    if not (out.exists() and len(list(out.glob("*.png"))) == len(xy)):
        im = cv2.imread(str(sd / "spatial/tissue_hires_image.png"), cv2.IMREAD_COLOR)
        clip_patches(im, xy * hs, patch, out)
    meta = {"image_source": "spatial/tissue_hires_image.png", "coords": "(pxl_col, pxl_row) = (x, y), as phikonv2_xy_emb",
            "patch_rule": f"{PATCH_SPOTS:.3f} spot diameters (288 um) x spot_diameter_fullres x tissue_hires_scalef",
            "spot_diameter_fullres": spot, "tissue_hires_scalef": hs, "patch_px_hires": patch,
            "old_patch_px_hires": max(8, int(round(504 * hs))), "n": len(xy)}
    return sd, meta


def clip_hartman() -> tuple[Path, dict]:
    import scanpy as sc

    sd = ROOT / "Dataset/visium/Hartman_BCSA_A1"
    ad = sc.read_visium(str(sd), count_file="filtered_feature_bc_matrix.h5", load_images=False)
    pos = pd.read_csv(sd / "spatial/tissue_positions_list.csv", header=None).set_index(0)
    pos.index = pos.index.astype(str)
    truth = pd.read_csv(sd / "truth_filled.txt", sep="\t", header=None, dtype=str, keep_default_na=False)
    if list(truth[0]) != list(ad.obs_names.astype(str)):
        raise RuntimeError("truth_filled.txt is not in read_visium order")
    keep = ~truth[1].isin({"", "nan", "NaN"}).to_numpy()
    bcs = truth[0][keep].tolist()
    sf = json.loads((sd / "spatial/scalefactors_json.json").read_text(encoding="utf-8"))
    hs = float(sf["tissue_hires_scalef"])
    patch = max(64, int(round(504 * hs)))
    xy = pos.loc[bcs, [5, 4]].to_numpy(dtype=np.float64) * hs
    out = sd / CLIP
    if not (out.exists() and len(list(out.glob("*.png"))) == len(xy)):
        im = cv2.imread(str(sd / "spatial/tissue_hires_image.png"), cv2.IMREAD_COLOR)
        clip_patches(im, xy, patch, out, cv2.INTER_AREA)
    (sd / "clip_barcodes_fix2.txt").write_text("\n".join(bcs) + "\n", encoding="utf-8")
    meta = {"image_source": "spatial/tissue_hires_image.png", "truth": "truth_filled.txt",
            "coords": "(pxl_col, pxl_row) x tissue_hires_scalef, as clip_hartman_bcsa_a1_fast",
            "patch_px_hires": patch, "resize": "INTER_AREA", "barcodes": "clip_barcodes_fix2.txt", "n": len(xy)}
    return sd, meta


def embed(sd: Path, meta: dict, state: dict) -> None:
    out = sd / f"{KEY}.npy"
    if out.is_file():
        print(f"[fix2] {sd.name}: {out.name} exists, skip", flush=True)
        return
    import torch
    from torch.utils.data import DataLoader

    if "model" not in state:
        mod = _load("extract_fm_emb", ROOT / "scripts/common/extract_fm_emb.py")
        mod.ROOT = ROOT
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state.update(mod=mod, dev=dev, **dict(zip(("model", "transform", "dim", "_"), mod.load_phikon_v2(dev))))
    ds = state["mod"].PatchFolder(sd / CLIP, state["transform"])
    e = state["mod"].extract(state["model"], DataLoader(ds, batch_size=32, shuffle=False, num_workers=0),
                             state["dev"], state["dim"])
    if e.shape[0] != meta["n"]:
        raise RuntimeError(f"{sd.name}: {e.shape[0]} embeddings != {meta['n']} spots")
    np.save(out, e.astype(np.float32))
    meta.update(slide=sd.name, encoder="phikonv2", shape=list(e.shape), time=datetime.now(timezone.utc).isoformat())
    (sd / f"{KEY}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[fix2] WROTE {out} shape={e.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "xenium", "valdeolivas", "hartman"], default="all")
    ap.add_argument("--stage", choices=["clip", "emb", "all"], default="all")
    args = ap.parse_args()
    jobs = []
    if args.only in ("all", "xenium"):
        jobs.append(clip_xenium)
    if args.only in ("all", "valdeolivas"):
        jobs += [lambda s=s: clip_valdeolivas(s) for s in VALDEOLIVAS]
    if args.only in ("all", "hartman"):
        jobs.append(clip_hartman)
    state: dict = {}
    for job in jobs:
        sd, meta = job()
        print(f"[fix2] {sd.name}: clipped n={meta['n']} {json.dumps({k: v for k, v in meta.items() if k != 'n'})}", flush=True)
        if args.stage in ("emb", "all"):
            embed(sd, meta, state)


if __name__ == "__main__":
    main()
