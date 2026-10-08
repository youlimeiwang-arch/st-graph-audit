"""Corrected-coordinate re-clip + Phikon-v2 for slides whose local phikonv2_emb.npy is transposed.

run_local_extension_pipeline._load_visium_coords falls back to tissue_positions columns [4, 5]
(10x: pxl_row_in_fullres = y, pxl_col_in_fullres = x) and uses them as (x, y). Same spot order,
image source, patch size and resize as clip_slide; only the coordinate order is fixed to (col, row).
New files only: <slide>/clip_image_xy/, <slide>/phikonv2_xy_emb.npy, <slide>/phikonv2_xy_emb_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
import run_local_extension_pipeline as pipe  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

# (dataset, slide, truth file used by clip_slide, image source)
JOBS = [
    ("DLPFC", "151507", "truth.txt", "spatial/tissue_full_image.tif"),
    ("DLPFC", "151673", "truth.txt", "spatial/tissue_full_image.tif"),
    ("visium", "Human_Breast_Cancer", "truth.txt", "fullres/V1_Breast_Cancer_Block_A_Section_1_image.tif"),
    ("visium", "CytAssist_FFPE_Human_Breast_Cancer", "truth_pathologist.txt",
     "fullres/CytAssist_FFPE_Human_Breast_Cancer_tissue_image.tif"),
    ("visium", "Wu_CID44971", "truth.txt", "hires"),
    ("visium", "CRC_Parent", "truth.txt", "hires"),
    ("visium", "HPR_Normal_Prostate", "truth.txt", "hires"),
    ("visium", "CRC_Valdeolivas", "truth.txt", "hires"),
    ("visium", "CRC_Valdeolivas_Rep2", "truth.txt", "hires"),
    ("visium", "Kidney_DKD1", "truth.txt", "hires"),
    ("visium", "Liver_H36", "truth.txt", "hires"),
    ("visium", "Liver_Yu_L5", "truth.txt", "hires"),
]
# already extracted at the correct location with the identical full-res protocol
PRECOMPUTED = {"Human_Breast_Cancer": "phikonv2_swap_fullres_emb.npy",
               "CytAssist_FFPE_Human_Breast_Cancer": "phikonv2_swap_fullres_emb.npy"}


def coords_xy(slide_dir: Path, truth: str) -> np.ndarray:
    tp = slide_dir / truth
    data = pipe._load_visium_coords(slide_dir, tp if tp.is_file() else None)
    return np.asarray(data.obsm["spatial"], dtype=np.float64)[:, ::-1].copy()


def read_image(slide_dir: Path, source: str):
    if source == "hires":
        im = cv2.imread(str(slide_dir / "spatial/tissue_hires_image.png"), cv2.IMREAD_COLOR)
        sf = json.loads((slide_dir / "spatial/scalefactors_json.json").read_text())
        scale = float(sf["tissue_hires_scalef"])
        return im, scale, max(8, int(round(pipe.PATCH_FULL * scale)))
    p = slide_dir / source
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        import tifffile

        im = tifffile.imread(str(p))
        im = cv2.cvtColor(im[..., :3] if im.ndim == 3 else np.dstack([im] * 3), cv2.COLOR_RGB2BGR)
    return im, 1.0, pipe.PATCH_FULL


def clip(slide_dir: Path, xy: np.ndarray, source: str) -> Path:
    out = slide_dir / "clip_image_xy"
    if out.exists() and len(list(out.glob("*.png"))) == len(xy):
        return out
    out.mkdir(parents=True, exist_ok=True)
    im, scale, patch = read_image(slide_dir, source)
    h, w = im.shape[:2]
    for i, (x, y) in enumerate(tqdm(xy, desc=f"clip_xy:{slide_dir.name}")):
        cx, cy = x * scale, y * scale
        left, top = int(round(cx - patch / 2)), int(round(cy - patch / 2))
        right, bottom = left + patch, top + patch
        pad_l, pad_t = max(0, -left), max(0, -top)
        pad_r, pad_b = max(0, right - w), max(0, bottom - h)
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)
        crop = im[top:bottom, left:right]
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        cv2.imwrite(str(out / f"{i}.png"), cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on slide name")
    args = ap.parse_args()
    jobs = [j for j in JOBS if args.only.lower() in j[1].lower()]
    model = None
    for dataset, slide, truth, source in jobs:
        sd = ROOT / "Dataset" / dataset / slide
        out = sd / "phikonv2_xy_emb.npy"
        n_old = np.load(sd / "phikonv2_emb.npy", mmap_mode="r").shape[0]
        if out.is_file():
            print(f"[xy] {slide}: exists, skip", flush=True)
            continue
        xy = coords_xy(sd, truth)
        if len(xy) != n_old:
            raise RuntimeError(f"{slide}: {len(xy)} spots != {n_old} rows in phikonv2_emb.npy")
        meta = {"slide": slide, "dataset": dataset, "encoder": "phikonv2", "coords": "(pxl_col, pxl_row) = (x, y)",
                "truth": truth, "image_source": source, "time": datetime.now(timezone.utc).isoformat()}
        if slide in PRECOMPUTED:
            src = sd / PRECOMPUTED[slide]
            emb = np.load(src)
            meta["copied_from"] = src.name
        else:
            cdir = clip(sd, xy, source)
            if model is None:
                import torch

                mod = _load_extractor()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, transform, dim, _ = mod.load_phikon_v2(device)
            from torch.utils.data import DataLoader

            ds = mod.PatchFolder(cdir, transform)
            emb = mod.extract(model, DataLoader(ds, batch_size=32, shuffle=False, num_workers=0), device, dim)
        if emb.shape[0] != n_old:
            raise RuntimeError(f"{slide}: new {emb.shape[0]} != old {n_old}")
        np.save(out, emb.astype(np.float32))
        meta["shape"] = list(emb.shape)
        (sd / "phikonv2_xy_emb_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[xy] WROTE {out} shape={emb.shape}", flush=True)


def _load_extractor():
    import importlib.util

    spec = importlib.util.spec_from_file_location("extract_fm_emb", ROOT / "scripts" / "common" / "extract_fm_emb.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT
    return mod


if __name__ == "__main__":
    main()
