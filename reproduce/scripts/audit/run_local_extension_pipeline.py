"""Local Windows pipeline: clip → phikonv2 → (optional) graph audit extreme.

Threshold strategy A: fixed Block A freeze values (no retune).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# override autodl hardcodes used by helper scripts
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import cv2
import numpy as np
from tqdm import tqdm

PATCH_FULL = int(3.5 * 144)

# PI-facing priority
JOBS = [
    ("visium", "CRC_Parent"),
    ("visium", "HPR_Normal_Prostate"),
    ("visium", "Wu_CID44971"),
    ("visium", "CytAssist_FFPE_Human_Breast_Cancer"),
    ("visium", "Human_Breast_Cancer"),
    ("DLPFC", "151673"),
]


def _load_visium_coords(slide_dir: Path, truth: Path | None):
    """Lightweight load without staig/ot dependency."""
    import scanpy as sc
    import pandas as pd

    h5 = slide_dir / "filtered_feature_bc_matrix.h5"
    if not h5.is_file():
        # DLPFC sometimes uses different layout
        cands = list(slide_dir.glob("*.h5")) + list(slide_dir.glob("filtered_feature_bc_matrix.h5"))
        if not cands:
            raise FileNotFoundError(slide_dir)
        h5 = cands[0]
    ad = sc.read_visium(str(slide_dir), count_file=h5.name, load_images=False)
    ad.var_names_make_unique()
    if "spatial" not in ad.obsm:
        pos = None
        for name in ("tissue_positions_list.csv", "tissue_positions.csv"):
            fp = slide_dir / "spatial" / name
            if fp.is_file():
                pos = pd.read_csv(fp, header=None)
                break
        if pos is None or pos.shape[1] < 6:
            raise FileNotFoundError(f"no tissue_positions under {slide_dir / 'spatial'}")
        pos = pos.set_index(0)
        pos.index = pos.index.astype(str)
        xy = pos.reindex(ad.obs_names.astype(str))[[4, 5]].to_numpy(dtype=np.float64)
        ad.obsm["spatial"] = xy
    if truth is not None and truth.is_file():
        meta = pd.read_csv(truth, sep="\t", header=None)
        lab = meta.set_index(meta[0].astype(str))[1]
        y = lab.reindex(ad.obs_names.astype(str))
        keep = y.notna() & ~y.astype(str).isin({"", "nan", "NaN"})
        ad = ad[keep].copy()
    return ad


def clip_slide(dataset: str, slide: str, overwrite: bool = False) -> Path:
    slide_dir = ROOT / "Dataset" / dataset / slide
    out = slide_dir / "clip_image"
    truth = slide_dir / "truth.txt"
    if dataset == "visium" and slide == "CytAssist_FFPE_Human_Breast_Cancer":
        if (slide_dir / "truth_pathologist.txt").is_file():
            truth = slide_dir / "truth_pathologist.txt"
    data = _load_visium_coords(slide_dir, truth if truth.is_file() else None)
    coords = np.asarray(data.obsm["spatial"])
    n = coords.shape[0]
    if out.exists() and not overwrite:
        n_png = len(list(out.glob("*.png")))
        if n_png == n:
            print(f"[clip] {slide}: already {n_png} pngs, skip", flush=True)
            return out
    out.mkdir(parents=True, exist_ok=True)
    tif = slide_dir / "spatial" / "tissue_full_image.tif"
    hires = slide_dir / "spatial" / "tissue_hires_image.png"
    scale = 1.0
    source = "full"
    if tif.exists() and tif.stat().st_size > 1_000_000:
        im = cv2.imread(str(tif), cv2.IMREAD_COLOR)
        patch = PATCH_FULL
    elif hires.exists():
        im = cv2.imread(str(hires), cv2.IMREAD_COLOR)
        sf = json.loads((slide_dir / "spatial" / "scalefactors_json.json").read_text())
        scale = float(sf["tissue_hires_scalef"])
        patch = max(8, int(round(PATCH_FULL * scale)))
        source = f"hires scale={scale:.4f} patch={patch}"
    else:
        raise FileNotFoundError(f"{slide}: no histology image")
    if im is None:
        raise RuntimeError(f"{slide}: failed to read histology")
    h, w = im.shape[:2]
    print(f"[clip] {slide}: n={n} image={w}x{h} source={source}", flush=True)
    for i, (x, y) in enumerate(tqdm(coords, desc=f"clip:{slide}")):
        cx, cy = x * scale, y * scale
        left = int(round(cx - patch / 2))
        top = int(round(cy - patch / 2))
        right = left + patch
        bottom = top + patch
        pad_l, pad_t = max(0, -left), max(0, -top)
        pad_r, pad_b = max(0, right - w), max(0, bottom - h)
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)
        crop = im[top:bottom, left:right]
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        resized = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out / f"{i}.png"), resized)
    print(f"[clip] WROTE {out} n={n}", flush=True)
    return out


def extract_phikonv2(dataset: str, slide: str, batch_size: int = 32) -> Path:
    import importlib.util

    spec = importlib.util.spec_from_file_location("extract_fm_emb", ROOT / "scripts" / "common" / "extract_fm_emb.py")
    mod = importlib.util.module_from_spec(spec)
    # monkeypatch ROOT inside module before exec
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT

    slide_dir = ROOT / "Dataset" / dataset / slide
    out = slide_dir / "phikonv2_emb.npy"
    clip_dir = slide_dir / "clip_image"
    if out.is_file():
        emb = np.load(out)
        print(f"[emb] {slide}: exists {out.name} shape={emb.shape}, skip", flush=True)
        return out
    if not clip_dir.exists():
        raise FileNotFoundError(clip_dir)

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[emb] {slide}: extract phikonv2 on {device}", flush=True)
    model, transform, dim, fname = mod.load_phikon_v2(device)
    ds = mod.PatchFolder(clip_dir, transform)
    # Windows: num_workers=0 safer
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    emb = mod.extract(model, loader, device, dim)
    np.save(slide_dir / fname, emb.astype(np.float32))
    meta = {
        "slide": slide,
        "dataset": dataset,
        "encoder": "phikonv2",
        "shape": list(emb.shape),
        "n_patches": len(ds),
        "threshold_strategy": "fixed_BlockA_freeze",
    }
    (slide_dir / "phikonv2_emb_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[emb] WROTE {slide_dir / fname} shape={emb.shape}", flush=True)
    return slide_dir / fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["clip", "emb", "both", "audit"], default="both")
    ap.add_argument("--only", default="", help="substring filter on slide name")
    args = ap.parse_args()
    jobs = JOBS
    if args.only:
        jobs = [j for j in jobs if args.only.lower() in j[1].lower()]
    if not jobs:
        raise SystemExit("no jobs matched --only")

    if args.stage in {"clip", "both"}:
        for ds, slide in jobs:
            clip_slide(ds, slide)
    if args.stage in {"emb", "both"}:
        for ds, slide in jobs:
            extract_phikonv2(ds, slide)
    if args.stage == "audit":
        from stamp.graph_audit_benchmark import run_extreme

        print("[audit] run_extreme …", flush=True)
        run_extreme(export_edges=True)
        print("[audit] done → lab/static/tables/st_graph_audit_extreme.md", flush=True)


if __name__ == "__main__":
    main()
