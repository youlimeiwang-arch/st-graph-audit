"""Reply-10 Phase 2 — confirm the x/y-swap hypothesis for the frozen AutoDL Phikon-v2 embedding.

Colour proxy flagged patches cut at (y, x) instead of (x, y). Re-clip at swapped coordinates
(full-res tif and hires-fallback protocol), extract real Phikon-v2, compare with AutoDL fm_cos.
New files only: <slide>/clip_image_swap_{fullres,hires}/, <slide>/phikonv2_swap_{fullres,hires}_emb.npy,
lab/static/tables/reply10_phase2_swap_reextract.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "phase2_coord_diag"))
import run_phase2_fullres_reextract as fr  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

OUT = ROOT / "lab/static/tables/reply10_phase2_swap_reextract.json"


def clip(im: np.ndarray, xy: np.ndarray, scale: float, patch: int, out: Path) -> None:
    if out.exists() and len(list(out.glob("*.png"))) == len(xy):
        return
    out.mkdir(parents=True, exist_ok=True)
    h, w = im.shape[:2]
    for i, (x, y) in enumerate(tqdm(xy, desc=out.name)):
        cx, cy = x * scale, y * scale
        l, t = int(round(cx - patch / 2)), int(round(cy - patch / 2))
        r, b = l + patch, t + patch
        pl, pt, pr, pb = max(0, -l), max(0, -t), max(0, r - w), max(0, b - h)
        c = im[max(0, t):min(h, b), max(0, l):min(w, r)]
        if c.size == 0:
            c = np.full((patch, patch, 3), 255, np.uint8)
        elif pl or pt or pr or pb:
            c = cv2.copyMakeBorder(c, pt, pb, pl, pr, cv2.BORDER_REFLECT)
        cv2.imwrite(str(out / f"{i}.png"), cv2.resize(c, (512, 512), interpolation=cv2.INTER_LINEAR))


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    enc = fr._load("autodl_enc", ROOT / "scripts/phase2_coord_diag/run_phase2_autodl_encoder_probe.py")
    pipe = fr._load("run_local_extension_pipeline", ROOT / "scripts/audit/run_local_extension_pipeline.py")
    mod = fr._load("extract_fm_emb", ROOT / "scripts/common/extract_fm_emb.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, dim, _ = mod.load_phikon_v2(device)
    out = {}
    for sid in ("BlockA", "CytAssist_Patho"):
        s = fr.SLIDES[sid]
        sd = ROOT / "Dataset/visium" / s["dir"]
        xy = np.asarray(pipe._load_visium_coords(sd, sd / s["truth"]).obsm["spatial"], dtype=np.float64)
        sw = xy[:, ::-1].copy()
        hs = float(json.loads((sd / "spatial/scalefactors_json.json").read_text())["tissue_hires_scalef"])
        e = pd.read_csv(s["edges"])
        res = {}
        for tag in ("fullres", "hires"):
            npy = sd / f"phikonv2_swap_{tag}_emb.npy"
            if not npy.is_file():
                if tag == "fullres":
                    im, scale, patch = fr.read_tif(sd / s["tif"]), 1.0, fr.PATCH_FULL
                else:
                    im = cv2.imread(str(sd / "spatial/tissue_hires_image.png"), cv2.IMREAD_COLOR)
                    scale, patch = hs, max(8, int(round(fr.PATCH_FULL * hs)))
                cdir = sd / f"clip_image_swap_{tag}"
                clip(im, sw, scale, patch, cdir)
                del im
                ds = mod.PatchFolder(cdir, transform)
                emb = mod.extract(model, DataLoader(ds, batch_size=32, shuffle=False, num_workers=0), device, dim)
                np.save(npy, emb.astype(np.float32))
            r = enc.score(np.load(npy).astype(np.float64), e)
            z = enc.fmz(np.load(npy).astype(np.float64))
            c = np.sum(z[e.i] * z[e.j], 1)
            r["max_abs_diff"] = float(np.abs(c - e["fm_cos"]).max())
            r["median_abs_diff"] = float(np.median(np.abs(c - e["fm_cos"])))
            res[f"swap (y, x), {tag}"] = r
            print(sid, tag, {k: round(v, 4) for k, v in r.items()}, flush=True)
        out[sid] = res
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
