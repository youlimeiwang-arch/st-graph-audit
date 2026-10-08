"""Reply-10 Phase 2 Step 1 — re-extract Phikon-v2 from full-resolution H&E (AutoDL clip protocol).

Local phikonv2_emb.npy for CytAssist / Block A was clipped from tissue_hires_image.png
(~45 px patches upsampled to 512). clip_spots.py on AutoDL preferred the full-res tif
(3.5*144 = 504 px patches). This script reproduces the full-res protocol and writes NEW files:
  <slide>/clip_image_fullres/{i}.png, <slide>/phikonv2_fullres_emb.npy (+ _meta.json)
and compares hires vs full-res embeddings against the frozen AutoDL audit edge lists.

Stages: clip | emb | compare | all
Output (compare): lab/static/tables/reply10_phase2_fullres_reextract.{md,json}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

if "harmonypy" not in sys.modules:
    _hm = types.ModuleType("harmonypy")
    _hm.compute_lisi = lambda emb, *a, **k: [[0.0] for _ in range(emb.shape[0])]
    sys.modules["harmonypy"] = _hm

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

TAB = ROOT / "lab/static/tables"
PATCH_FULL = int(3.5 * 144)
SNAP = Path(
    r"<local copy of the 2026-09-06 STAIG code snapshot>"
) / "lab/static/tables/st_graph_audit_edges"
SLIDES = {
    "CytAssist_Patho": dict(
        dir="CytAssist_FFPE_Human_Breast_Cancer",
        tif="fullres/CytAssist_FFPE_Human_Breast_Cancer_tissue_image.tif",
        truth="truth_pathologist.txt",
        edges=SNAP / "CytAssist_Patho_STAIGplusungated_skip_phikonv2_emb_edges.csv",
        frozen=dict(n=92, nn=14.247, gene=-0.031, on=True, look=73, remote=73),
    ),
    "BlockA": dict(
        dir="Human_Breast_Cancer",
        tif="fullres/V1_Breast_Cancer_Block_A_Section_1_image.tif",
        truth="truth.txt",
        edges=SNAP / "BlockA_STAIGplusungated_skip_phikonv2_emb_edges.csv",
        frozen=dict(n=None, nn=None, gene=None, on=True, look=68, remote=100),
    ),
}
FULLRES_KEY = "phikonv2_fullres_emb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_tif(fp: Path) -> np.ndarray:
    import tifffile

    im = tifffile.imread(str(fp))
    if im.ndim == 3 and im.shape[0] in (3, 4) and im.shape[-1] not in (3, 4):
        im = np.moveaxis(im, 0, -1)
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    im = im[..., :3]
    if im.dtype != np.uint8:
        im = (255.0 * (im.astype(np.float32) / max(1.0, float(im.max())))).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2BGR)


def clip(sid: str) -> None:
    pipe = _load("run_local_extension_pipeline", ROOT / "scripts/audit/run_local_extension_pipeline.py")
    s = SLIDES[sid]
    slide_dir = ROOT / "Dataset/visium" / s["dir"]
    out = slide_dir / "clip_image_fullres"
    data = pipe._load_visium_coords(slide_dir, slide_dir / s["truth"])
    coords = np.asarray(data.obsm["spatial"], dtype=np.float64)
    n = coords.shape[0]
    if out.exists() and len(list(out.glob("*.png"))) == n:
        print(f"[clip] {sid}: {n} pngs exist, skip", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    im = read_tif(slide_dir / s["tif"])
    h, w = im.shape[:2]
    print(f"[clip] {sid}: n={n} image={w}x{h} patch={PATCH_FULL} (full-res)", flush=True)
    for i, (x, y) in enumerate(tqdm(coords, desc=f"clip:{sid}")):
        left, top = int(round(x - PATCH_FULL / 2)), int(round(y - PATCH_FULL / 2))
        right, bottom = left + PATCH_FULL, top + PATCH_FULL
        pad_l, pad_t = max(0, -left), max(0, -top)
        pad_r, pad_b = max(0, right - w), max(0, bottom - h)
        crop = im[max(0, top):min(h, bottom), max(0, left):min(w, right)]
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        cv2.imwrite(str(out / f"{i}.png"), cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR))
    print(f"[clip] WROTE {out} n={n}", flush=True)


def emb(sid: str) -> None:
    import torch
    from torch.utils.data import DataLoader

    s = SLIDES[sid]
    slide_dir = ROOT / "Dataset/visium" / s["dir"]
    out = slide_dir / f"{FULLRES_KEY}.npy"
    if out.is_file():
        print(f"[emb] {sid}: {out.name} exists, skip", flush=True)
        return
    mod = _load("extract_fm_emb", ROOT / "scripts/common/extract_fm_emb.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, dim, _ = mod.load_phikon_v2(device)
    ds = mod.PatchFolder(slide_dir / "clip_image_fullres", transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    e = mod.extract(model, loader, device, dim)
    np.save(out, e.astype(np.float32))
    (slide_dir / f"{FULLRES_KEY}_meta.json").write_text(json.dumps({
        "slide": s["dir"], "encoder": "phikonv2", "shape": list(e.shape), "n_patches": len(ds),
        "source": s["tif"], "patch_px_fullres": PATCH_FULL, "resize": 512,
        "protocol": "AutoDL clip_spots.py full-res branch", "time": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"[emb] WROTE {out} shape={e.shape}", flush=True)


def compare() -> None:
    if "stamp" not in sys.modules:
        pkg = types.ModuleType("stamp")
        pkg.__path__ = [str(ROOT / "stamp")]
        sys.modules["stamp"] = pkg
    _load("stamp.graph_audit", ROOT / "stamp/graph_audit/__init__.py")
    core = _load("stamp.graph_audit.core", ROOT / "stamp/graph_audit/core.py")
    edges = _load("stamp.edges", ROOT / "stamp/edges.py")
    bench = _load("stamp.graph_audit_benchmark", ROOT / "stamp/graph_audit_benchmark.py")
    bench.setup_env()
    from sklearn.preprocessing import StandardScaler

    registry = {s["id"]: s for s in bench.load_slide_registry()}
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "slides": {}}
    for sid, s in SLIDES.items():
        spec = registry[sid]
        path, tf = spec["path"], spec.get("truth_file", "truth.txt")
        ref = bench.load_staig(path, spec["kind"], tf)
        raw = bench.load_raw(path, spec["kind"], tf)
        names = [b for b in ref.obs_names if b in raw.obs_names]
        ref, raw = ref[names].copy(), raw[names].copy()
        core.ensure_feat(raw)
        if "feat" not in ref.obsm:
            ref.obsm["feat"] = raw.obsm["feat"]
        fk = bench.fm_key(ref)
        full = np.load(ROOT / "Dataset/visium" / s["dir"] / f"{FULLRES_KEY}.npy").reshape(-1, 1024)
        if full.shape[0] != ref.n_obs:
            raise RuntimeError(f"{sid}: fullres {full.shape[0]} != {ref.n_obs}")
        variants = {"hires (current local)": ref.obsm[fk],
                    "full-res (AutoDL protocol)": StandardScaler().fit_transform(full).astype(np.float64)}
        g0 = np.array(ref.obsm["graph_neigh"], dtype=np.float64, copy=True)
        ep0 = np.array(ref.obsm["edge_probabilities"], dtype=np.float64, copy=True)
        e = pd.read_csv(s["edges"])
        sp = e["nn_mult"] <= 1.05
        ii, jj = e["i"].to_numpy(), e["j"].to_numpy()

        rows = []
        for name, fm in variants.items():
            ad = ref.copy()
            ad.obsm[fk] = fm
            z = edges._fm_matrix(ad, fk, pca_dim=16)
            zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
            cos = np.sum(zn[ii] * zn[jj], axis=1)
            ad.obsm["graph_neigh"] = g0.copy()
            ad.obsm["edge_probabilities"] = ep0.copy()
            edges.add_fm_skip_edges(ad, fk, k=1, drop_prob=0.05, pca_dim=16)
            sm, _ = core.score_graph(
                ad, np.asarray(ad.obsm["graph_neigh"]), name,
                img_key="img_emb" if "img_emb" in ad.obsm else None, fm_key=fk, notes="",
            )
            bench._attach_primary(sm, spec)
            n = int(sm.get("n_idc_dcis") or 0)
            nn, gene = sm.get("nn_mult_idc_dcis_median"), sm.get("gene_cos_idc_dcis_median")
            on = core.skip_rule_on(n, float(nn or 0), float(gene if gene is not None else float("nan")))
            rows.append({
                "variant": name, "n": n, "nn": None if nn is None else float(nn),
                "gene": None if gene is None else float(gene), "on": bool(on),
                "look": sm.get("primary_lookalike"), "remote": sm.get("primary_remote"),
                "corr_with_autodl_fm_cos_all_edges": float(np.corrcoef(cos, e["fm_cos"])[0, 1]),
                "corr_with_autodl_fm_cos_spatial_edges": float(np.corrcoef(cos[sp], e["fm_cos"][sp])[0, 1]),
                "median_cos_on_autodl_skip_edges": float(np.median(cos[~sp])),
                "autodl_median_cos_on_its_skip_edges": float(np.median(e["fm_cos"][~sp])),
            })
            print(sid, rows[-1], flush=True)
        out["slides"][sid] = {"frozen": s["frozen"], "rows": rows}

    (TAB / "reply10_phase2_fullres_reextract.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    def f(v, d=2):
        return "—" if v is None else f"{v:.{d}f}"

    L = [
        "# Reply-10 Phase 2 Step 1 — full-resolution Phikon-v2 re-extract vs frozen AutoDL audit",
        "",
        "> Local `phikonv2_emb.npy` was clipped from `tissue_hires_image.png` (~45 px patches → 512).",
        "> AutoDL `clip_spots.py` prefers the full-res tif (504 px patches → 512). Full-res tifs from 10x; new files only.",
        "> AutoDL edge lists: 2026-09-06 code snapshot. Rule unchanged. Graph-only.",
        "",
    ]
    for sid, d in out["slides"].items():
        fz = d["frozen"]
        L += [f"## {sid}", "",
              "| Embedding | n_collide | nn_mult | gene_cos | Switch | lookalike / remote | corr with AutoDL fm_cos (all / spatial edges) | cos on AutoDL skip edges |",
              "|-----------|----------:|--------:|---------:|--------|-------------------:|----------------------------------------------|-------------------------:|"]
        L.append(
            f"| **frozen AutoDL** | {fz['n'] if fz['n'] is not None else '—'} | {f(fz['nn'])} | {f(fz['gene'], 3)} | ON | "
            f"{fz['look']} / {fz['remote']} | 1 / 1 | {d['rows'][0]['autodl_median_cos_on_its_skip_edges']:.2f} |"
        )
        for r in d["rows"]:
            L.append(
                f"| {r['variant']} | {r['n']} | {f(r['nn'])} | {f(r['gene'], 3)} | {'ON' if r['on'] else 'OFF'} | "
                f"{r['look']} / {r['remote']} | {r['corr_with_autodl_fm_cos_all_edges']:.2f} / "
                f"{r['corr_with_autodl_fm_cos_spatial_edges']:.2f} | {r['median_cos_on_autodl_skip_edges']:.2f} |"
            )
        L.append("")
    (TAB / "reply10_phase2_fullres_reextract.md").write_text("\n".join(L), encoding="utf-8")
    print("WROTE", TAB / "reply10_phase2_fullres_reextract.md", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["clip", "emb", "compare", "all"], default="all")
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    sids = list(SLIDES) if args.only == "all" else [args.only]
    if args.stage in ("clip", "all"):
        for sid in sids:
            clip(sid)
    if args.stage in ("emb", "all"):
        for sid in sids:
            emb(sid)
    if args.stage in ("compare", "all"):
        compare()


if __name__ == "__main__":
    main()
