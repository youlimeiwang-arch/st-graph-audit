"""Reply-10 Phase 2 — which encoder/preprocessing could have produced the frozen AutoDL fm_cos?

AutoDL's phikonv2_emb is self-consistent (logs/breast_skip_patches.json == frozen edge CSV) and
spatially smooth, but uncorrelated with real Phikon-v2 on the same spots. Probe candidate encoders
on the local patches and correlate their PCA-16 cosine with AutoDL fm_cos on the frozen edges.
New files only: lab/static/tables/reply10_phase2_autodl_encoder_probe.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "phase2_coord_diag"))
import run_phase2_fullres_reextract as fr  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

SLIDES = {"BlockA": "Human_Breast_Cancer", "CytAssist_Patho": "CytAssist_FFPE_Human_Breast_Cancer"}
OUT = ROOT / "lab/static/tables/reply10_phase2_autodl_encoder_probe.json"


def fmz(x: np.ndarray) -> np.ndarray:
    z = StandardScaler().fit_transform(StandardScaler().fit_transform(x))
    z = StandardScaler().fit_transform(PCA(16, random_state=42).fit_transform(z))
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def score(emb: np.ndarray, e: pd.DataFrame) -> dict:
    z = fmz(emb)
    c = np.sum(z[e["i"].to_numpy()] * z[e["j"].to_numpy()], axis=1)
    sp = (e["nn_mult"] <= 1.05).to_numpy()
    return {"corr_all": float(np.corrcoef(c, e["fm_cos"])[0, 1]),
            "corr_spatial": float(np.corrcoef(c[sp], e["fm_cos"][sp])[0, 1]),
            "cos_on_autodl_skip": float(np.median(c[~sp]))}


def color_feats(clip_dir: Path, n: int) -> np.ndarray:
    rows = []
    for i in range(n):
        a = np.asarray(Image.open(clip_dir / f"{i}.png").convert("RGB").resize((128, 128)), dtype=np.float32) / 255
        hsv = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).convert("HSV"), dtype=np.float32) / 255
        f = [a.mean((0, 1)), a.std((0, 1)), hsv.mean((0, 1)), hsv.std((0, 1))]
        for ch in range(3):
            f.append(np.histogram(a[..., ch], bins=8, range=(0, 1))[0] / a[..., ch].size)
        rows.append(np.concatenate(f))
    return np.asarray(rows)


def run_model(model_fn, transform, clip_dir: Path, device) -> np.ndarray:
    mod = fr._load("extract_fm_emb", ROOT / "scripts/common/extract_fm_emb.py")
    ds = mod.PatchFolder(clip_dir, transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    return mod.extract(model_fn, loader, device, 1024)


def main() -> None:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "owkin/phikon-v2"
    proc = AutoImageProcessor.from_pretrained(name)
    mean, std = tuple(proc.image_mean), tuple(proc.image_std)
    norm_t = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                 transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    raw_t = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.CenterCrop(224), transforms.ToTensor()])
    full_t = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(mean, std)])

    def wrap(vit, token="cls"):
        class Enc(torch.nn.Module):
            def forward(self, x):
                h = vit(pixel_values=x).last_hidden_state
                return h[:, 0, :] if token == "cls" else h[:, 1:, :].mean(1)
        return Enc().to(device)

    pre = AutoModel.from_pretrained(name).eval().to(device)
    cfg = AutoConfig.from_pretrained(name)
    torch.manual_seed(0)
    rnd = AutoModel.from_config(cfg).eval().to(device)

    out = {}
    for sid, d in SLIDES.items():
        sd = ROOT / "Dataset/visium" / d
        e = pd.read_csv(fr.SLIDES[sid]["edges"])
        res = {}
        for patches in ("clip_image_fullres", "clip_image_hires"):
            cdir = sd / patches
            n = len(list(cdir.glob("*.png")))
            variants = {
                "phikonv2 pretrained CLS (reference)": lambda: run_model(wrap(pre), norm_t, cdir, device),
                "phikonv2 pretrained mean-patch-token": lambda: run_model(wrap(pre, "mean"), norm_t, cdir, device),
                "phikonv2 pretrained, no normalisation": lambda: run_model(wrap(pre), raw_t, cdir, device),
                "phikonv2 pretrained, 512 px no crop": lambda: run_model(wrap(pre), full_t, cdir, device),
                "phikonv2 architecture RANDOM init": lambda: run_model(wrap(rnd), norm_t, cdir, device),
                "colour statistics (no network)": lambda: color_feats(cdir, n),
            }
            for vname, fn in variants.items():
                key = f"{patches} | {vname}"
                with torch.inference_mode():
                    emb = fn()
                res[key] = score(np.asarray(emb, dtype=np.float64), e)
                print(sid, key, {k: round(v, 3) for k, v in res[key].items()}, flush=True)
        out[sid] = res
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
