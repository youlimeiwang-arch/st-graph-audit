"""Part 3 — frozen FM embeddings from the same patches behind the Step 1 Phikon-v2 files. Inference only.

For each Step 1 dataset the patch folder is the one whose Phikon-v2 embedding Step 1 used (_prereg_common.SLIDES).
Writes <slide>/<encoder>_prereg_emb.npy (+ _meta.json), rows in patch order 0..N-1 = the Phikon-v2 row order.
Existing files are never overwritten.

Encoders: phikon, plip (public, transformers); uni (timm + HuggingFace access to MahmoodLab/UNI);
conch (the MahmoodLab CONCH package + access to MahmoodLab/conch). Weights go to .cache/ inside the repo.

--verify re-embeds the first 64 patches of every slide with Phikon-v2 and compares them with the Step 1 file, to
confirm the patch folder <-> embedding mapping before anything else is extracted.

  python scripts/prereg/extract_fm_prereg.py --verify
  python scripts/prereg/extract_fm_prereg.py --encoder phikon [--only BlockA]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

SLIDES = {  # same table as _prereg_common.SLIDES; kept here so extraction does not import the training stack
    "BlockA": ("visium", "Human_Breast_Cancer", "clip_image_swap_fullres", "phikonv2_xy_emb.npy"),
    "CytAssist_Patho": ("visium", "CytAssist_FFPE_Human_Breast_Cancer", "clip_image_swap_fullres", "phikonv2_xy_emb.npy"),
    "Wu": ("visium", "Wu_CID44971", "clip_image_xy", "phikonv2_xy_emb.npy"),
    "HPR_Prostate": ("visium", "HPR_Normal_Prostate", "clip_image_xy", "phikonv2_xy_emb.npy"),
    "Kidney_DKD1": ("visium", "Kidney_DKD1", "clip_image_xy", "phikonv2_xy_emb.npy"),
    "Liver_Yu_L5": ("visium", "Liver_Yu_L5", "clip_image_xy", "phikonv2_xy_emb.npy"),
    "Xenium_Rep1_IDC": ("visium", "Xenium_Rep1_IDC", "clip_image_fix2", "phikonv2_fix2_emb.npy"),
    "Hartman_BCSA_A1": ("visium", "Hartman_BCSA_A1", "clip_image_fix2", "phikonv2_fix2_emb.npy"),
    "CRC_Valdeolivas": ("visium", "CRC_Valdeolivas", "clip_image_fix2", "phikonv2_fix2_emb.npy"),
    "CRC_Valdeolivas_Rep2": ("visium", "CRC_Valdeolivas_Rep2", "clip_image_fix2", "phikonv2_fix2_emb.npy"),
    "DLPFC151507": ("DLPFC", "151507", "clip_image_xy", "phikonv2_xy_emb.npy"),
    "DLPFC151673": ("DLPFC", "151673", "clip_image_xy", "phikonv2_xy_emb.npy"),
}
VERIFY_N = 64


def _mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("extract_fm_emb", ROOT / "scripts/common/extract_fm_emb.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT
    mod.CKPT_DIR = ROOT / "stamp" / "fm_ckpts"
    return mod


def load_conch(device):
    import torch
    from conch.open_clip_custom import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
    model.eval().to(device)

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return model.encode_image(x, proj_contrast=False, normalize=False)

    return Encoder().to(device), preprocess, None, "conch"


def _size(size) -> int:
    if isinstance(size, (int, float)):
        return int(size)
    get = size.get if isinstance(size, dict) else (lambda k: getattr(size, k, None))
    return int(get("shortest_edge") or get("height") or 224)


def _transform(sz: int, mean, std):
    from torchvision import transforms

    return transforms.Compose([transforms.Resize(sz, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(sz), transforms.ToTensor(),
                               transforms.Normalize(mean=tuple(mean), std=tuple(std))])


def load_phikon(device):
    """As extract_fm_emb.load_phikon (CLS token of owkin/phikon), with the transformers 5 size format handled."""
    import torch
    from transformers import AutoImageProcessor, ViTModel

    proc = AutoImageProcessor.from_pretrained("owkin/phikon")
    vit = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False).eval().to(device)

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return vit(pixel_values=x).last_hidden_state[:, 0, :]

    return Encoder().to(device), _transform(_size(proc.size), proc.image_mean, proc.image_std), 768, "phikon"


def load_plip(device):
    """As extract_fm_emb.load_plip (CLIP image features of vinid/plip), with the transformers 5 size format handled."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    clip = CLIPModel.from_pretrained("vinid/plip").eval().to(device)
    ip = CLIPProcessor.from_pretrained("vinid/plip").image_processor

    class Encoder(torch.nn.Module):
        def forward(self, x):
            out = clip.get_image_features(pixel_values=x)
            return out if isinstance(out, torch.Tensor) else out.pooler_output

    return Encoder().to(device), _transform(_size(getattr(ip, "size", 224)), ip.image_mean, ip.image_std), 512, "plip"


def load_encoder(name: str, device):
    mod = _mod()
    fn = {"phikon": load_phikon, "phikonv2": mod.load_phikon_v2, "plip": load_plip, "uni": mod.load_uni,
          "conch": load_conch}[name]
    return fn(device), mod


def embed(model, transform, clip_dir: Path, device, limit: int | None = None) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, Subset

    mod = _mod()
    ds = mod.PatchFolder(clip_dir, transform)
    if limit:
        ds = Subset(ds, range(min(limit, len(ds))))
    out = []
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            f = model(batch.to(device))
            if isinstance(f, (tuple, list)):
                f = f[0]
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def verify(only: list[str]) -> None:
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (model, transform, _, _), _ = load_encoder("phikonv2", dev)
    for ds in only:
        sub, slide, clip, ref = SLIDES[ds]
        sd = ROOT / "Dataset" / sub / slide
        n_png = len(list((sd / clip).glob("*.png")))
        e_ref = np.load(sd / ref, mmap_mode="r")
        e = embed(model, transform, sd / clip, dev, VERIFY_N)
        r = np.asarray(e_ref[: len(e)], dtype=np.float32)
        cos = np.sum(e * r, axis=1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(r, axis=1) + 1e-8)
        status = "OK" if n_png == e_ref.shape[0] and cos.min() > 0.99 else "MISMATCH"
        print(f"{status:8s} {ds:22s} patches={n_png} ref_rows={e_ref.shape[0]} "
              f"cos(first {len(e)}): min={cos.min():.4f} median={np.median(cos):.4f}", flush=True)


def extract(name: str, only: list[str]) -> None:
    import torch

    todo = []
    for ds in only:
        sub, slide, clip, ref = SLIDES[ds]
        sd = ROOT / "Dataset" / sub / slide
        out = sd / f"{name}_prereg_emb.npy"
        if out.exists():
            print(f"SKIP {ds}: {out.name} exists", flush=True)
            continue
        todo.append((ds, sd, clip, ref, out))
    if not todo:
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (model, transform, _, _), _ = load_encoder(name, dev)
    for ds, sd, clip, ref, out in todo:
        n_ref = np.load(sd / ref, mmap_mode="r").shape[0]
        e = embed(model, transform, sd / clip, dev)
        if e.shape[0] != n_ref:
            raise RuntimeError(f"{ds}: {e.shape[0]} embeddings != {n_ref} rows in {ref}")
        np.save(out, e.astype(np.float32))
        meta = {"dataset": ds, "slide": sd.name, "encoder": name, "patches": clip, "row_order_as": ref,
                "shape": list(e.shape), "l2_normalized": True, "frozen": True,
                "time": datetime.now(timezone.utc).isoformat()}
        (sd / f"{out.stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"WROTE {out} shape={e.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=["phikon", "plip", "uni", "conch"])
    ap.add_argument("--only", default="all", help="dataset id or 'all'")
    ap.add_argument("--verify", action="store_true", help="check patches vs the Step 1 Phikon-v2 file (64 per slide)")
    ap.add_argument("--list", action="store_true", help="only list patch folders and existing files")
    args = ap.parse_args()
    only = list(SLIDES) if args.only == "all" else [args.only]
    if any(d not in SLIDES for d in only):
        raise SystemExit(f"--only must be one of {list(SLIDES)}")
    if args.list:
        for ds in only:
            sub, slide, clip, ref = SLIDES[ds]
            sd = ROOT / "Dataset" / sub / slide
            have = sorted(p.name for p in sd.glob("*_prereg_emb.npy"))
            print(f"{ds:22s} {clip:24s} patches={len(list((sd / clip).glob('*.png')))} "
                  f"ref={ref} rows={np.load(sd / ref, mmap_mode='r').shape[0]} prereg={have or '-'}")
        return
    if args.verify:
        verify(only)
        return
    if not args.encoder:
        raise SystemExit("give --encoder, --verify or --list")
    extract(args.encoder, only)


if __name__ == "__main__":
    main()
