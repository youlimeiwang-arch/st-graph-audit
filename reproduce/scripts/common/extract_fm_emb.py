"""Frozen pathology FM embeddings for STAMP M1. Does not train the encoder.

Preferred: UNI (ViT-L/16, 1024-d). Needs a HuggingFace token that has accepted
MahmoodLab/UNI. Fallback: PLIP (public CLIP pathology encoder, 512-d).

Saves Dataset/DLPFC/<slide>/uni_emb.npy (or plip_emb.npy) in the same spot
order as clip_image/{i}.png, which matches LoadSingle10xAdata after filter_na.
Never overwrites embeddings.npy (STAIG BYOL).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("TORCH_HOME", "/root/autodl-tmp/torch_cache")
os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path("/root/autodl-tmp/STAIG")
CKPT_DIR = ROOT / "stamp" / "fm_ckpts"


class PatchFolder(Dataset):
    def __init__(self, clip_dir: Path, transform):
        self.clip_dir = clip_dir
        self.transform = transform
        self.names = sorted(
            [p for p in os.listdir(clip_dir) if p.endswith(".png")],
            key=lambda x: int(Path(x).stem),
        )
        missing = [i for i, n in enumerate(self.names) if int(Path(n).stem) != i]
        if missing:
            raise RuntimeError(f"{clip_dir} png names are not 0..N-1 contiguous")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = Image.open(self.clip_dir / self.names[idx]).convert("RGB")
        return self.transform(img)


def imagenet_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_uni(device: torch.device):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    local = CKPT_DIR / "uni"
    local.mkdir(exist_ok=True)
    bin_path = local / "pytorch_model.bin"
    if not bin_path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=str(local))
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(bin_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    try:
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    except Exception:
        transform = imagenet_transform(224)
    return model, transform, 1024, "uni_emb.npy"


def load_phikon_v2(device: torch.device):
    from transformers import AutoImageProcessor, AutoModel

    name = "owkin/phikon-v2"
    processor = AutoImageProcessor.from_pretrained(name)
    vit = AutoModel.from_pretrained(name)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False
    vit = vit.to(device)
    size = processor.size
    if isinstance(size, dict):
        sz = int(size.get("shortest_edge") or size.get("height") or 224)
    else:
        # transformers SizeDict / object with attributes
        sz = int(getattr(size, "shortest_edge", None) or getattr(size, "height", None) or 224)
    transform = transforms.Compose(
        [
            transforms.Resize(sz, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(processor.image_mean), std=tuple(processor.image_std)),
        ]
    )

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return vit(pixel_values=x).last_hidden_state[:, 0, :]

    return Encoder().to(device), transform, 1024, "phikonv2_emb.npy"


def load_phikon(device: torch.device):
    from transformers import AutoImageProcessor, ViTModel

    processor = AutoImageProcessor.from_pretrained("owkin/phikon")
    vit = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False
    vit = vit.to(device)
    size = processor.size
    if isinstance(size, dict):
        sz = int(size.get("shortest_edge") or size.get("height") or 224)
    else:
        sz = int(size)
    transform = transforms.Compose(
        [
            transforms.Resize(sz, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(processor.image_mean), std=tuple(processor.image_std)),
        ]
    )

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return vit(pixel_values=x).last_hidden_state[:, 0, :]

    return Encoder().to(device), transform, 768, "phikon_emb.npy"


def load_plip(device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    name = "vinid/plip"
    clip = CLIPModel.from_pretrained(name)
    processor = CLIPProcessor.from_pretrained(name)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False
    clip = clip.to(device)
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std
    size = getattr(processor.image_processor, "size", {"shortest_edge": 224})
    if isinstance(size, dict):
        sz = int(size.get("shortest_edge") or size.get("height") or 224)
    else:
        sz = int(size)
    transform = transforms.Compose(
        [
            transforms.Resize(sz, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(image_mean), std=tuple(image_std)),
        ]
    )

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return clip.get_image_features(pixel_values=x)

    return Encoder().to(device), transform, 512, "plip_emb.npy"


def extract(model, loader, device, out_dim: int) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            feat = model(batch)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = torch.nn.functional.normalize(feat.float(), dim=-1)
            chunks.append(feat.cpu().numpy())
    emb = np.concatenate(chunks, axis=0)
    if emb.ndim != 2 or emb.shape[1] != out_dim:
        raise RuntimeError(f"unexpected embedding shape {emb.shape}, expected (*, {out_dim})")
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", default="151673")
    parser.add_argument("--dataset", default="DLPFC")
    parser.add_argument(
        "--encoder",
        default="phikon",
        choices=["uni", "phikon", "phikonv2", "plip"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    slide_dir = ROOT / "Dataset" / args.dataset / args.slide
    clip_dir = slide_dir / "clip_image"
    if not clip_dir.exists() or not any(clip_dir.glob("*.png")):
        raise FileNotFoundError(f"missing patches: {clip_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== extract {args.encoder} slide={args.slide} device={device} ===", flush=True)
    loaders = {"uni": load_uni, "phikon": load_phikon, "phikonv2": load_phikon_v2, "plip": load_plip}
    model, transform, dim, fname = loaders[args.encoder](device)
    ds = PatchFolder(clip_dir, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    emb = extract(model, loader, device, dim)
    out = slide_dir / fname
    np.save(out, emb.astype(np.float32))
    meta = {
        "slide": args.slide,
        "dataset": args.dataset,
        "encoder": args.encoder,
        "path": str(out),
        "shape": list(emb.shape),
        "n_patches": len(ds),
        "l2_normalized": True,
        "frozen": True,
    }
    (slide_dir / f"{Path(fname).stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"WROTE {out} shape={emb.shape} mean_norm={np.linalg.norm(emb, axis=1).mean():.4f}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    main()
