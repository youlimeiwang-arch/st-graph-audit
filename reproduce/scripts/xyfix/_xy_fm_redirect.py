"""In-process redirect of transposed Phikon-v2 files to the corrected-coordinate re-extract.

install() patches numpy.load so that, for any slide that has phikonv2_xy_emb.npy:
  <slide>/phikonv2_emb.npy                          -> <slide>/phikonv2_xy_emb.npy
  <slide>/embeddings.npy (byte copy of phikonv2_emb) -> <slide>/phikonv2_xy_emb.npy
BYOL embeddings.npy (not a copy of phikonv2_emb.npy) is left alone. Files on disk are never touched.
STAIG_FM_PREFER (comma-separated file names, e.g. phikonv2_fix2_emb.npy) is checked before
phikonv2_xy_emb.npy on every slide that has one of those files; unset = original behaviour.
"""
from __future__ import annotations

import filecmp
import os
from pathlib import Path

import numpy as np

XY_NAME = "phikonv2_xy_emb.npy"
_orig_load = np.load
_seen: dict[str, str] = {}


def _target(p: Path) -> Path | None:
    names = [n.strip() for n in os.environ.get("STAIG_FM_PREFER", "").split(",") if n.strip()] + [XY_NAME]
    xy = next((p.parent / n for n in names if (p.parent / n).is_file()), None)
    if xy is None:
        return None
    if p.name == "phikonv2_emb.npy":
        return xy
    if p.name == "embeddings.npy":
        old = p.parent / "phikonv2_emb.npy"
        if old.is_file() and p.stat().st_size == old.stat().st_size and filecmp.cmp(p, old, shallow=False):
            return xy
    return None


def _load(file, *args, **kwargs):
    if isinstance(file, (str, os.PathLike)):
        p = Path(file)
        key = str(p.resolve())
        if key not in _seen:
            t = _target(p)
            _seen[key] = str(t) if t is not None else ""
            if t is not None:
                print(f"[xy-redirect] {p} -> {t.name}", flush=True)
        if _seen[key]:
            file = _seen[key]
    return _orig_load(file, *args, **kwargs)


def install() -> None:
    np.load = _load


def _xy_name(p: Path) -> Path:
    return p if p.stem.endswith("_xy") else p.with_name(f"{p.stem}_xy{p.suffix}")


def install_write_guard(dataset_root: Path) -> None:
    """np.save / to_csv into dataset_root get an _xy suffix; copies onto embeddings.npy are skipped."""
    import shutil

    import pandas as pd

    root = str(Path(dataset_root).resolve()).lower()

    def inside(target) -> bool:
        return isinstance(target, (str, os.PathLike)) and str(Path(target).resolve()).lower().startswith(root)

    orig_save = np.save

    def save(file, arr, *a, **k):
        if inside(file):
            file = _xy_name(Path(file if str(file).endswith(".npy") else f"{file}.npy"))
        return orig_save(file, arr, *a, **k)

    np.save = save

    for cls in (pd.DataFrame, pd.Series):
        orig = cls.to_csv

        def to_csv(self, path_or_buf=None, *a, _orig=orig, **k):
            if inside(path_or_buf):
                path_or_buf = _xy_name(Path(path_or_buf))
            return _orig(self, path_or_buf, *a, **k)

        cls.to_csv = to_csv

    orig_copy2 = shutil.copy2

    def copy2(src, dst, *a, **k):
        if inside(dst) and Path(dst).name == "embeddings.npy":
            print(f"[xy] skip copy {src} -> {dst} (redirected in memory)", flush=True)
            return dst
        return orig_copy2(src, dst, *a, **k)

    shutil.copy2 = copy2


def redirects() -> dict[str, str]:
    return {k: v for k, v in _seen.items() if v}
