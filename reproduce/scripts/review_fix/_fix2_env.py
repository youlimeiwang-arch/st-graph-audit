"""In-memory environment for the review-fix (fix2) reruns. Import before anything reads a slide.

- Phikon-v2: phikonv2_emb.npy (and its embeddings.npy byte copy) -> phikonv2_fix2_emb.npy where it exists,
  else phikonv2_xy_emb.npy, else the original (scripts/xyfix/_xy_fm_redirect.py, STAIG_FM_PREFER).
- Truth: LoadSingle10xAdata.load_label reads TRUTH_OVERRIDE[slide] instead of truth.txt, so no truth file is
  ever copied over truth.txt (the staging used by the original runners is disabled by the callers).
- Writes into Dataset/ get a _fix2 suffix; copies onto embeddings.npy are skipped.
STAIG_FIX2_FM (e.g. phikonv2_p144_emb.npy) replaces phikonv2_fix2_emb.npy as the preferred file.
STAIG_HARTMAN_TRUTH (e.g. truth_var_core.txt) replaces truth_filled.txt for Hartman_BCSA_A1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX2_EMB = os.environ.get("STAIG_FIX2_FM", "phikonv2_fix2_emb.npy")
os.environ["STAIG_FM_PREFER"] = FIX2_EMB
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "xyfix"))
import _xy_fm_redirect  # noqa: E402

TRUTH_OVERRIDE = {"Hartman_BCSA_A1": os.environ.get("STAIG_HARTMAN_TRUTH", "truth_filled.txt")}
FIX2_SLIDES = ("Xenium_Rep1_IDC", "Hartman_BCSA_A1", "CRC_Valdeolivas", "CRC_Valdeolivas_Rep2")

_xy_fm_redirect._xy_name = lambda p: p if p.stem.endswith("_fix2") else p.with_name(f"{p.stem}_fix2{p.suffix}")
_xy_fm_redirect.install()
_xy_fm_redirect.install_write_guard(ROOT / "Dataset")


def truth_path(slide_dir) -> Path:
    slide_dir = Path(slide_dir)
    return slide_dir / TRUTH_OVERRIDE.get(slide_dir.name, "truth.txt")


def _patch_load_label() -> None:
    import pandas as pd

    from staig.adata_processing import LoadSingle10xAdata

    def load_label(self):
        df_meta = pd.read_csv(truth_path(self.path), sep="\t", header=None)
        self.adata.obs["ground_truth"] = df_meta[1].values
        if self.filter_na:
            self.adata = self.adata[~pd.isnull(self.adata.obs["ground_truth"])]

    LoadSingle10xAdata.load_label = load_label


_patch_load_label()


def check_fix2_files(slides=FIX2_SLIDES) -> None:
    missing = [s for s in slides if not (ROOT / "Dataset/visium" / s / FIX2_EMB).is_file()]
    if missing:
        raise SystemExit(f"missing {FIX2_EMB} for {missing}")


def redirects() -> dict[str, str]:
    return _xy_fm_redirect.redirects()
