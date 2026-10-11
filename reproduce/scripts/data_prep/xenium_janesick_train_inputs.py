"""Training inputs for Xenium_Rep1_IDC with Janesick labels (truth_janesick.txt, 3975 of 4006 bins labeled).

LoadSingle10xAdata assigns labels by row position and drops empty ones, and the runners need FM rows equal to the
kept spots. So this writes, as new files only:
  truth_janesick_full.txt      all 4006 bins in truth.txt order, empty label for the 31 bins with no labeled cell
  <fm>_janesick_emb.npy        FM rows of the 3975 labeled bins (phikonv2 from phikonv2_fix2_emb.npy,
                               the others from <fm>_prereg_emb.npy)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SLIDE = ROOT / "Dataset/visium/Xenium_Rep1_IDC"
SRC = {"phikonv2": "phikonv2_fix2_emb.npy", "phikon": "phikon_prereg_emb.npy", "plip": "plip_prereg_emb.npy",
       "uni": "uni_prereg_emb.npy", "conch": "conch_prereg_emb.npy"}


def main() -> None:
    out_truth = SLIDE / "truth_janesick_full.txt"
    outs = [out_truth] + [SLIDE / f"{fm}_janesick_emb.npy" for fm in SRC]
    taken = [p.name for p in outs if p.exists()]
    if taken:
        raise SystemExit(f"refusing to overwrite {taken}")

    order = pd.read_csv(SLIDE / "truth.txt", sep="\t", header=None, dtype=str)[0]
    new = pd.read_csv(SLIDE / "truth_janesick.txt", sep="\t", header=None, dtype=str).set_index(0)[1]
    if not new.index.isin(order).all():
        raise SystemExit("truth_janesick.txt has barcodes not in truth.txt")
    lab = new.reindex(order)
    keep = lab.notna().to_numpy()
    out_truth.write_text("".join(f"{bc}\t{'' if pd.isna(v) else v}\n" for bc, v in zip(order, lab)),
                         encoding="utf-8", newline="\n")

    check = pd.read_csv(out_truth, sep="\t", header=None)[1]
    if len(check) != len(order) or int(check.notna().sum()) != int(keep.sum()):
        raise SystemExit("truth_janesick_full.txt does not read back as expected")

    for fm, name in SRC.items():
        x = np.load(SLIDE / name)
        if x.shape[0] != len(order):
            raise SystemExit(f"{name}: {x.shape[0]} rows, expected {len(order)}")
        np.save(SLIDE / f"{fm}_janesick_emb.npy", x[keep])
        print(f"{fm}: {name} {x.shape} -> {fm}_janesick_emb.npy {x[keep].shape}", flush=True)
    print(f"truth_janesick_full.txt: {len(order)} rows, {int(keep.sum())} labeled, "
          f"{int(lab.dropna().nunique())} classes")


if __name__ == "__main__":
    main()
