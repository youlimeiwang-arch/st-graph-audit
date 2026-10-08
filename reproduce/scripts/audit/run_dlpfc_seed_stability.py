"""Reply-8: DLPFC ARI direction check across seeds (ungated vs gated only).

Usage:
  python scripts/audit/run_dlpfc_seed_stability.py --seeds 39788,42
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

if "harmonypy" not in sys.modules:
    import types

    _hm = types.ModuleType("harmonypy")

    def _compute_lisi(emb, *args, **kwargs):
        return [[0.0] for _ in range(emb.shape[0])]

    _hm.compute_lisi = _compute_lisi
    sys.modules["harmonypy"] = _hm

_R_HOME = os.environ.get("R_HOME") or r"C:\Program Files\R\R-4.6.0"
_R_BIN = Path(_R_HOME) / "bin" / "x64"
if _R_BIN.is_dir():
    os.environ["R_HOME"] = str(_R_HOME)
    os.environ["PATH"] = str(_R_BIN) + os.pathsep + os.environ.get("PATH", "")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from yaml import SafeLoader

warnings.filterwarnings("ignore")

from staig.adata_processing import LoadSingle10xAdata
from stamp.edges import add_fm_skip_edges, recompute_edge_probabilities
from stamp.model import STAMP

FM_KEY = "phikonv2_emb"

JOBS = [
    {"id": "DLPFC151673", "slide": "151673", "dataset": "DLPFC", "config_key": 151673, "k": 80, "epochs": 300},
    {"id": "DLPFC151507", "slide": "151507", "dataset": "DLPFC", "config_key": 151507, "k": 80, "epochs": 300},
]


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(12345)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def run_one(job: dict, seed: int, gene_k: int) -> float:
    slide_dir = ROOT / "Dataset/DLPFC" / job["slide"]
    cfg = copy.deepcopy(yaml.load(open(ROOT / "train_img_config.yaml"), Loader=SafeLoader)[job["config_key"]])
    n_truth = int(pd.read_csv(slide_dir / "truth.txt", sep="\t", header=None)[1].dropna().nunique())
    cfg["seed"] = seed
    cfg["k"] = job["k"]
    cfg["num_epochs"] = job["epochs"]
    cfg["num_clusters"] = n_truth
    cfg["img_key"] = "img_emb"
    cfg["img_dim"] = 16
    cfg["ckpt_dir"] = "stamp/checkpoints"
    cfg["fusion_mode"] = "identity"
    cfg["fusion_warmup"] = 0
    cfg["fusion_lr"] = cfg["learning_rate"] * 0.1
    cfg["res_scale"] = 0.05

    seed_all(seed)
    data = LoadSingle10xAdata(
        path=str(slide_dir),
        n_neighbors=cfg["num_neigh"],
        n_top_genes=cfg["num_gene"],
        image_emb=True,
        label=True,
    ).run()
    extra = np.load(slide_dir / f"{FM_KEY}.npy").reshape(data.n_obs, -1)
    extra = StandardScaler().fit_transform(extra).astype(np.float64)
    data.obsm[FM_KEY] = extra
    x = PCA(n_components=16, random_state=0).fit_transform(extra)
    data.obsm["img_emb"] = StandardScaler().fit_transform(x).astype(np.float64)
    recompute_edge_probabilities(data, "img_emb", pca_dim=0)
    add_fm_skip_edges(
        data,
        FM_KEY,
        k=1,
        drop_prob=0.05,
        pca_dim=16,
        gene_key="feat" if gene_k else "",
        gene_k=gene_k,
        gene_mutual=bool(gene_k),
    )
    import argparse

    args = argparse.Namespace(dataset=job["dataset"], slide=job["slide"], config="train_img_config.yaml", label=True)
    model = STAMP(args=args, config=cfg, single=False, refine=True)
    model.adata = data
    model.train()
    model.eva()
    model.cluster(True)
    y = data.obs["ground_truth"].astype(str)
    pred = model.adata.obs["domain"].astype(str)
    return float(adjusted_rand_score(y, pred))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="39788,42")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for job in JOBS:
        for seed in seeds:
            print(f"== {job['id']} seed={seed} ungated", flush=True)
            u = run_one(job, seed, 0)
            print(f"   ARI_u={u:.4f}", flush=True)
            print(f"== {job['id']} seed={seed} gated", flush=True)
            g = run_one(job, seed, 10)
            print(f"   ARI_g={g:.4f} delta={g-u:+.4f}", flush=True)
            rows.append(
                {
                    "slide": job["id"],
                    "seed": seed,
                    "ari_ungated": u,
                    "ari_gated": g,
                    "delta_gated_minus_ungated": g - u,
                    "direction_up": g > u,
                }
            )
    out = ROOT / "lab/static/tables/dlpfc_seed_stability_reply8.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = ROOT / "lab/static/tables/dlpfc_seed_stability_reply8.md"
    lines = [
        "# DLPFC seed stability (Reply-8)",
        "",
        "| slide | seed | ARI ungated | ARI gated | Δ (g−u) | direction up |",
        "|-------|-----:|------------:|----------:|--------:|:------------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['slide']} | {r['seed']} | {r['ari_ungated']:.4f} | {r['ari_gated']:.4f} | "
            f"{r['delta_gated_minus_ungated']:+.4f} | {r['direction_up']} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
