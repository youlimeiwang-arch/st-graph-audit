"""Reply-10 Phase 2 Step 1 — multi-seed ARI for the existing skip rule.

Three arms per dataset, each dataset on its own original recipe (only seed / arm vary):
  none     = spatial graph only (no FM skip) — the Skip=OFF deployment
  ungated  = FM skip k=1, gene_k=0
  gated    = FM skip k=1, gene_k=10 mutual (Strategy A)

One JSON line per finished run in logs/reply10_phase2_multiseed.jsonl (resumable).
No thresholds are touched; no new datasets.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
import types
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

if "harmonypy" not in sys.modules:
    _hm = types.ModuleType("harmonypy")
    _hm.compute_lisi = lambda emb, *a, **k: [[0.0] for _ in range(emb.shape[0])]
    sys.modules["harmonypy"] = _hm

_R_HOME = os.environ.get("R_HOME") or r"C:\Program Files\R\R-4.6.0"
_R_BIN = Path(_R_HOME) / "bin" / "x64"
if _R_BIN.is_dir():
    os.environ["R_HOME"] = str(_R_HOME)
    os.environ["PATH"] = str(_R_BIN) + os.pathsep + os.environ.get("PATH", "")
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from yaml import SafeLoader

from staig.adata_processing import LoadSingle10xAdata
from stamp.edges import add_fm_skip_edges, recompute_edge_probabilities
from stamp.model import STAMP

LOG = ROOT / "logs"
JSONL = LOG / "reply10_phase2_multiseed.jsonl"
FM_KEY = "phikonv2_emb"
SEEDS = [39788, 42, 0, 1, 123]
ARMS = ["none", "ungated", "gated"]

# recipe fields:
#   img: "byol" keeps LoadSingle10xAdata BYOL img_emb + edge probs (Block A freeze);
#        "fm_pca" replaces img_emb with PCA-16 of Phikon-v2 (all other slides)
#   k / epochs: None = keep train_img_config.yaml value
#   det: torch.use_deterministic_algorithms as in the original runner
#   ncl: None = number of truth classes
RECIPES = {
    "Kidney_DKD1": dict(slide="Kidney_DKD1", truth="truth.txt", img="fm_pca", k=None, epochs=None, det=False, ncl=None),
    "Wu": dict(slide="Wu_CID44971", truth="truth.txt", img="fm_pca", k=280, epochs=580, det=True, ncl=None),
    "HPR_Prostate": dict(slide="HPR_Normal_Prostate", truth="truth.txt", img="fm_pca", k=280, epochs=580, det=True, ncl=None),
    "CytAssist_Patho": dict(
        slide="CytAssist_FFPE_Human_Breast_Cancer", truth="truth_pathologist.txt", img="fm_pca",
        k=280, epochs=580, det=True, ncl=None,
    ),
    "Hartman_BCSA_A1": dict(slide="Hartman_BCSA_A1", truth="truth.txt", img="fm_pca", k=None, epochs=None, det=False, ncl=None),
    "CRC_Valdeolivas": dict(slide="CRC_Valdeolivas", truth="truth.txt", img="fm_pca", k=280, epochs=580, det=True, ncl=None),
    "CRC_Valdeolivas_Rep2": dict(slide="CRC_Valdeolivas_Rep2", truth="truth.txt", img="fm_pca", k=280, epochs=580, det=True, ncl=None),
    "BlockA": dict(slide="Human_Breast_Cancer", truth="truth.txt", img="byol", k=280, epochs=580, det=True, ncl=20),
    "Liver_Yu_L5": dict(slide="Liver_Yu_L5", truth="truth.txt", img="fm_pca", k=None, epochs=None, det=False, ncl=None),
    "Xenium_Rep1_IDC": dict(slide="Xenium_Rep1_IDC", truth="truth.txt", img="fm_pca", k=None, epochs=None, det=False, ncl=None),
}
ORDER = list(RECIPES)


def seed_all(seed: int, det: bool) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(12345)
    if det:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def n_truth_classes(path: Path) -> int:
    labs = pd.read_csv(path, sep="\t", header=None)[1].dropna().astype(str)
    return int(labs[~labs.isin({"", "nan", "NaN", "None"})].nunique())


class StagedTruth:
    """LoadSingle10xAdata always reads truth.txt; stage an alternate truth file and restore after."""

    def __init__(self, slide_dir: Path, truth_name: str):
        self.dir, self.name = slide_dir, truth_name
        self.bak = slide_dir / "truth.txt.__bak_reply10_ms__"
        self.staged = False

    def __enter__(self):
        if self.name == "truth.txt":
            return self
        src, dst = self.dir / self.name, self.dir / "truth.txt"
        if dst.is_file() and src.read_bytes() == dst.read_bytes():
            return self
        if dst.is_file():
            shutil.copy2(dst, self.bak)
        shutil.copy2(src, dst)
        self.staged = True
        return self

    def __exit__(self, *exc):
        if self.staged and self.bak.is_file():
            shutil.copy2(self.bak, self.dir / "truth.txt")
            self.bak.unlink(missing_ok=True)
        return False


def run_one(ds: str, arm: str, seed: int) -> dict:
    rc = RECIPES[ds]
    slide_dir = ROOT / "Dataset" / "visium" / rc["slide"]
    if rc["img"] == "fm_pca":
        src, dst = slide_dir / f"{FM_KEY}.npy", slide_dir / "embeddings.npy"
        if (not dst.is_file()) or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)

    with StagedTruth(slide_dir, rc["truth"]):
        cfg = copy.deepcopy(yaml.load(open(ROOT / "train_img_config.yaml"), Loader=SafeLoader)[rc["slide"]])
        n_truth = n_truth_classes(slide_dir / "truth.txt")
        cfg["seed"] = seed
        if rc["k"] is not None:
            cfg["k"] = rc["k"]
        if rc["epochs"] is not None:
            cfg["num_epochs"] = rc["epochs"]
        cfg["num_clusters"] = rc["ncl"] or n_truth
        cfg["img_key"] = "img_emb"
        cfg["img_dim"] = 16
        cfg["ckpt_dir"] = "stamp/checkpoints"
        cfg["fusion_mode"] = "identity"
        cfg["fusion_warmup"] = 0
        cfg["fusion_lr"] = cfg.get("learning_rate", 5e-4) * 0.1
        cfg["res_scale"] = 0.05
        cfg.setdefault("drop_edge_rate_1", 0.1)
        cfg.setdefault("drop_edge_rate_2", 0.3)
        cfg.setdefault("tau_decay", 0.0)

        print(f"=== {ds} arm={arm} seed={seed} ncl={cfg['num_clusters']} ===", flush=True)
        t0 = time.time()
        seed_all(seed, rc["det"])
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
    if rc["img"] == "fm_pca":
        x = PCA(n_components=16, random_state=0).fit_transform(extra)
        data.obsm["img_emb"] = StandardScaler().fit_transform(x).astype(np.float64)
        recompute_edge_probabilities(data, "img_emb", pca_dim=0)

    if arm == "ungated":
        add_fm_skip_edges(data, FM_KEY, k=1, drop_prob=0.05, pca_dim=16, gene_key="", gene_k=0, gene_mutual=False)
    elif arm == "gated":
        add_fm_skip_edges(data, FM_KEY, k=1, drop_prob=0.05, pca_dim=16, gene_key="feat", gene_k=10, gene_mutual=True)

    args = argparse.Namespace(dataset="visium", slide=rc["slide"], config="train_img_config.yaml", label=True)
    model = STAMP(args=args, config=cfg, single=False, refine=True)
    model.adata = data
    model.train()
    model.eva()
    model.cluster(True)

    y = data.obs["ground_truth"].astype(str)
    pred = model.adata.obs["domain"].astype(str)
    row = {
        "dataset": ds,
        "slide": rc["slide"],
        "arm": arm,
        "seed": seed,
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "n": int(data.n_obs),
        "ncl": int(cfg["num_clusters"]),
        "k": int(cfg.get("k", -1)),
        "epochs": int(cfg["num_epochs"]),
        "recipe": rc,
        "sec": round(time.time() - t0, 1),
        "time": datetime.now(timezone.utc).isoformat(),
    }
    print(f"DONE {ds} {arm} seed={seed} ARI={row['ari']:.4f} ({row['sec']}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", help="comma list of dataset ids or 'all'")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    datasets = ORDER if args.only == "all" else [d.strip() for d in args.only.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]

    # CUBLAS_WORKSPACE_CONFIG only takes effect before the first cuBLAS handle, so each
    # dataset gets a fresh process, matching its original single-dataset runner.
    if len(datasets) > 1:
        import subprocess

        for ds in datasets:
            subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--only", ds,
                 "--seeds", args.seeds, "--arms", args.arms],
                check=False,
            )
        return

    LOG.mkdir(parents=True, exist_ok=True)
    done = set()
    if JSONL.is_file():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ok", True):
                    done.add((r["dataset"], r["arm"], r["seed"]))

    for ds in datasets:
        for seed in seeds:
            for arm in arms:
                if (ds, arm, seed) in done:
                    print(f"SKIP done {ds} {arm} {seed}", flush=True)
                    continue
                try:
                    row = run_one(ds, arm, seed)
                    row["ok"] = True
                except Exception as e:
                    row = {"dataset": ds, "arm": arm, "seed": seed, "ok": False, "error": f"{type(e).__name__}: {e}"}
                    print(f"ERR {ds} {arm} {seed}: {row['error']}", flush=True)
                with JSONL.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
