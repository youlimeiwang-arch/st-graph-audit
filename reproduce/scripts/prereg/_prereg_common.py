"""Shared pieces for the pre-registered package (plan: lab/static/replies/reply12_response_v2_EN.md).

Training goes through the same runners as the Step 1 / review-fix runs: scripts/review_fix/_fix2_env.py (in-memory
Phikon-v2 and Hartman label redirects), the run_phase2_step1_multiseed recipes, and run_dlpfc_seed_stability for DLPFC.
The parts only wrap add_fm_skip_edges (in the runner's namespace) and STAMP.cluster; no recipe, seed rule or
threshold is changed. Import this module before anything reads a slide. STAIG_FIX2_FM must be set before import
when a different FM file is wanted (Part 3).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "review_fix"))
import _fix2_env  # noqa: E402

LOG = ROOT / "logs" / "prereg"
TAB = ROOT / "lab" / "static" / "tables" / "prereg"
SEEDS10 = [0, 1, 2, 3, 7, 42, 99, 123, 2023, 39788]
FM_KEY = "phikonv2_emb"
UNGATED_KW = dict(k=1, drop_prob=0.05, pca_dim=16, gene_key="", gene_k=0, gene_mutual=False)
GATED_KW = dict(k=1, drop_prob=0.05, pca_dim=16, gene_key="feat", gene_k=10, gene_mutual=True)

# Step 1 datasets: id -> (Dataset subfolder, slide folder, patch folder behind the Phikon-v2 file used in Step 1)
SLIDES = {
    "BlockA": ("visium", "Human_Breast_Cancer", "clip_image_swap_fullres"),
    "CytAssist_Patho": ("visium", "CytAssist_FFPE_Human_Breast_Cancer", "clip_image_swap_fullres"),
    "Wu": ("visium", "Wu_CID44971", "clip_image_xy"),
    "HPR_Prostate": ("visium", "HPR_Normal_Prostate", "clip_image_xy"),
    "Kidney_DKD1": ("visium", "Kidney_DKD1", "clip_image_xy"),
    "Liver_Yu_L5": ("visium", "Liver_Yu_L5", "clip_image_xy"),
    "Xenium_Rep1_IDC": ("visium", "Xenium_Rep1_IDC", "clip_image_fix2"),
    "Hartman_BCSA_A1": ("visium", "Hartman_BCSA_A1", "clip_image_fix2"),
    "CRC_Valdeolivas": ("visium", "CRC_Valdeolivas", "clip_image_fix2"),
    "CRC_Valdeolivas_Rep2": ("visium", "CRC_Valdeolivas_Rep2", "clip_image_fix2"),
    "DLPFC151507": ("DLPFC", "151507", "clip_image_xy"),
    "DLPFC151673": ("DLPFC", "151673", "clip_image_xy"),
}
DLPFC = ("DLPFC151507", "DLPFC151673")


def slide_dir(ds: str) -> Path:
    sub, slide, _ = SLIDES[ds]
    return ROOT / "Dataset" / sub / slide


def primary_relation(ds: str) -> str:
    return "cross" if ds in DLPFC else "idc_dcis"


def fm_file(fm: str) -> str:
    return "phikonv2_fix2_emb.npy" if fm == "phikonv2" else f"{fm}_prereg_emb.npy"


def expected_fm(ds: str) -> str:
    """File that phikonv2_emb.npy must be redirected to for this slide in this process."""
    want = _fix2_env.FIX2_EMB
    if (slide_dir(ds) / want).is_file():
        return want
    if want != "phikonv2_fix2_emb.npy":
        raise SystemExit(f"{ds}: {want} missing in {slide_dir(ds)}")
    return "phikonv2_xy_emb.npy"


def check_redirect(ds: str) -> dict[str, str]:
    slide = SLIDES[ds][1]
    used = {Path(k).name: Path(v).name for k, v in _fix2_env.redirects().items() if Path(k).parent.name == slide}
    want = expected_fm(ds)
    if used.get("phikonv2_emb.npy") != want:
        raise RuntimeError(f"{ds}: phikonv2_emb.npy redirected to {used.get('phikonv2_emb.npy')}, expected {want}")
    return used


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Runner:
    """train(arm, seed) -> row, exactly as in run_mclust_seed_randomness; owner = module whose add_fm_skip_edges runs."""

    def __init__(self, ds: str):
        self.ds = ds
        if ds in DLPFC:
            dl = load_module("dlpfc_orig", ROOT / "scripts/audit/run_dlpfc_seed_stability.py")
            job = next(j for j in dl.JOBS if j["id"] == ds)
            self.owner = dl

            def train(arm: str, seed: int) -> dict:
                import time
                t0 = time.time()
                ari = dl.run_one(job, seed, 10 if arm == "gated" else 0)
                return {"dataset": ds, "slide": job["slide"], "arm": arm, "seed": seed, "ari": ari,
                        "sec": round(time.time() - t0, 1)}
        else:
            mf = load_module("ms_fix2", ROOT / "scripts/review_fix/run_phase2_step1_multiseed_fix2.py")
            self.owner = mf.ms
            base = mf.run_one if ds in _fix2_env.FIX2_SLIDES else mf._run_one

            def train(arm: str, seed: int) -> dict:
                row = base(ds, arm, seed)
                row.pop("recipe", None)
                return row
        self._train = train

    def train(self, arm: str, seed: int) -> dict:
        if arm not in ("ungated", "gated"):
            raise ValueError(arm)
        row = self._train(arm, seed)
        row["fm_redirects"] = check_redirect(self.ds)
        return row


SKIP_HOOK = None  # callable(data, key, orig, kw) -> None, replaces the runner's add_fm_skip_edges call when set


def install_skip_hook(owner) -> None:
    orig = owner.add_fm_skip_edges
    if getattr(orig, "_prereg", False):
        return

    def wrapped(data, key, **kw):
        if SKIP_HOOK is None:
            return orig(data, key, **kw)
        return SKIP_HOOK(data, key, orig, kw)

    wrapped._prereg = True
    owner.add_fm_skip_edges = wrapped


CAPTURED: dict = {}


def install_cluster_capture() -> None:
    from stamp.model import STAMP

    orig = STAMP.cluster
    if getattr(orig, "_prereg", False):
        return

    def cluster(self, label=True):
        orig(self, label)
        CAPTURED.clear()
        CAPTURED.update(obs_names=np.asarray(self.adata.obs_names).astype(str),
                        pred=self.adata.obs["domain"].astype(str).to_numpy(),
                        truth=self.adata.obs["ground_truth"].astype(str).to_numpy())

    cluster._prereg = True
    STAMP.cluster = cluster


def skip_pairs(data, key: str, orig, **kw) -> set[tuple[int, int]]:
    """Undirected skip pairs add_fm_skip_edges would add to data, computed on a light copy (data is untouched)."""
    import anndata

    light = anndata.AnnData(X=np.zeros((data.n_obs, 1), dtype=np.float32))
    for k in ("graph_neigh", "edge_probabilities", key, "feat"):
        if k in data.obsm:
            light.obsm[k] = np.array(data.obsm[k], copy=True)
    before = np.asarray(light.obsm["graph_neigh"]) > 0
    orig(light, key, **kw)
    after = np.asarray(light.obsm["graph_neigh"]) > 0
    i, j = np.nonzero(np.triu(after & ~before, 1))
    return set(zip(i.tolist(), j.tolist()))


def edge_flags(data, pairs, key: str):
    """Per-edge audit table (stamp.graph_audit.core.edge_table, frozen thresholds) on the training AnnData."""
    from stamp.graph_audit.core import edge_table

    return edge_table(data, list(pairs), fm_key=key)


def hungarian_correct(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-spot correctness after a one-to-one cluster->label mapping fixed on all spots (unmatched clusters wrong)."""
    from scipy.optimize import linear_sum_assignment

    labs, ti = np.unique(truth, return_inverse=True)
    cls, pi = np.unique(pred, return_inverse=True)
    cont = np.zeros((len(cls), len(labs)), dtype=np.int64)
    np.add.at(cont, (pi, ti), 1)
    r, c = linear_sum_assignment(-cont)
    mapping = np.full(len(cls), -1)
    mapping[r] = c
    return mapping[pi] == ti


_TBL = None


def direction(deltas: list[float]) -> dict:
    """Step 1 direction rule (build_phase2_step1_multiseed_table.direction), loaded so the rule stays identical."""
    global _TBL
    if _TBL is None:
        _TBL = load_module("tbl_orig", ROOT / "scripts/phase2_reply10/build_phase2_step1_multiseed_table.py")
    return _TBL.direction(deltas)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def done_keys(path: Path, fields: tuple[str, ...]) -> set[tuple]:
    return {tuple(r[f] for f in fields) for r in read_jsonl(path) if r.get("ok")}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def new_path(path: Path) -> Path:
    """path if free, else path with _v2, _v3, ... (outputs are never overwritten)."""
    if not path.exists():
        return path
    k = 2
    while (p := path.with_name(f"{path.stem}_v{k}{path.suffix}")).exists():
        k += 1
    return p


def add_run_flag(ap) -> None:
    ap.add_argument("--run", action="store_true",
                    help="actually train; without it the script only lists the jobs and checks inputs (dry run)")


def free_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
