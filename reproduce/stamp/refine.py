"""Post-hoc domain refine: do not touch GNN training.

Spots whose spatial neighbors mostly disagree with their mclust label are
uncertain. Relabel those by majority vote of Phikon-nearest confident spots.
Strong slices should have few uncertain spots, so 151673 stays near P0.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .edges import _fm_matrix


def spatial_agree(data) -> np.ndarray:
    graph = np.asarray(data.obsm["graph_neigh"])
    domain = np.asarray(data.obs["domain"].astype(str))
    n = domain.shape[0]
    agree = np.ones(n, dtype=np.float64)
    for i in range(n):
        js = np.flatnonzero(graph[i] == 1)
        if js.size:
            agree[i] = float((domain[js] == domain[i]).mean())
    return agree


def refine_domains_with_fm(
    data,
    key: str = "phikon_emb",
    min_agree: float = 0.5,
    k: int = 10,
    pca_dim: int = 16,
) -> dict:
    if "domain" not in data.obs:
        raise RuntimeError("cluster() must run before refine")
    domain = np.asarray(data.obs["domain"].astype(str))
    agree = spatial_agree(data)
    uncertain = agree < min_agree
    n_unc = int(uncertain.sum())
    out = {
        "n": int(len(domain)),
        "n_uncertain": n_unc,
        "mean_agree": float(agree.mean()),
        "n_changed": 0,
        "min_agree": min_agree,
        "refine_k": k,
    }
    print(
        f"refine: mean_spatial_agree={agree.mean():.3f} uncertain={n_unc}/{len(domain)} "
        f"(agree<{min_agree})",
        flush=True,
    )
    if n_unc == 0 or n_unc == len(domain):
        return out
    x = _fm_matrix(data, key, pca_dim)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    confident = np.flatnonzero(~uncertain)
    new = domain.copy()
    kk = min(k, len(confident))
    changed = 0
    for i in np.flatnonzero(uncertain):
        nn = confident[np.argpartition(-sim[i, confident], kth=kk - 1)[:kk]]
        vals, counts = np.unique(domain[nn], return_counts=True)
        vote = vals[int(counts.argmax())]
        if vote != new[i]:
            new[i] = vote
            changed += 1
    data.obs["domain_raw"] = domain
    data.obs["domain"] = new
    out["n_changed"] = int(changed)
    if "ground_truth" in data.obs:
        y = data.obs["ground_truth"].astype(str)
        out["ari_before"] = float(adjusted_rand_score(y, domain))
        out["nmi_before"] = float(normalized_mutual_info_score(y, domain))
        out["ari_after"] = float(adjusted_rand_score(y, new))
        out["nmi_after"] = float(normalized_mutual_info_score(y, new))
        data.uns["ari"] = out["ari_after"]
        data.uns["nmi"] = out["nmi_after"]
        print(
            f"refine: changed={changed} ARI {out['ari_before']:.4f} -> {out['ari_after']:.4f} "
            f"NMI {out['nmi_before']:.4f} -> {out['nmi_after']:.4f}",
            flush=True,
        )
    else:
        print(f"refine: changed={changed}", flush=True)
    return out
