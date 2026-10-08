"""Cross-slice Phikon probe. Does not touch GNN training.

Teacher slice provides domain ids (STAIG clusters, or ground truth for the
oracle diagnostic). A linear probe on Phikon is applied to a target slice.
673 is left as its own STAIG clustering; only gated-weak targets are replaced.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _fm(data, key: str) -> np.ndarray:
    x = np.asarray(data.obsm[key], dtype=np.float64)
    x = x.reshape(x.shape[0], -1)
    return x


def fit_fm_probe(teacher, y, key: str = "phikon_emb", pca: int = 64):
    x = _fm(teacher, key)
    steps = [("scale", StandardScaler())]
    if pca and pca < x.shape[1]:
        steps.append(("pca", PCA(n_components=pca, random_state=0)))
    steps.append(
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=0))
    )
    pipe = Pipeline(steps)
    pipe.fit(x, np.asarray(y).astype(str))
    return pipe


def predict_fm_probe(pipe, target, key: str = "phikon_emb") -> np.ndarray:
    return pipe.predict(_fm(target, key))


def spatial_smooth(pred: np.ndarray, graph, rounds: int = 2) -> np.ndarray:
    g = np.asarray(graph)
    out = np.asarray(pred).astype(str).copy()
    for _ in range(rounds):
        nxt = out.copy()
        for i in range(len(out)):
            js = np.flatnonzero(g[i] == 1)
            if js.size == 0:
                continue
            vals, counts = np.unique(out[js], return_counts=True)
            nxt[i] = vals[int(counts.argmax())]
        out = nxt
    return out


def scores(y_true, y_pred) -> dict:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    return {
        "ari": float(adjusted_rand_score(yt, yp)),
        "nmi": float(normalized_mutual_info_score(yt, yp)),
    }


def _dense_X(adata, genes):
    x = adata[:, genes].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def aligned_gene_pca(teacher, target, n_pca: int = 50):
    genes = teacher.var_names.intersection(target.var_names)
    if len(genes) < n_pca + 10:
        raise RuntimeError(f"shared genes={len(genes)} too few for pca={n_pca}")
    scaler = StandardScaler()
    xt = scaler.fit_transform(_dense_X(teacher, genes))
    xq = scaler.transform(_dense_X(target, genes))
    pca = PCA(n_components=n_pca, random_state=0)
    return pca.fit_transform(xt), pca.transform(xq), int(len(genes))


def knn_vote(teacher_x, teacher_y, query_x, k: int = 15) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    y = np.asarray(teacher_y).astype(str)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(teacher_x)
    idx = nn.kneighbors(query_x, return_distance=False)
    out = []
    for row in idx:
        vals, counts = np.unique(y[row], return_counts=True)
        out.append(vals[int(counts.argmax())])
    return np.asarray(out)


def spatial_axis_bins(data, n_clusters: int) -> np.ndarray:
    xy = np.asarray(data.obsm["spatial"], dtype=np.float64)
    axis = PCA(n_components=1, random_state=0).fit_transform(xy).ravel()
    qs = np.quantile(axis, np.linspace(0, 1, n_clusters + 1))
    qs[0] -= 1e-8
    qs[-1] += 1e-8
    return np.digitize(axis, qs[1:-1]).astype(str)
