"""Synthetic section for the example and tests: two classes that look alike on H&E but differ in expression."""
from __future__ import annotations

import numpy as np


def toy_section(side: int = 20, seed: int = 0):
    """Grid of side x side spots; left half IDC, right half DCIS.

    Expression: each class switches on its own 10 of 20 genes. FM embedding (8-d): four appearance
    prototypes drawn at random in both classes, so FM-nearest spots often sit in the other class, far away.
    Returns xy, gene, fm, labels.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side]
    xy = np.c_[xx.ravel(), yy.ravel()].astype(float)
    labels = np.where(xy[:, 0] < side / 2, "IDC", "DCIS")
    n = len(labels)
    gene = rng.poisson(1.0, size=(n, 20)).astype(float)
    gene[labels == "IDC", :10] += rng.poisson(8.0, size=((labels == "IDC").sum(), 10))
    gene[labels == "DCIS", 10:] += rng.poisson(8.0, size=((labels == "DCIS").sum(), 10))
    protos = rng.normal(size=(4, 8))
    fm = protos[rng.integers(0, 4, size=n)] + 0.1 * rng.normal(size=(n, 8))
    return xy, gene, fm, labels
