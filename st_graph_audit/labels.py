"""Coarse tissue labels and the label pairs whose edges count as collisions."""
from __future__ import annotations

from typing import Iterable

import numpy as np

# Unordered label pairs that are biologically distinct but can look alike on H&E.
DEFAULT_COLLIDE_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(p)
    for p in (
        ("IDC", "DCIS"),
        ("Invasive", "DCIS"),
        ("Tumor", "Stroma"),
        ("Zone_PP", "Zone_PC"),
        ("Periportal", "Pericentral"),
        ("Proximal", "Distal"),
        ("PT", "DT"),
        ("Tubule", "Glomerulus"),
        ("LH", "CD"),
    )
)

_PREFIXES = (
    ("IDC", "IDC"),
    ("DCIS", "DCIS"),
    ("Healthy", "Healthy"),
    ("Tumor_edge", "Edge"),
    ("Invasive", "Invasive"),
    ("Lymphocytes", "Immune"),
    ("Stroma", "Stroma"),
    ("Normal", "Normal"),
)


def coarse_label(label) -> str:
    """Collapse annotation variants (e.g. ``IDC_3``, ``DCIS/LCIS_1``, ``Stroma_high``) to one coarse class."""
    s = str(label)
    for prefix, name in _PREFIXES:
        if s.startswith(prefix):
            return name
    return s.split()[0]


def coarse_labels(labels: Iterable) -> np.ndarray:
    return np.array([coarse_label(x) for x in labels])


def parse_pairs(specs: Iterable[str]) -> frozenset[frozenset[str]]:
    """``["IDC:DCIS", "Tumor:Stroma"]`` -> collide pair set."""
    out = set()
    for s in specs:
        a, sep, b = s.partition(":")
        if not sep or not a or not b or a == b:
            raise ValueError(f"collide pair must look like LABEL_A:LABEL_B, got {s!r}")
        out.add(frozenset({a, b}))
    return frozenset(out)


def relation(ci: np.ndarray, cj: np.ndarray, collide_pairs=DEFAULT_COLLIDE_PAIRS) -> np.ndarray:
    """Per edge: ``same`` (equal coarse labels), ``collide`` (a collide pair) or ``other_cross``."""
    rel = np.full(len(ci), "other_cross", dtype=object)
    for k, (a, b) in enumerate(zip(ci, cj)):
        if a == b:
            rel[k] = "same"
        elif frozenset({a, b}) in collide_pairs:
            rel[k] = "collide"
    return rel
