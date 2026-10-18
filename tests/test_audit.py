"""python -m unittest discover tests   (also runs under pytest)"""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np

from st_graph_audit import (
    add_skip_edges,
    audit_section,
    coarse_label,
    edge_pairs,
    parse_pairs,
    spatial_knn_graph,
    switch_on,
)
from st_graph_audit.cli import main as cli_main
from st_graph_audit.demo import toy_section


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.xy, self.gene, self.fm, self.labels = toy_section()
        self.g0 = spatial_knn_graph(self.xy, 5)

    def test_spatial_graph_symmetric(self):
        self.assertTrue(np.array_equal(self.g0, self.g0.T))
        self.assertEqual(np.trace(self.g0), 0)
        self.assertTrue((self.g0.sum(1) >= 5).all())

    def test_spatial_graph_excludes_self_with_duplicate_coordinates(self):
        xy = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        graph = spatial_knn_graph(xy, k=1)
        self.assertTrue(np.array_equal(np.diag(graph), np.zeros(3)))
        self.assertTrue(np.array_equal(graph, graph.T))

    def test_skip_edges_are_new_and_kept_in_graph(self):
        g, pairs = add_skip_edges(self.g0, self.fm, k=1)
        self.assertTrue(pairs)
        for i, j in pairs:
            self.assertEqual(self.g0[i, j], 0)
            self.assertEqual(g[i, j], 1)
            self.assertEqual(g[j, i], 1)
        self.assertEqual(len(edge_pairs(g)), len(edge_pairs(self.g0)) + len(pairs))

    def test_gated_skip_edges_stay_within_class(self):
        _, gated = add_skip_edges(self.g0, self.fm, self.gene, k=1, gene_k=10)
        self.assertTrue(gated)
        same = [self.labels[i] == self.labels[j] for i, j in gated]
        self.assertGreater(np.mean(same), 0.95)


class TestAudit(unittest.TestCase):
    def test_toy_section(self):
        xy, gene, fm, labels = toy_section()
        res = audit_section(spatial_knn_graph(xy, 5), xy, gene, fm, labels)
        self.assertEqual(res.switch, "ON")
        self.assertGreater(res.ungated["lookalike"], 50)
        self.assertGreater(res.ungated["remote"], 50)
        self.assertLess(res.gated["lookalike"], res.ungated["lookalike"] / 10)
        self.assertLess(res.gated["remote"], res.ungated["remote"] / 10)

    def test_cross_primary_has_no_switch(self):
        xy, gene, fm, labels = toy_section()
        res = audit_section(spatial_knn_graph(xy, 5), xy, gene, fm, labels, primary="cross")
        self.assertEqual(res.switch, "N/A")

    def test_switch_rule(self):
        self.assertTrue(switch_on(10, 3.0, 0.20))
        self.assertFalse(switch_on(9, 3.0, 0.20))
        self.assertFalse(switch_on(10, 2.9, 0.20))
        self.assertFalse(switch_on(10, 3.0, 0.21))
        self.assertFalse(switch_on(10, None, None))

    def test_labels(self):
        self.assertEqual(coarse_label("IDC_3"), "IDC")
        self.assertEqual(coarse_label("Tumor_edge_1"), "Edge")
        self.assertEqual(parse_pairs(["A:B"]), frozenset({frozenset({"A", "B"})}))
        with self.assertRaises(ValueError):
            parse_pairs(["A"])


class TestCLI(unittest.TestCase):
    def test_refuses_to_overwrite_partial_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "out"
            out.mkdir()
            partial = out / "edges_gated.csv"
            partial.write_text("keep me", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                status = cli_main([
                    "h5ad", "missing.h5ad", "--fm-key", "fm", "--label-key", "label", "--out", str(out)
                ])
            self.assertEqual(status, 1)
            self.assertIn("edges_gated.csv", stderr.getvalue())
            self.assertEqual(partial.read_text(encoding="utf-8"), "keep me")


if __name__ == "__main__":
    unittest.main()
