"""Runs in a few seconds without any download: python examples/toy_example.py"""
from st_graph_audit import audit_section, spatial_knn_graph
from st_graph_audit.demo import toy_section

xy, gene, fm, labels = toy_section()
res = audit_section(spatial_knn_graph(xy, 5), xy, gene, fm, labels)

print("switch:", res.switch)
for name, h in (("ungated", res.ungated), ("gated", res.gated)):
    print(f"{name:8s} collide edges {h['n_primary']:4d}  lookalike {h['lookalike']:4d}  remote {h['remote']:4d}")
print("skip edges kept by the gene gate:", len(res.skip_gated), "of", len(res.skip_ungated))
