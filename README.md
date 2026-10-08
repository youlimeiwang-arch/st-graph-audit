# ST-Graph Audit

Edge-level audit of foundation-model (FM) skip edges in spatial transcriptomics graphs, with a gene gate that removes the inconsistent ones.

Histology-augmented spatial GNNs often add "skip" edges between spots that look alike on H&E, wherever they are on the slide. Some of those edges join spots that look alike but differ in expression (*lookalike* edges), often far apart (*remote* edges). ST-Graph Audit:

- **scores a graph**: per-edge gene cosine, FM cosine and distance in units of the median nearest-neighbor spacing, and counts lookalike and remote edges between label classes that are hard to tell apart on H&E;
- **decides a section-level switch**: ON when FM skip edges are likely to add remote lookalike edges on that section;
- **builds a gated graph**: an FM skip edge is kept only if the two spots are also mutual nearest neighbors in expression space.

It trains nothing and needs no GPU. Graph construction uses coordinates, expression and the FM embedding only; labels are used to count harmful edges and to compute the switch.

## Install

```bash
git clone https://github.com/youlimeiwang-arch/st-graph-audit.git
cd st-graph-audit
pip install -e .
```

Python ≥ 3.9. Tested with Python 3.13, NumPy 2.4, SciPy 1.17, scikit-learn 1.8, pandas 2.3, anndata 0.12, scanpy 1.12, POT 0.9.7, scikit-misc 0.5.

## Quick start

A synthetic section (two classes that look alike but differ in expression), no download needed:

```bash
python examples/toy_example.py
```

```text
switch: ON
ungated  collide edges  185  lookalike  157  remote  144
gated    collide edges   39  lookalike   11  remote    0
skip edges kept by the gene gate: 36 of 307
```

Tests: `python -m unittest discover tests`

### Command line

```bash
# Space Ranger folder + label file (barcode<TAB>label, no header) + FM embedding (.npy, one row per spot)
python -m st_graph_audit visium path/to/section --labels truth.txt --fm fm_emb.npy --out out/section

# AnnData with raw counts in X and coordinates in obsm["spatial"]
python -m st_graph_audit h5ad section.h5ad --fm-key uni --label-key domain --out out/section

# cortex: count every cross-layer edge (the switch is then not defined)
python -m st_graph_audit visium path/to/151507 --labels truth.txt --fm fm_emb.npy --primary cross --out out/151507

# your own label pairs
python -m st_graph_audit h5ad liver.h5ad --fm-key uni --label-key zone --collide Periportal:Pericentral --out out/liver
```

`python -m st_graph_audit --help` lists all options. The output folder gets `summary.json` (switch and harm counts for both graphs), `edges_ungated.csv`, `edges_gated.csv` (one row per edge) and `skip_gated.csv` (skip edges kept by the gate). Existing results are never overwritten.

### Python

```python
from st_graph_audit import audit_section, spatial_knn_graph, gene_features

gene = gene_features(adata)                      # raw counts -> STAIG expression features
graph = spatial_knn_graph(adata.obsm["spatial"], 5)
res = audit_section(graph, adata.obsm["spatial"], gene, fm, labels)
res.switch, res.ungated, res.gated               # "ON"/"OFF"/"N/A", harm counts
res.graph_gated, res.skip_gated                  # gated adjacency, kept skip edges
```

## Method

| Step | Setting |
|---|---|
| Spatial graph | 5 nearest spots, symmetrized (STAIG default) |
| Expression features | 3,000 seurat_v3 HVGs on raw counts, normalize to 10⁴, log1p, scale (max 10); then standardize, PCA 32, standardize, L2 |
| FM features | standardize, PCA 16, standardize, L2 |
| Ungated skip edge | per spot, the most FM-similar non-neighbor (k = 1) |
| Gated skip edge | per spot, the first of its 10 most FM-similar non-neighbors that is also a mutual gene-space 10-nearest non-neighbor |
| Lookalike edge | gene cosine ≤ 0.20 and FM cosine ≥ 0.80 |
| Remote edge | distance ≥ 3 × median nearest-neighbor distance |
| Switch ON | on the collide edges of the ungated graph: n ≥ 10, median distance ratio ≥ 3, median gene cosine ≤ 0.20 |

Thresholds were fixed on one section (10x Visium breast cancer, Block A) and used unchanged everywhere else.

Built-in collide pairs (after coarsening labels such as `IDC_3` → `IDC`): IDC or Invasive vs DCIS, Tumor vs Stroma, Periportal vs Pericentral (and Zone_PP vs Zone_PC), Proximal vs Distal (and PT vs DT), Tubule vs Glomerulus, LH vs CD. Use `--collide A:B` for other tissues.

Similarities are computed as dense N × N matrices, so memory grows as O(N²): about 0.2 GB per matrix at 5,000 spots.

## Reproducing the paper

`reproduce/` holds the exact research code behind every table (STAIG training, controlled injection, the five-FM comparison), copied unchanged, with SHA-256 checksums. It is not needed to use the tool. See [reproduce/README.md](reproduce/README.md).

On the 12 annotated sections × 5 FMs of the paper, this package reproduces the stored edge-level audit exactly (60/60: switch, collide-edge count, lookalike and remote counts, edge totals): `python reproduce/check_package_against_stageA.py --data-root <Dataset>`.

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). `reproduce/staig/` is a modified copy of [STAIG](https://github.com/y-itao/STAIG), also Apache-2.0.

## Citation

Manuscript in preparation.
