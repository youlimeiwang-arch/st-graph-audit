# Reproducing the paper

This folder is the research code that produced the paper's tables, copied byte for byte from the project
(checksums in `SHA256SUMS.txt`, computed with LF line endings), except for six anonymized lines that change
no computation: a local folder path in `scripts/phase2_coord_diag/run_phase2_fullres_reextract.py` (`SNAP`, used
only to compare with an earlier code snapshot) was replaced by a placeholder, and a person's name in five comments
or display strings was replaced by "PI" (`config/st_graph_audit_slides.yaml` twice,
`scripts/audit/run_local_extension_pipeline.py`, `scripts/phase2_reply10/build_phase2_step1_multiseed_table.py`,
and the report text in `scripts/prereg/build_prereg_summary.py`). Because of that last edit, this copy of
`build_prereg_summary.py` no longer matches the SHA-256 recorded before the pre-registered runs; the other eight
pre-registered scripts and `scripts/fm_complete/run_phikonv2_fill.py` match it. One further line differs from the
project: in `scripts/review_fix/run_xenium_janesick_audit.py`, `PKG` (the location of the `st_graph_audit`
package) points to the repository root instead of the project's sibling folder, so the stored
`results/xenium_relabel/stageA_janesick.json` records the SHA-256 of the project copy. The code is kept as it ran, including in-memory
redirects of embedding and label files, so the scripts are not tidy library code. To use the method on your
own data, use the `st_graph_audit` package in the repository root instead.

The scripts treat this folder as the project root: they read `Dataset/` and write new files under `logs/` and
`lab/static/tables/` here. Outputs are never overwritten; training scripts only list their jobs unless run
with `--run`.

## Requirements

Edge-level audit: the packages of the main README. Training additionally needs PyTorch with CUDA,
PyTorch Geometric, R with the `mclust` package, and `rpy2`. The final runs used Python 3.13,
PyTorch 2.7.1 (CUDA 11.8), PyTorch Geometric 2.8.0, R 4.6, rpy2 3.6.7, scanpy 1.12.1, POT 0.9.7.
Patch embedding needs the FM weights (Phikon-v2, Phikon, PLIP, UNI, CONCH) from their providers; UNI and
CONCH require accepting their licenses on Hugging Face.

## Data

Place each section under `Dataset/visium/<folder>/` or `Dataset/DLPFC/<id>/` as a Space Ranger output
folder (`filtered_feature_bc_matrix.h5`, `spatial/`) plus a label file `truth.txt`
(barcode, tab, label; no header). Sections, folders and label sources are listed in
`config/st_graph_audit_slides.yaml`; per-section STAIG hyperparameters are in `train_img_config.yaml`.
Data are not redistributed here; download links are given in Supplementary Table S1 of the paper.

| Section | Folder | Tissue | Primary edges |
|---|---|---|---|
| Visium Block A | `visium/Human_Breast_Cancer` | breast | IDC/Invasive–DCIS |
| Visium CytAssist FFPE | `visium/CytAssist_FFPE_Human_Breast_Cancer` | breast | IDC/Invasive–DCIS |
| Wu CID44971 | `visium/Wu_CID44971` | breast | Invasive–DCIS |
| Hartman A1 | `visium/Hartman_BCSA_A1` (labels `truth_filled.txt`) | breast | Invasive–DCIS |
| Xenium Rep1, 100 µm bins | `visium/Xenium_Rep1_IDC` (built by `scripts/data_prep/pack_xenium_rep1_phase1a.py`; corrected labels `truth_janesick.txt`, see below) | breast | IDC/Invasive–DCIS |
| Valdeolivas SN048_A121573 Rep1, Rep2 | `visium/CRC_Valdeolivas`, `visium/CRC_Valdeolivas_Rep2` | colorectal | Tumor–Stroma |
| Kidney DKD1 | `visium/Kidney_DKD1` | kidney | Proximal–Distal |
| Liver Yu L5 | `visium/Liver_Yu_L5` | liver | Zone_PP–Zone_PC |
| HPR normal prostate | `visium/HPR_Normal_Prostate` | prostate | none (n = 0) |
| DLPFC 151507, 151673 | `DLPFC/151507`, `DLPFC/151673` | cortex | every cross-layer edge |

## Pipeline

1. **Patches and Phikon-v2 embeddings** (about 288 µm per patch, resized to 512 × 512):
   - `scripts/xyfix/run_local_extension_pipeline_xyfix.py` (most sections; `phikonv2_xy_emb.npy`);
   - `scripts/phase2_coord_diag/run_phase2_swap_reextract.py` (Block A and CytAssist, full resolution);
   - `scripts/review_fix/reextract_fix2.py` (Xenium, Hartman, both Valdeolivas sections; `phikonv2_fix2_emb.npy`).
2. **Other FMs on the same patches:** `scripts/prereg/extract_fm_prereg.py --verify`, then
   `--encoder phikon` (and `plip`, `uni`, `conch`) → `<fm>_prereg_emb.npy`.
3. **Edge-level audit and switch** (no training):
   `python scripts/prereg/run_part3_stageA_audit.py --fm phikonv2` (and the other four FMs).
   Stored outputs are in `results/part3_stageA/`; `check_package_against_stageA.py` checks that the
   `st_graph_audit` package gives the same numbers from the raw inputs.
4. **Clustering, Phikon-v2, 10 seeds** (Step 1): `scripts/review_fix/run_phase2_step1_multiseed_fix2.py` and
   `scripts/review_fix/run_phase2_step1_switch_fix2.py` (they reuse `scripts/phase2_reply10/` and `scripts/xyfix/`);
   DLPFC through `scripts/audit/run_dlpfc_seed_stability.py`.
5. **Pre-registered experiments** (plan and SHA-256 fixed before any run; `scripts/prereg/`):
   ```
   python scripts/prereg/run_part1_local_endpoint.py --run      # affected-spot accuracy
   python scripts/prereg/run_part2_injection.py --run           # controlled injection, 260 runs
   python scripts/prereg/run_part3_stageB_ari.py --fm phikon --run   # and plip / uni / conch
   python scripts/prereg/build_prereg_summary.py
   ```
6. **Five-FM comparison table:** `python scripts/fm_complete/run_phikonv2_fill.py --run`, then
   `python scripts/fm_complete/build_fm_comparison_table.py`.

7. **Xenium Rep1 with corrected labels.** The pathologist file `idc_pathologist.csv` (Zenodo 15411357) that
   `pack_xenium_rep1_phase1a.py` joins on `cell_id` annotates a different Xenium breast dataset, so the
   original `truth.txt` labels do not describe the Rep1 cells. The bins, coordinates and embeddings are kept;
   only the labels are replaced by the majority of the supervised Rep1 cell types of Janesick et al. 2023
   (`Cell_Barcode_Type_Matrices.xlsx`, sheet "Xenium R1 Fig1-5 (supervised)"; unlabeled cells do not vote).
   Place the Rep1 `cell_feature_matrix.h5`, `cells.csv.gz`, `idc_pathologist.csv` and the xlsx under
   `Dataset/xenium/_downloads/`, then:
   ```
   python scripts/data_prep/relabel_xenium_rep1_janesick.py     # truth_janesick.txt; first checks it rebuilds truth.txt
   python scripts/review_fix/run_xenium_janesick_audit.py       # stage A, old labels must match results/part3_stageA
   python scripts/data_prep/xenium_janesick_train_inputs.py     # label file and FM rows for training
   python scripts/review_fix/run_xenium_janesick_stageB.py --run   # one old-label sanity run, then 100 runs
   ```
   Stored outputs, the bin labels (`truth_janesick.txt`, 3,975 of 4,006 bins) and a summary table are in
   `results/xenium_relabel/`.

Seeds: 0, 1, 2, 3, 7, 42, 99, 123, 2023, 39788. ARI values depend on GPU, CUDA and library versions;
the edge-level audit is deterministic.

`scripts/maintenance/notify.py` sends optional phone notifications from long queues; it reads its settings
from `~/.staig_notify.json` and has no effect on any result.
