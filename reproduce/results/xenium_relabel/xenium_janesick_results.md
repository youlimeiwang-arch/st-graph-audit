# Xenium Rep1 IDC — results with corrected labels

> **Section:** 10x Xenium human breast Rep1 (Janesick et al. 2023), aggregated into 100 µm bins (4,006 bins).
> **What changed:** labels only. Bins, coordinates, FM embeddings, graph construction, thresholds and training settings are unchanged.
> **Sources:** `logs/xenium_relabel/stageA_janesick.json`, `logs/xenium_relabel/stageB_summary.json`, `logs/xenium_relabel/stageB_janesick.jsonl`, `Dataset/visium/Xenium_Rep1_IDC/LABEL_SOURCE_janesick.json`. Old-label values are from `fm_comparison_12x5.md`.

## 1. Label sets

| | Old labels (`truth.txt`) | Corrected labels (`truth_janesick.txt`) |
|---|---|---|
| Source | pathologist IDC/DCIS file (`idc.csv`) joined on `cell_id`; the file belongs to a **different** Xenium dataset | Janesick et al. 2023, `Cell_Barcode_Type_Matrices.xlsx`, sheet "Xenium R1 Fig1-5 (supervised)" |
| Bin label | pathologist domain of joined cells | majority mapped cell type; unlabeled cells do not vote |
| Labeled bins | 4,006 | 3,975 (31 bins have no labeled cell) |
| Classes | 6 (IDC, DCIS, Stroma, Immune, Normal_duct, Adipose) | 6 (Stroma 2,011; IDC 841; DCIS 493; Immune 489; Myoepi 98; Vessel 43) |
| Agreement with the other label set | — | 25.0 % of bins |

Cell-type mapping: Invasive_Tumor and Prolif_Invasive_Tumor → IDC; DCIS_1 and DCIS_2 → DCIS; Stromal and Perivascular-Like → Stroma; Endothelial → Vessel; both Myoepi types → Myoepi; T, B, macrophage, DC and mast cells → Immune; the two hybrid types → Hybrid, which is not the majority in any bin.

## 2. Stage A — edge audit (primary collide pair IDC–DCIS; fixed τ_fm = 0.80)

| FM | Labels | Switch | n_collide | nn_mult | gene_cos | Lookalike u → g | Remote u → g |
|---|---|---|---:|---:|---:|---|---|
| Phikon-v2 | old | OFF | 627 | 1.59 | 0.324 | 143 → 24 | 266 → 41 |
| Phikon-v2 | corrected | OFF | 103 | 1.55 | 0.536 | 9 → 0 | 34 → 8 |
| Phikon | old | OFF | 650 | 1.63 | 0.315 | 172 → 27 | 283 → 40 |
| Phikon | corrected | OFF | 98 | 1.52 | 0.552 | 10 → 1 | 28 → 8 |
| PLIP | old | OFF | 660 | 1.72 | 0.289 | 157 → 17 | 311 → 36 |
| PLIP | corrected | **ON** | 191 | 17.22 | 0.154 | 69 → 1 | 124 → 6 |
| UNI | old | OFF | 626 | 1.59 | 0.346 | 126 → 20 | 276 → 57 |
| UNI | corrected | OFF | 98 | 1.52 | 0.545 | 8 → 1 | 25 → 9 |
| CONCH | old | OFF | 609 | 1.58 | 0.337 | 103 → 8 | 261 → 38 |
| CONCH | corrected | OFF | 106 | 1.56 | 0.428 | 18 → 0 | 40 → 6 |

u = ungated, g = gated. Recomputing the old-label rows with the same script reproduced the stored stage A values for all five FMs.

## 3. Stage B — ARI, ungated vs gated (10 seeds per FM)

| FM | Labels | Mean ARI ungated | Mean ARI gated | Median Δ ARI | Seeds up / down | Wilcoxon p | Direction |
|---|---|---:|---:|---:|---|---:|---|
| Phikon-v2 | old | −0.001 | 0.001 | +0.001 | 6 / 4 | — | at floor |
| Phikon-v2 | corrected | 0.264 | 0.260 | −0.013 | 3 / 7 | 0.432 | no stable change |
| Phikon | old | −0.001 | −0.001 | +0.001 | 8 / 2 | — | at floor |
| Phikon | corrected | 0.259 | 0.256 | −0.011 | 4 / 6 | 0.695 | no stable change |
| PLIP | old | 0.001 | 0.000 | −0.002 | 4 / 6 | — | at floor |
| PLIP | corrected | 0.262 | 0.259 | −0.006 | 5 / 5 | 0.770 | no stable change |
| UNI | old | −0.000 | −0.001 | +0.001 | 6 / 4 | — | at floor |
| UNI | corrected | 0.278 | 0.250 | −0.028 | 3 / 7 | 0.160 | no stable change |
| CONCH | old | −0.000 | −0.002 | −0.001 | 3 / 7 | — | at floor |
| CONCH | corrected | 0.272 | 0.251 | −0.010 | 4 / 6 | 0.375 | no stable change |

Seeds 0, 1, 2, 3, 7, 42, 99, 123, 2023, 39788. STAIG settings as in Supplementary Table S1 (k = 80, 400 epochs, 300 genes, 6 clusters). Direction rule as in the pre-registered package: "up" or "down" requires ≥ 8 of 10 seeds with the same sign. With old labels every arm was below the ARI floor, so direction was not interpretable.

## 4. Effect on the pre-registered decision

- The corrected labels add one switch-ON combination (Xenium, PLIP). Its median Δ ARI is −0.006 (5 up, 5 down), so it is not an "up" case.
- The other four Xenium combinations stay OFF.
- The primary Part 3 test has **not** been recomputed with the corrected labels. Even so, adding a non-positive ON case cannot turn p = 1.000 into a positive result, so the Benefit criterion stays unmet.
