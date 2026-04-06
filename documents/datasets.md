# Datasets

## Data Sources

### Real dataset for answering a biological question using the tool

**Colorectal Cancer CODEX Dataset** — Schürch et al., *Cell*, 2020

> Schürch CM, Bhate SS, Barlow GL, Phillips DJ, Noti L, Zlobec I, Chu P, Black S, Demeter J, McIlwain DR, Kinoshita S, Samusik N, Goltsev Y, Nolan GP. Coordinated Cellular Neighborhoods Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front. *Cell*. 2020 Sep 3;182(5):1341-1359.e19. doi: [10.1016/j.cell.2020.07.005](https://doi.org/10.1016/j.cell.2020.07.005). PMID: 32763154; PMCID: PMC7479520.

**Dataset URL:** Raw data (FCS and TIFF files) are publicly available at [https://data.mendeley.com/datasets/mpjzbtfgfr/1](https://data.mendeley.com/datasets/mpjzbtfgfr/1). The processed cell-level CSV (cell type labels + 2D coordinates per sample) used here was derived from the paper's published supplementary data.

**Description:** Colorectal cancer histology samples collected by Schürch et al. with approval from the Canton of Bern Ethics Committee and Stanford IRB. CODEX (CO-Detection by indEXing) multiplex immunofluorescence imaging produced 140 samples total: 72 Diffuse Inflammatory Infiltration (DII) and 68 Crohn's-like lymphoid reaction (CLR). Contains 16 cell type labels — B cells, CD163+ macrophages, CD4+ T cells, CD68+ macrophages, CD8+ T cells, NK cells, Tregs, adipocytes, generic immune, granulocytes, memory CD4+ T cells, plasma cells, smooth muscle cells, stromal cells, tumor cells, and vasculature — with 2D spatial coordinates (x, y) for each cell centroid.

**Why this is a good dataset for this tool:** MatSpat is designed to quantify immune cell spatial organization around tumors and classify disease states from those spatial patterns. This dataset is ideal because (1) it provides precise per-cell coordinates from a high-plex imaging platform, (2) it contains two biologically distinct immune infiltration subtypes (CLR vs. DII) that differ in *how* immune cells are organized rather than simply *how many* are present, making spatial context essential for classification, and (3) it is large enough (140 samples) to train and evaluate an HMM classifier with meaningful statistical power while remaining computationally tractable.

**Biological question:** Do CD163+ macrophages (a marker of M2-polarized, potentially immunosuppressive macrophages) exhibit distinct spatial co-localization patterns with tumor cells in CLR versus DII colorectal cancer, and can these spatial trajectories alone discriminate between the two subtypes?

**Expected results and answer:** CLR is associated with a more organized, adaptive immune response and better prognosis, while DII features a diffuse, less structured infiltration. We expect CD163+ macrophages in CLR tumors to show a characteristic spatial trajectory — higher co-localization with tumor cells at specific quantile distances — compared to DII, where the relationship is flatter or inversely structured. The HMM classifier trained on these tumor × CD163+ spatial trajectories (via Lee's L or GWPCor) should achieve above-chance discrimination between CLR and DII (AUROC > 0.6), providing evidence that macrophage spatial positioning around the tumor is a subtype-informative feature beyond simple cell-count statistics. GWPCor is expected to outperform raw Lee's L by removing the confounding effect of globally dense tissue regions.

---

## Data Format and Structure

**File Format:** CSV files
**Required Fields:**
- Spatial coordinates (x, y)
- Cell type label
- Image/sample identifier

**Optional Fields:**
- Patient identifier
- Disease type/pathology label
- Cell identifier

**Dataset Size:**
- Colorectal: 140 samples, ~400 cells per sample on average, more gaussian distribution due to CODEX processing.

## Example Dataset (Synthetic)

MatSpat includes a synthetic data generator (`matspat.sample_data.generate_sample_dataset`) for tutorials and quick demos. It produces 10 tissue images (5 per class) with two disease classes:

- **"inflamed"**: Immune cells cluster near the tumor core.
- **"excluded"**: Immune cells form a ring around the tumor, away from the core.

Each image has ~120 tumor cells and ~25 each of CD4, CD8, and Macrophage (jittered ±20%), giving a 4-5x tumor-to-immune ratio. Cells are placed in a 500×500 coordinate field. Columns: `X`, `Y`, `cell_type`, `image_id`, `pathology`.

## Development Strategy

**The full colorectal dataset is used during development** as the data size is manageable for computational resources. No representative subset or simulated data is required. Final validation utilizes the complete dataset to ensure the package handles real-world multi-sample histology data effectively.
