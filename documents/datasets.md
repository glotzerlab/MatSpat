# Datasets

## Data Sources(dataset descriptions taken from paper)

Pancreatic Dataset not available for public usage yet. 
### Pancreatic Dataset
Pancreatic histology samples collected at the University of Michigan Pancreatic Cancer Clinic with IRB approval. 199 biopsied tissue samples total: 56 chronic pancreatitis (CP) and 143 pancreatic ductal adenocarcinoma (PDAC). Samples underwent multiplex immunofluorescence staining to identify 5 cell types: antigen-presenting cells (APCs), cytotoxic T lymphocytes (CTLs), epithelial cells (Epi), helper T cells (Thelpers), and regulatory T cells (Tregs). DAPI nuclei staining provided 2D spatial coordinates (x, y) for each cell centroid.

Colorectal Dataset available in paper by Schürch et al.
Schürch CM, Bhate SS, Barlow GL, Phillips DJ, Noti L, Zlobec I, Chu P, Black S, Demeter J, McIlwain DR, Kinoshita S, Samusik N, Goltsev Y, Nolan GP. Coordinated Cellular Neighborhoods Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front. Cell. 2020 Sep 3;182(5):1341-1359.e19. doi: 10.1016/j.cell.2020.07.005. Epub 2020 Aug 6. Erratum in: Cell. 2020 Oct 29;183(3):838. doi: 10.1016/j.cell.2020.10.021. PMID: 32763154; PMCID: PMC7479520.

### Colorectal Dataset
Colorectal cancer histology samples collected by Schürch et al. with approval from the Canton of Bern Ethics Committee and Stanford IRB. CODEX imaging technology produced 140 samples total: 72 Diffuse Inflammatory Infiltration (DII) and 68 Crohn's-like lymphoid reaction (CLR). Dataset contains 16 cell type labels including B cells, CD163+ macros, CD4+ T cells, CD68+ macros, CD8+ T cells, NK cells, Tregs, adipocytes, generic immune, granulocytes, memory CD4+ T cells, plasma cells, smooth muscle cells, stromal cells, tumor cells, and vasculature, providing 2D spatial coordinates (x, y) for each cell centroid.

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
- Pancreatic: 199 samples, ~500-1500cells per sample
- Colorectal: 140 samples, similar cell density per sample, more gaussian distribution than pancreatic due to processing techniques.

## Development Strategy

**Full datasets will be used during development** as the data size is manageable for computational resources. No representative subset or simulated data is required. Final validation will utilize the complete datasets to ensure the package handles real-world multi-sample histology data effectively. There have been some talks about more kinds of this data being available soon, but I will focus on these two datasets for now. 
