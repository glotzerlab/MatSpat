# Datasets

## Data Sources(datset descriptions taken from paper)

### Pancreatic Dataset
Pancreatic histology samples collected at the University of Michigan Pancreatic Cancer Clinic with IRB approval. 199 biopsied tissue samples total: 56 chronic pancreatitis (CP) and 143 pancreatic ductal adenocarcinoma (PDAC). Samples underwent multiplex immunofluorescence staining to identify 5 cell types: antigen-presenting cells (APCs), cytotoxic T lymphocytes (CTLs), epithelial cells (Epi), helper T cells (Thelpers), and regulatory T cells (Tregs). DAPI nuclei staining provided 2D spatial coordinates (x, y) for each cell centroid.

Potential example dataset for using the tool: 
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
