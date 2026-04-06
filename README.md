# MatSpat

**MAT**erial analysis of **SPAT**ially resolved histology images.

A Python package that applies physics-informed spatial analysis to single-cell histology data for binary disease classification. MatSpat treats cells as point particles and borrows techniques from materials science (freud, spatial correlation functions) to quantify how immune cells organize around tumors — then uses those spatial patterns to distinguish disease states.

## Approach

1. **Characterize** local tissue structure using freud (local density, bond order, neighbor composition)
2. **Measure** how immune–tumor spatial correlation (Lee's L) changes across a tumor density gradient, producing per-image trajectories
3. **Control** for tissue structure confounds using geographically weighted partial correlation (GWPCor)
4. **Classify** disease by fitting class-dependent Hidden Markov Models to the spatial trajectories

## Pipeline Modules

| Module | Purpose |
|--------|---------|
| `preprocessing` | Load CSV data, standardize columns, ICA-based tissue rotation |
| `density` | Gaussian KDE on tumor cells, quantile-based spatial partitioning |
| `descriptors` | freud-based per-cell features: local density, k-atic order, neighbor counts |
| `interactions` | Lee's L correlation, GWPCor partial correlation, pairwise trajectory computation |
| `classification` | Sequence assembly, Gaussian HMM training, stratified evaluation with bootstrap CIs |

## Installation

```bash
pip install -e ".[dev]"
mamba install -c conda-forge freud  # freud must be installed via conda
```

## Getting Started

See the [tutorial notebook](notebooks/tutorials/tutorial_happy_path_executed.ipynb) for a full pipeline walkthrough on built-in [example data](documents/datasets.md#example-dataset-synthetic). For a real-data example using the Schürch et al. colorectal cancer CODEX dataset, see the [real data pipeline notebook](notebooks/tutorials/real_data_pipeline.ipynb).

## Input

Any spatial point-pattern dataset with x/y coordinates, a type label, and an image/sample identifier. The primary use case is multiplexed tissue imaging (e.g., CODEX, mIF) but the methods generalize to any labeled 2D point cloud.

## Output

Per-image disease classification with confidence intervals, confusion matrices, and per-class log-likelihoods. Intermediate outputs (trajectories, descriptors, sequences) are all accessible as DataFrames for custom downstream analysis.
