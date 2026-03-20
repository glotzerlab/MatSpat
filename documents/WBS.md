# Work Breakdown Structure (WBS)

## Activity 1: Project Setup & Data Loader

**Task 1:** Set up repository, documentation, and dev tools
- Deliverable: SRS.md, datasets.md, WBS.md, pyproject.toml configured
- Done when: All docs created, dependencies installed, pre-commit hooks working

**Task 2:** Implement data loader with column mapping and validation
- Deliverable: `load_spatial_data()` function in `matspat.preprocessing`
- Done when: Handles arbitrary column names, validates required fields, auto-generates IDs

**Task 3:** Create tests and validate with real datasets
- Deliverable: Unit tests with CSV fixtures, tests pass on pancreatic and colorectal data(basic tests like checking if columns exist or not, or checking if there are all uniqIDs for cells)
- Done when: All tests pass, no validation errors

---

## Activity 2: Freud Spatial Descriptors

**Task 1:** Implement local density computation
- Deliverable: Function computing local density for cells/cell types
- Done when: Returns density array, handles variable radii, supports cell type filtering

**Task 2:** Implement k-atic bond orientational order
- Deliverable: Function computing hexatic (k=6) and tetratic (k=4) order parameters
- Done when: Returns magnitude and phase for each cell, tested on sythentically generated symmetric configurations 

**Task 3:** Implement Voronoi tessellation
- Deliverable: Function returning neighbor lists and cell areas
- Done when: Correctly identifies neighbors, computes areas, handles boundary cells

---

## Activity 3: Correlation Function

**Task 1:** Implement Lee's L spatial correlation
- Deliverable: Function computing Lee's L with bisquare kernel weighting
- Done when: Returns correlation coefficients between -1 and 1, bandwidth parameter works

**Task 2:** Implement partial correlation sequences
- Deliverable: Function generating GWPCor and Partial Distance Correltions
- Done when: Correlations between -1 to 1, works on simulated data with all sorts of associations
---

## Activity 4: KDE and Quantilization, Markov Sequence Modeling

**Task 1:** Implement grid-based KDE computation
- Deliverable: Function computing density grid for cell types
- Done when: Builds spatial grid, computes KDE with bandwidth, density array returned

**Task 2:** Implement marginal quantile binning
- Deliverable: Function returning boolean masks for equal-mass bins
- Done when: Divides density into n bins, bins don't overlap, equal mass verified

**Task 3:** Implement cumulative quantile binning
- Deliverable: Function returning cumulative density masks
- Done when: Returns masks up to each quantile, handles edge cases (0th/100th percentile)

**Task 4:** Implement Sequence Models
- Deliverable: runnable HMM wrapper on sequences with different parameters for various masks
- Done when: Returns a validated test/train accruacy on the dataset
---

## Activity 5: Tutorial, Demo, and Report Writing
**Task 1: Tutorial Development**
- Deliverable: jupyter notebook that runs on sample dataset, with examples of different pipelines.
- Done when: pipeline executes succesfully with various parameter changes
**Task 2: Demo and Report Writing**
- Deliverable: Demonstration slides and final report with details of each module
- Done when: in-class demonstration complete, with reviewed final report submitted.
