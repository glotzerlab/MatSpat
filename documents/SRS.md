# Software Requirements Specification (SRS)
## MatSpat - Material Analysis of Spatially Resolved Histology Images

**Version:** 0.3.0
**Date:** 2026-03-25
**Author:** Ninad Kamath
**Status:** Entering final stages of development for beta release!

---

## 1. Introduction

### 1.1 Purpose of this Document
This document specifies the functional and non-functional requirements for MatSpat, a Python package for physics-informed analysis of spatially resolved, single-cell histology images. It serves as a guide for developers, testers, and stakeholders during the initial development phase.

### 1.2 Scope
MatSpat enables researchers to analyze spatial patterns in histology data using physics-based methods from materials science. The software converts static histology slides into quantitative spatial representations, computes statistical features, and performs classification and changepoint detection.

**Value Proposition:** Bridges materials science spatial analysis techniques with histology imaging to enable novel disease classification approaches based on cellular spatial organization.

**Development Stage:** Late Development(Weeks 10-11), All Modules Implemented with thorough test suite - about 90% test coverage, so some work to be done there. In the process of implementing full happy path tutorial and also creating a demonstration and paper. 

### 1.3 Overview
MatSpat is a high-level Python package that treats cells in histology images as point particles in a spatial field, applying computational methods from materials science (via the `freud` library) to extract spatial features and patterns. The package performs radial sequence analysis from specified core regions, computes partial correlations, and uses sequence modeling (HMMs and deep learning) for disease classification.

---

## 2. General Description

### 2.1 Product Functions
MatSpat provides the following core capabilities:

1. **Spatial Data Loading & Preprocessing**
   - Universal CSV loader for various spatial transcriptomics formats
   - Standardization of heterogeneous input data
   - Coordinate validation and missing value handling

2. **Density Estimation**
   - Kernel Density Estimation (KDE) for spatial cell density
   - Quantilization of Tumor density for Correlation Curve Computation

3. **Spatial Descriptors**
   - Local density metrics
   - Orientational Order metrics
   - Cell type filtering (i.e. we look at a subsection of the cells.)

4. **Statistical Analysis**
   - Partial correlation coefficient computation
   - Spatial autocorrelation metrics

5. **Sequence Modeling & Classification**
   - Radial sequence generation from defined tumor core regions
   - Hidden Markov Models and other statistical sequence-aware AI for sequence analysis
   - Changepoint detection in spatial patterns using ruptures

### 2.2 User Characteristics
**Primary Users:** Computational biologists, biomedical researchers, and data scientists analyzing spatial transcriptomics or histology data.

**Technical Level:** The hope is that any biologist with some degree of computational experience can easily use this package: aiming to abstract as much detail of implementation as possiblewith modular programming. 

**Domain Knowledge:** Understanding of histology/spatial biology preferred but not required; physics background helpful for interpreting spatial descriptors.

### 2.3 Benefits
- Applies  materials science methods to biological spatial data
- Format-agnostic data loading reduces preprocessing overhead and ease of use for users.
- Modular design allows flexible analysis pipelines
- Physics-informed features captures patterns missed by conventional approaches(validated in exploratory analysis, submitted in Scientific Reports)

---

## 3. Functional Requirements

### 3.1 Data Input and Preprocessing
**FR-1.1:** The system shall load CSV files containing spatial coordinates and cell type labels with arbitrary column naming conventions.

**FR-1.2:** The system shall standardize input data to a consistent DataFrame format with standardized column names for coordinates, cell identifiers, cell types, and optional metadata fields.

**FR-1.3:** The system shall validate required columns exist and raise descriptive errors if missing.

**FR-1.4:** The system shall handle missing values by dropping rows with incomplete required fields and warning users with affected row indices.

**FR-1.5:** The system shall auto-generate unique identifiers for cells and images when not provided in input data.

### 3.2 Spatial Analysis
**FR-2.1:** The system shall compute kernel density estimates at specified spatial resolutions.

**FR-2.2:** The system shall calculate orientational order, local density using freud library methods.

**FR-2.4:** The system shall compute partial correlation coefficients between spatial features to integrate into radial correlation sequences.

**FR-2.3:** The system shall generate radial correlation sequences from user-specified core regions.



### 3.3 Classification and Detection
**FR-3.1:** The system shall train statistical, sequence aware models on quantile-discretized radial sequences.

**FR-3.2:** The system shall classify samples based on spatial sequence patterns.

**FR-3.3:** The system shall detect changepoints in spatial radial sequences.

**FR-3.4:** The system shall support deep learning-based sequence modeling (future extension if enough data is acquired that deep learning is successful, or will be implemented anyway once the package is released since other labs may have access to larger data banks).

### 3.4 Output and Reporting
**FR-4.1:** The system shall return analysis results as pandas DataFrames or numpy arrays.

**FR-4.2:** The system shall provide classification labels and confidence scores.

**FR-4.3:** The system shall report changepoint locations and statistical significance using ruptures.

---

## 4. Interface Requirements

### 4.1 Programming Interface
- **Language:** Python 3.9+
- **API Style:** Functional API with clear function signatures, type hints, and docstrings
- **Data Exchange:** pandas DataFrames for tabular data, numpy arrays for numerical results
- **Import Structure:** Modular package structure with subpackages for preprocessing, density kernel computation, descriptors, and classification:classification is not necessary the end goal, so further analysis methods like changepoint detection/curve analysis might be implemented.  

### 4.2 Input Format
- **File Format:** CSV (future: parquet, feather)
- **Required Fields:** Spatial coordinates (x and y), cell type label
- **Optional Fields:** Cell identifier, image/sample identifier, dataset/disease type, patient identifier

### 4.3 Output Format
- Standardized pandas DataFrames
- CSV files for classification results/changepoint detection

---

## 5. Performance Requirements

- Maximum file size for single CSV is highly dependent on computational resources. Pandas takes a while to load large dataframes: a method to load in "chunks" of each csv will probably be implemented to handle larger datasets. 
- KDE computation: Scalable to grid sizes dependent on RAM/ Resources available. Preliminary results showed diminishing returns with grid sizes > 100 x 100. 
- HMM training: Convergence within 2000 iterations for typical datasets
- Memory usage: Not Profiled yet, but most computationally intensive portion (freud), has required 50Gb of RAM if parallelization is warranted. 

---

## 6. Design Constraints

### 6.1 Technology Stack
- **Core Dependencies:** numpy, pandas, scipy, scikit-learn, hmmlearn, pytorch
- **Spatial Analysis:** freud, developed by the Glotzer Group
- **Python Version:** >=3.9, tested for 3.9,3.10,3.11,3.12

### 6.2 Development Tools
- **Formatting:** Black (line length 100)
- **Linting:** Ruff (with pydocstyle Google convention)
- **Type Checking:** mypy
- **Testing:** pytest


### 6.3 Platform Constraints
- freud library requires conda/mamba installation (not pip-installable)
- Cross-platform support: Linux, macOS, Windows (via WSL or conda environments)

---

## 7. Non-Functional Attributes

### 7.1 Reliability
- Comprehensive input validation with descriptive error messages
- Handling of missing/malformed data with user warnings
- Reproducible results with fixed random seed = 42

### 7.2 Usability
- Clear API with consistent naming conventions
- Detailed docstrings with usage examples

### 7.3 Maintainability
- Modular architecture with separation of concerns (preprocessing, density, descriptors, classification)
- Comprehensive test suite
- Pre-commit hooks for code quality in the development files 

### 7.4 Scalability
- Modular design allows processing subsets of large datasets by allowing for functions to be parallelized among GPU or CPU nodes. Inherent algorithm will not be CUDA-based, but the partial correlation/correlation computations may have their backend developed in Rust/C++ since it is the most computationally expensive part
- Development using small representative samples, validation on full datasets at completion

---

## 8. Preliminary Schedule and Budget

### 8.1 Development Timeline (Initial Phase)
- **Week 1-2:** Project infrastructure, data loader implementation (CURRENT)
- **Week 3-4:** Spatial descriptors and KDE functionality
- **Week 5-6:** Partial correlation computation
- **Week 7-8:** HMM sequence modeling and classification, **and start writing tutorial notebooks**
- **Week 9-10:** Testing, documentation, validation on full datasets, demonstration development
- **Week 10-11:** Project report / paper write-up

### 8.2 Budget
Open-source project, but funded via the Propelling Original Data Science Program by the Michigan Institute of Data, AI, and Society (Grant ID: 2578). Development time estimated at 9-12 weeks over the course of BIOINF 576 for the core functionality, but additional functionality added in the Sumemer Semester. 

---

## 9. Appendices

### 9.11 References
- freud documentation: https://freud.readthedocs.io/
- Project repository: (internal development)

### 9.12 Acronyms
- **CSV:** Comma-Separated Values
- **SRS:** Software Requirements Specification

---

**Document Control:**
- **Created By:** Specifications written by Ninad Kamath. Formatting done using GenAI (Claude), following geekforgeeks website for the outline formatting. 
- **Next Update:** After tutorial has been written, and initial testing has begun using full dataset.
