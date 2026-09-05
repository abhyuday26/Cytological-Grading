# AI-Based Cytological Grading Algorithm

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated machine learning framework for **3-tier cytological grading** (Grade 1: Low grade, Grade 2: Intermediate grade, Grade 3: High grade) using nuclear morphometric parameters derived from ImageJ microscopy analysis.

The system utilizes an **unsupervised learning paradigm** evaluated on a **75% Train / 25% Validation Holdout split**, supported by Principal Component Analysis (PCA) and rigorous statistical significance testing ($p < 0.001$).

---

## 🌟 Key Features

- **Automated Data Cleaning**: Detects and prunes anomalous or near-zero area measurements.
- **Derived Morphometric Feature Engineering**: Computes domain-specific cytopathological metrics:
  - **Nuclear Size Index**: $\text{Area} \times \text{Feret}$
  - **Irregularity Index**: $1 - \text{Circularity}$
  - **Pleomorphism Index**: $\text{Aspect Ratio} \times (1 - \text{Roundness})$
  - **Nuclear Compactness**: $\frac{\text{Perimeter}^2}{4\pi \cdot \text{Area}}$
- **Dimensionality Reduction**: Retains 95%+ variance using 5 Principal Components.
- **Multi-Model Benchmark**: Evaluates and compares **K-Means**, **Hierarchical Agglomerative (Ward)**, and **Gaussian Mixture Models (GMM)** across Train & Validation holdout metrics.
- **Case-Level Aggregation**: Aggregates individual cell grades to final case diagnosis via **majority voting**.
- **Publication-Ready Visualizations**: Generates 10 high-resolution analytical plots (PCA projections, feature boxplots, heatmaps, dendrograms).

---

## 📁 Repository Structure

```text
├── data/
│   └── Results_ImageJ.csv            # Raw nuclear morphometric data from ImageJ
├── output/
│   ├── graded_cells.csv              # Cell-level predictions with morphometric features
│   ├── case_grades.csv               # Case-level summary and majority vote diagnosis
│   ├── model_metrics.csv             # Train vs Validation clustering performance metrics
│   ├── Executive_Report.md           # Non-technical summary report for medical practitioners
│   └── plots/                        # 10 High-resolution visualization charts
│       ├── 01_pca_scree_plot.png
│       ├── 02_pca_loadings.png
│       ├── 03_cluster_optimization.png
│       ├── 04_model_comparison.png
│       ├── 05_pca_grade_scatter.png
│       ├── 06_feature_boxplots.png
│       ├── 07_correlation_heatmap.png
│       ├── 08_dendrogram.png
│       ├── 09_case_grades.png
│       └── 10_centroid_heatmap.png
├── cytological_grading.py            # Primary ML execution pipeline script
├── validate_results.py               # Automated verification and statistical test suite
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🔬 Morphometric Grade Profiles

The model categorizes cells into 3 distinct cytological tiers based on physical geometry:

| Grade | Description | Mean Area ($\mu m^2$) | Mean Circularity | Mean Roundness | Mean Irregularity Index | Morphometric Characteristics |
|---|---|---|---|---|---|---|
| **Grade 1** | Low grade | $82.42$ | $0.7940$ | $0.7817$ | $0.2060$ | Small, smooth, highly regular round nuclei |
| **Grade 2** | Intermediate grade | **$231.17$** | $0.7935$ | $0.8122$ | $0.2065$ | **Nuclear enlargement (Macro-nuclei)** |
| **Grade 3** | High grade | $82.36$ | **$0.6775$** | **$0.6162$** | **$0.3225$** | **Severe shape distortion & jagged edges** |

> All features show statistically significant separation across grades ($p < 10^{-12}$ on ANOVA & Kruskal-Wallis tests).

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure Python 3.10+ is installed. Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/cytological-grading-ai.git
cd cytological-grading-ai
pip install -r requirements.txt
```

### 2. Run Main Pipeline
To run the full end-to-end data cleaning, feature engineering, clustering, and plot generation pipeline:

```bash
python cytological_grading.py
```

### 3. Run Validation Suite
To execute statistical tests (ANOVA/Kruskal-Wallis) and verify data integrity:

```bash
python validate_results.py
```

---

## 📊 Benchmark Model Results

Models were evaluated on a 25% unseen holdout validation set:

| Model | Train Silhouette | **Validation Silhouette** | Train DB Index | **Validation DB Index** | Selected |
|---|---|---|---|---|---|
| **Hierarchical (Ward)** | 0.2498 | **0.2757** | 1.3641 | **1.2400** | **Best Model** |
| **K-Means** | 0.2705 | 0.2678 | 1.2594 | 1.2545 | Baseline |
| **Gaussian Mixture Model** | 0.2244 | 0.2488 | 1.4774 | 1.3489 | Rejected |

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
