# MLIS Group Assignment: Nepal Rainfall Prediction & Clustering

**Course:** Machine Learning & Intelligent Systems (MLIS) - Sixth Semester  
**Dataset:** [HDX Nepal Rainfall Subnational](https://data.humdata.org/dataset/npl-rainfall-subnational/resource/7fdfce1e-b6df-403b-be59-9065b3cac549)

---

##  Project Overview

This project implements a comprehensive machine learning pipeline for **rainfall prediction** and **climate zone discovery** in Nepal using subnational rainfall data (1981–2026). The work covers both supervised regression/forecasting models and unsupervised clustering techniques for pattern discovery.

###  Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Exploratory Data Analysis (EDA) of 138K+ rainfall records | 📋 Planned |
| 2 | Feature engineering for temporal, spatial, and lag features | 📋 Planned |
| 3 | Baseline regression models (Linear Regression) | 📋 Planned |
| 4 | Tree-based ensemble models (Random Forest, XGBoost) | 📋 Planned |
| 5 | Deep learning for time-series forecasting (LSTM) | 📋 Planned |
| 6 | Model comparison and hyperparameter tuning | 📋 Planned |
| 7 | Unsupervised clustering for rainfall zone discovery (K-Means, DTW K-Means, GMM, HDBSCAN) | 📋 Planned |
| 8 | Stratified modeling using cluster assignments | 📋 Planned |

---

## 📊 Dataset Description

| Property | Details |
|----------|---------|
| **Source** | HDX Nepal Rainfall Subnational |
| **Records** | 138,264 |
| **Date Range** | 1981-01-01 to 2026-09-11 |
| **Frequency** | 10-day intervals |
| **Administrative Levels** | 1 (Provinces: 7), 2 (Districts: 77) |
| **Target Variables** | `rfh`, `rfh_avg`, `r1h`, `r1h_avg`, `r3h`, `r3h_avg`, `rfq`, `r1q`, `r3q` |

### Variable Definitions

| Column | Description |
|--------|-------------|
| `date` | Observation date (10-day period) |
| `adm_level` | Administrative level (1=Province, 2=District) |
| `adm_id` | Unique administrative unit identifier |
| `PCODE` | Administrative unit code (e.g., NP01, NP0101) |
| `n_pixels` | Number of satellite pixels in the unit |
| `rfh` | **Total rainfall height** (mm) — primary target |
| `rfh_avg` | Average rainfall per pixel (mm) |
| `r1h` | Maximum 1-hour rainfall accumulation (mm) |
| `r1h_avg` | Average 1-hour maximum (mm) |
| `r3h` | Maximum 3-hour rainfall accumulation (mm) |
| `r3h_avg` | Average 3-hour maximum (mm) |
| `rfq` | Rainfall quantity index |
| `r1q` | 1-hour rainfall quantity index |
| `r3q` | 3-hour rainfall quantity index |
| `version` | Data version (`final` / `prelim`) |

---

## 🏗️ Project Structure

```
MLIS-GroupAssingment/
├── data/
│   └── nepal_rainfall_dataset.csv       # Raw dataset (138K records)
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb     # Feature engineering pipeline
│   ├── 03_baseline_models.ipynb         # Linear Regression baseline
│   ├── 04_tree_models.ipynb             # Random Forest & XGBoost
│   ├── 05_lstm_model.ipynb              # LSTM time-series forecasting
│   ├── 06_model_comparison.ipynb        # Model evaluation & comparison
│   ├── 07_clustering_kmeans.ipynb       # K-Means clustering
│   ├── 08_clustering_dtw.ipynb          # DTW K-Means (time-series)
│   └── 09_stratified_modeling.ipynb     # Cluster-stratified regression
├── src/                                 # Reusable source modules (planned)
│   ├── features.py
│   ├── models.py
│   ├── clustering.py
│   └── utils.py
├── models/                              # Saved model artifacts
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.json
│   ├── lstm_best.pth
│   ├── kmeans_rainfall.pkl
│   ├── dtw_kmeans_rainfall.pkl
│   ├── gmm_rainfall.pkl
│   └── xgb_cluster_*.pkl
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 🤖 Models Implemented

### Supervised Regression / Forecasting

| Model | Type | Target Variables | Key Strengths |
|-------|------|------------------|---------------|
| **Linear Regression** | Regression | `rfh`, `rfh_avg` | Interpretable, fast baseline |
| **Random Forest** | Regression | `rfh`, `rfh_avg`, `r3h` | Nonlinear, robust, feature importance |
| **XGBoost** | Regression | `rfh`, `r1h`, `r3h` | State-of-the-art tabular, handles missing |
| **LSTM** | Time-Series Forecasting | `rfh`, `rfh_avg` (per admin unit) | Captures temporal dependencies, seasonality |

### Unsupervised Clustering

| Model | Type | Use Case |
|-------|------|----------|
| **K-Means** | Clustering (aggregated features) | Quick baseline, interpretable zones |
| **DTW K-Means** | Clustering (time-series) ⭐ | **Monsoon timing patterns**, phase-invariant |
| **GMM** | Soft Clustering | Probabilistic assignments, uncertainty |
| **Hierarchical** | Multi-scale Clustering | Nested zones (province → district) |
| **HDBSCAN** | Density-based + Outliers | Automatic K, explicit outlier detection |

> **Note:** Clustering is unsupervised — no target variable needed. Used for rainfall zoning, anomaly detection, and stratified modeling.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MLIS-GroupAssingment

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Additional clustering dependencies
pip install tslearn hdbscan umap-learn
```

### Running Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or run specific notebooks in order:
# 1. 01_eda.ipynb              → Data exploration
# 2. 02_feature_engineering.ipynb  → Feature creation
# 3. 03_baseline_models.ipynb  → Linear Regression
# 4. 04_tree_models.ipynb      → Random Forest, XGBoost
# 5. 05_lstm_model.ipynb       → LSTM forecasting
# 6. 06_model_comparison.ipynb → Compare all models
# 7. 07_clustering_kmeans.ipynb → K-Means clustering
# 8. 08_clustering_dtw.ipynb   → DTW K-Means
# 9. 09_stratified_modeling.ipynb → Cluster-stratified regression
```

---

## 🔬 Methodology Highlights

### Feature Engineering
- **Temporal**: Year, month, day-of-year, season (4 seasons), cyclical encoding (sin/cos)
- **Lag features**: Previous 1, 2, 3, 6 periods (10–60 days) grouped by admin unit
- **Rolling statistics**: 3, 6, 12-period rolling mean/std
- **Administrative**: One-hot encoded admin level, categorical PCODE

### Time-Aware Validation
- **No random splits** — data split by date (pre-2020 train, post-2020 test)
- **TimeSeriesSplit** for cross-validation
- Prevents data leakage from temporal autocorrelation

### Clustering Best Practices
1. Aggregate to admin-unit level (don't cluster 138K raw rows)
2. Select meaningful features: mean, std, extremes, seasonality
3. Log-transform skewed rainfall variables
4. Encode cyclical features (month → sin/cos)
5. Scale with RobustScaler
6. Validate with multiple metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)

---

## 📈 Expected Performance

| Model | Expected R² | Notes |
|-------|-------------|-------|
| Linear Regression | 0.45–0.65 | Baseline |
| Random Forest | 0.65–0.80 | Strong nonlinear |
| XGBoost | 0.70–0.85 | Typically best for tabular |
| LSTM | RMSE 0.05–0.15 (scaled) | Best for temporal dynamics |

### Clustering Output
- **5–7 interpretable rainfall zones** across Nepal's 77 districts
- Zone examples: "High Monsoon Core", "Rain Shadow/Trans-Himalayan", "Moderate Monsoon (Mid-Hills)"

---

## 🔑 Key Considerations for This Dataset

1. **Time-series nature**: Always split by date, never random shuffle
2. **Hierarchical structure**: 7 provinces → 77 districts; consider hierarchical modeling
3. **Strong seasonality**: Monsoon signal (Jun–Sep); use cyclical encoding
4. **Spatial correlation**: Nearby districts have similar rainfall
5. **Heavy-tailed distribution**: Extreme events; consider log-transform or quantile regression
6. **Missing data**: Lag features create NaN at start of each admin unit; handle appropriately

---

## 📚 References

### Regression & Forecasting
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

### Clustering
- [tslearn Documentation](https://tslearn.readthedocs.io/) — Time series clustering (DTW K-Means)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/) — Density-based clustering
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html) — K-Means, GMM, Hierarchical
- [DTW Algorithm Explained](https://en.wikipedia.org/wiki/Dynamic_time_warping)
- [Sakoe-Chiba Band](https://www.cs.unm.edu/~mueen/DTW.pdf) — Constrained DTW
- [UMAP for Visualization](https://umap-learn.readthedocs.io/) — Dimensionality reduction

---

## 👥 Contributors

- **Apil120** - Project setup, EDA, implementation guide

---

## 📄 License

Academic project for MLIS coursework. Dataset sourced from HDX (Humanitarian Data Exchange) under their terms of use.

---

*Last updated: September 2026*