<div align="center">

# Customer Segmentation Using RFM Analysis and   K-Means Clustering

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-0052CC?style=for-the-badge&logo=archive&logoColor=white)](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> An end-to-end unsupervised machine learning pipeline to segment retail customers based on historical purchasing behavior using RFM (Recency, Frequency, Monetary) feature engineering, skewness correction, K-Means clustering, and interactive 3D Plotly visualizations.

</div>

---

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Project Workflow & Preprocessing](#project-workflow--preprocessing)
- [Feature Skewness & Transformation](#feature-skewness--transformation)
- [Optimal Cluster Selection & Segmentation](#optimal-cluster-selection--segmentation)
- [Interactive 3D Visualization](#interactive-3d-visualization)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)

---

## About the Project

Customer segmentation allows retail businesses to understand distinct purchasing behaviors, optimize marketing resource allocation, and tailor retention strategies. This project implements a complete unsupervised learning pipeline on the **Online Retail II** dataset (covering transactions from December 2009 to December 2011).

Because raw retail data exists at the individual invoice-line level and contains missing identifiers, catalog inconsistencies, and order cancellations, the pipeline systematically cleans the transaction records, aggregates them into customer-level **RFM** profiles, normalizes heavily skewed distributions, and groups customers into 4 distinct behavioral segments using **K-Means Clustering**.

**Key Highlights:**
- **Data Hygiene:** Processed 1,067,371 raw transaction records by imputing recoverable `Description` and `Customer ID` values, enforcing a 1-to-1 `StockCode` description mapping with a chronological tie-breaker, and filtering out 18,744 cancelled orders (`'C'` prefix) and 71 non-positive price entries.
- **RFM Feature Engineering:** Consolidated 805,549 valid transaction lines into **5,878 unique customer profiles** defined by `Recency` (days since last purchase), `Frequency` (unique invoice count), and `Monetary` (total spend).
- **Distribution Normalization & Clustering:** Mitigated extreme right-skewness (`Monetary` skewness reduced from `25.31` to `0.27`) using `np.log1p` transformation and `StandardScaler` normalization before fitting an optimal `K = 4` K-Means model validated via Inertia and Silhouette scores.

---

## Dataset

| Property | Details |
|----------|---------|
| **Source** | [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| **Time Span** | December 1, 2009 – December 9, 2011 |
| **Original Size** | 1,067,371 transaction rows across 8 columns |
| **Cleaned Transaction Size** | 805,549 valid positive purchase records |
| **Aggregated Customer Profiles** | 5,878 unique customers across 3 behavioral RFM features |

---

## Project Workflow & Preprocessing

The pipeline cleans and transforms the raw transaction logs through five structured stages:

1. **Missing Value Recovery & Imputation:** Grouped records by `StockCode` to forward-fill and backward-fill missing product `Description` values, and grouped by `Invoice` to recover missing `Customer ID` entries across shared checkout baskets before dropping remaining unidentifiable rows (243,007 initial missing `Customer ID` rows and 4,382 missing `Description` rows).
2. **Product Catalog Standardization:** Resolved inconsistent text descriptions assigned to identical `StockCode` identifiers by mapping each `StockCode` to its statistical mode, backed by a deterministic chronological tie-breaker (`resolve_tie_chronological`) when multiple descriptions shared the highest frequency.
3. **Identifier Formatting & Feature Selection:** Converted `Customer ID` from floating-point decimals (`13085.0`) to clean integer strings (`"13085"`) and removed the out-of-scope `Country` column.
4. **Anomaly & Cancellation Filtering:** Removed 18,744 cancelled orders (`Invoice` numbers prefixed with `'C'` carrying negative quantities) and 71 non-positive price adjustments (`Quantity > 0` and `Price > 0`) to prevent distortion of Euclidean distances and logarithmic errors, then computed `TotalPrice = Quantity * Price` per line item.
5. **Customer-Level RFM Aggregation:** Using a reference date set to one day after the final transaction (`2011-12-10`), consolidated all valid purchases by `Customer ID` to compute `Recency` (days since last purchase), `Frequency` (`nunique` invoices), and `Monetary` (`sum` of `TotalPrice`).

---

## Feature Skewness & Transformation

Retail purchasing metrics exhibit severe right-skewness due to wholesale buyers and high-frequency regulars. Because K-Means relies on Euclidean distance, `np.log1p` ($\ln(1 + x)$) was applied prior to standardization with `StandardScaler`:

| Feature | Raw Skewness | Log-Transformed Skewness (`np.log1p`) | Post-Scaling Mean | Post-Scaling Std Dev |
|---------|:------------:|:-------------------------------------:|:-----------------:|:--------------------:|
| **Recency** | 0.89 | -0.55 | 0.00 | 1.00 |
| **Frequency** | 12.64 | 1.00 | 0.00 | 1.00 |
| **Monetary** | 25.31 | 0.27 | 0.00 | 1.00 |

---

## Optimal Cluster Selection & Segmentation

The optimal number of clusters was evaluated across `K = 2` to `K = 11` (`random_state=42`, `n_init=10`) using both the **Elbow Method (Within-Cluster Sum of Squares / Inertia)** and **Silhouette Scores**. `K = 4` was selected as the optimal configuration, capturing the primary inflection point on the inertia curve alongside a strong local peak in cluster separation[cite: 2, 5].

| Cluster Label | Customer Count | Share of Customer Base (%) |
|:-------------:|:--------------:|:--------------------------:|
| **Cluster 0** | **2,055** | **34.96%** |
| **Cluster 1** | **1,478** | **25.14%** |
| **Cluster 2** | **1,228** | **20.89%** |
| **Cluster 3** | **1,117** | **19.00%** |
| **Total** | **5,878** | **100.00%** |

---

# Customer Segmentation Using RFM and K-Means

## 📁 Project Structure

```text
Customer-Segmentation-RFM-KMeans/
│
├── Customer Segregation.ipynb   # Main end-to-end Jupyter Notebook
├── online_retail_II.csv         # Raw UCI Online Retail II dataset
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

The Jupyter Notebook covers the complete customer segmentation workflow, including data cleaning, RFM feature engineering, K-Means clustering, and interactive 3D visualization using Plotly.

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Aniket728497/Customer-Segregation.git
```

Navigate to the project directory:

```bash
cd Customer-Segmentation
```

### 2. Create a Virtual Environment

Create a Python virtual environment to isolate the project's dependencies:

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Launch the Jupyter Notebook

Ensure that `online_retail_II.csv` is located in the project root directory alongside the Jupyter Notebook.

Launch the notebook using:

```bash
jupyter notebook "Customer Segregation.ipynb"
```

Once Jupyter Notebook opens in your browser, open the notebook and execute the cells sequentially to run the complete workflow.

---

## 🛠️ Tech Stack

| Category                    | Technologies                      |
| --------------------------- | --------------------------------- |
| **Programming Language**    | Python 3.8+                       |
| **Data Manipulation**       | Pandas, NumPy                     |
| **Data Visualization**      | Matplotlib, Seaborn, Plotly       |
| **Machine Learning**        | Scikit-learn                      |
| **Clustering Algorithm**    | K-Means                           |
| **Cluster Validation**      | Silhouette Score, Inertia (WCSS)  |
| **Feature Transformation**  | `np.log1p`, `StandardScaler`, PCA |
| **Development Environment** | Jupyter Notebook                  |

