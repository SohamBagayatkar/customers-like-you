# Customers Like You: PCA or SVD for a Real Retail Problem?

A retail recommendation project using the **UCI Online Retail II** dataset to discover customer purchasing patterns, recommend products, and compare dimensionality-reduction techniques.

## 1. Success Criteria — Section 1.4

| Success criterion | Result | Evidence |
|---|---|---|
| Compressed data is much smaller than the full matrix | **Achieved** | Sparse matrix: 3.21 MB; equivalent dense matrix: 94.85 MB |
| Nightly recommendation job runs quickly | **Achieved for the current prototype** | 27.89 seconds to generate and save recommendations for 5,484 customers, starting from the saved training matrix |
| Recommendations outperform the best-seller baseline | **Achieved by the binary follow-up experiment** | Binary SVD + product-to-product cosine similarity: 24.74% Hit Rate@10; best-seller baseline: 17.38% |
| Customer map contains interpretable groups | **Achieved provisionally** | Four customer segments identified and interpreted from purchasing patterns |

**Important:** The original SVD recommendation method achieved a Hit Rate@10 of 15.29%, below the 17.38% best-seller baseline. The improvement to 24.74% came from the follow-up experiment using a binary purchase matrix with the ML3 product-to-product cosine similarity method.

---

## 2. Project Overview

Retail businesses need to recommend relevant products to customers without manually analyzing every customer-product relationship.

This project uses transaction data to:

- Inspect and clean the retail dataset.
- Split historical transactions into training and held-out test periods.
- Build a sparse customer-product matrix.
- Compare PCA and SVD for dimensionality reduction.
- Recommend similar products and generate personalized top-10 recommendations.
- Evaluate recommendation quality using Hit Rate@10 against a best-seller baseline.
- Segment customers and visualize their purchasing patterns.
- Investigate how count, binary, and log-transformed purchase representations affect performance.

### Dataset

**Source:** UCI Machine Learning Repository — Online Retail II

The dataset contains transactions from a UK-based online retailer between December 2009 and December 2011. The original workbook contains **1,067,371 transaction rows across its two sheets**, before cleaning.

The dataset includes information such as invoice number, product code, product description, quantity, invoice date, unit price, customer ID, and country.

The project uses the two sheets for data profiling, cleaning, temporal splitting, and recommendation analysis.

---

## 3. Technology Stack

- **Python** — data processing and analysis
- **Pandas and NumPy** — cleaning and numerical computation
- **SciPy sparse matrices** — memory-efficient customer-product representation
- **scikit-learn** — Truncated SVD, PCA, cosine similarity, normalization, and clustering
- **Matplotlib** — visualizations
- **Jupyter Notebook** — experiments and reporting

---

## 4. Project Structure

```text
customers-like-you/
├── data/
│   └── raw/
│       └── online_retail_II.xlsx
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── build_matrix.py
│   └── customer_segments.py
├── outputs/
│   ├── figures/
│   │   └── customer_segmentation_map.png
│   └── matrix/
│       ├── customer_product_matrix.npz
│       ├── customers.json
│       └── products.json
├── reports/
│   ├── binary_ml3_experiment.csv
│   ├── cleaning_log.csv
│   ├── data_profile.csv
│   ├── example_customer_recommendations.csv
│   ├── hit_rate_evaluation.csv
│   ├── matrix_report.csv
│   ├── ml7_sensitivity_analysis.csv
│   ├── nightly_recommendations.csv
│   ├── pca_component_experiments.csv
│   ├── pca_vs_svd_comparison.csv
│   ├── sheet_row_counts.csv
│   ├── svd_component_experiments.csv
│   └── top_similar_products.csv
├── cleaned_transactions.csv
├── test_targets.csv
├── test_targets_evaluable.csv
├── test_transactions.csv
├── train_transactions.csv
├── .gitignore
├── README.md
└── requirements.txt
```

The `outputs/` directory also contains additional reports and figures generated during the analysis. Some output files are created when the relevant scripts or notebook cells run.

---

## 5. Setup and Run Instructions

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies
- Jupyter Notebook
- The Online Retail II Excel workbook

### Step 1: Clone or open the project

Open a terminal in the project directory:

```powershell
cd C:\Users\Dell\Documents\customers-like-you
```

### Step 2: Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 4: Place the dataset

Ensure the original workbook is available at:

```text
data/raw/online_retail_II.xlsx
```

### Step 5: Build the cleaned data and sparse matrix

From the project root, run:

```powershell
python src/build_matrix.py
```

This script profiles the workbook, applies the cleaning rules, creates the temporal train/test split, and saves the training matrix and related mappings and reports.

### Step 6: Run the analysis notebook

Start Jupyter:

```powershell
jupyter notebook
```

Open `notebooks/analysis.ipynb` and run the cells from top to bottom. This reproduces the analysis, dimensionality-reduction comparisons, recommendation evaluations, sensitivity experiments, and customer segmentation, provided the required dependencies and generated files are available.

---

## 6. Data Processing and Matrix Construction

The preprocessing pipeline applies cleaning rules to remove records with missing customer IDs or descriptions, cancelled invoices, non-positive quantities or prices, and non-product codes according to the project's cleaning rules.

The cleaned transactions are divided temporally, reserving the last two months for testing. The customer-product matrix is built using training transactions only, preventing held-out purchases from entering the training representation.

### Matrix statistics

| Metric | Result |
|---|---:|
| Cleaned transaction rows | 802,945 |
| Training transaction rows | 684,346 |
| Test transaction rows | 118,599 |
| Customer rows | 5,484 |
| Product columns | 4,534 |
| Non-zero entries | 417,804 |
| Sparsity | 98.32% |
| Dense matrix size | 94.85 MB |
| Sparse matrix size | 3.21 MB |

The sparse matrix is approximately **29.5 times smaller** than its dense equivalent, making sparse storage appropriate for this dataset.

---

## 7. Dimensionality Reduction: PCA vs. SVD

Truncated SVD was selected as the primary technique because the customer-product matrix is sparse and should not be centered into a dense matrix.

Reconstruction error was evaluated at component counts of 10, 20, 50, 100, and 200.

| Components | SVD reconstruction error | PCA reconstruction error |
|---:|---:|---:|
| 10 | 0.448859 | 0.448492 |
| 20 | 0.315569 | 0.315360 |
| 50 | 0.189402 | 0.189256 |
| 100 | 0.123869 | 0.123759 |
| 200 | 0.083629 | 0.083490 |

Both methods showed decreasing reconstruction error as the number of components increased. PCA produced slightly lower reconstruction error in these experiments, but the differences were small. SVD remained the practical choice for the sparse input representation.

See `outputs/figures/` for the reconstruction-error and PCA-versus-SVD comparison plots.

---

## 8. Recommendation Results

### Evaluation method

Recommendations are evaluated using held-out purchases from the final two months. The primary metric is **Hit Rate@10**: the proportion of evaluated customers for whom at least one of the ten recommended products matches a held-out purchase.

The evaluation used 1,916 customers with eligible held-out targets.

### Original ML4 comparison

| Method | Hit Rate@10 |
|---|---:|
| Original SVD + ML3 product-similarity method | 15.29% |
| Best-seller baseline | 17.38% |

The original SVD recommendation method did not outperform the best-seller baseline.

### Binary purchase representation: follow-up experiment

The follow-up experiment converted non-zero purchase quantities to binary indicators and applied SVD followed by ML3's product-to-product cosine similarity scoring method.

| Metric | Result |
|---|---:|
| Customers evaluated | 1,916 |
| Customers with a hit | 474 |
| Hit Rate@10 | 24.74% |
| Recommendation evaluation time | 12.17 seconds |

This result exceeds the best-seller baseline by **7.36 percentage points**. It suggests that purchase occurrence can be more useful than raw purchase quantity for this recommendation task.

The binary experiment is reported separately from the original ML4 results because it uses a different purchase representation and recommendation configuration.

### Sensitivity analysis

ML7 compared count, binary, and log-transformed purchase representations using customer-factor scoring:

| Representation | Hit Rate@10 |
|---|---:|
| Count | 12.06% |
| Binary | 29.96% |
| Log-transformed | 27.40% |

These results use a different scoring method from the binary follow-up experiment above and should not be compared directly with its 24.74% result.

---

## 9. Runtime Evaluation

The full recommendation prototype was timed starting from the saved customer-product matrix.

| Pipeline step | Runtime |
|---|---:|
| Load data and mappings | 0.14 seconds |
| Build binary matrix | 0.01 seconds |
| Fit SVD | 0.83 seconds |
| Build product-similarity matrix | 0.15 seconds |
| Generate recommendations | 26.58 seconds |
| Save recommendations | 0.18 seconds |
| **Total** | **27.89 seconds** |

The pipeline generated **54,840 recommendations for 5,484 customers**, providing ten recommendations per customer in this run.

Recommendation generation was the main computational bottleneck. The measured runtime demonstrates that the current prototype is fast for this workload.

This measurement does not include rebuilding the cleaned dataset and training matrix from the original Excel workbook, and the pipeline is not yet a scheduled production job.

The generated recommendations are saved to:

```text
outputs/reports/nightly_recommendations.csv
```

---

## 10. Customer Segmentation

Customer purchasing behavior was projected into two dimensions for visualization and grouped into four clusters.

The resulting groups were interpreted using their leading product preferences:

1. **Everyday Giftware & Novelty Shoppers**
2. **Party & Seasonal Gift Buyers**
3. **Heart & Decorative Homeware Buyers**
4. **Reusable Bags & Storage Shoppers**

The analysis suggests that product preferences provide useful distinctions between customer groups, while customers from the UK dominate across the clusters.

The customer segmentation visualization is saved as:

```text
outputs/figures/customer_segmentation_map.png
```

---

## 11. Key Findings

- Sparse matrix storage substantially reduces memory requirements.
- Reconstruction error decreases as more components are retained.
- PCA and SVD had similar reconstruction errors in the experiments.
- The original SVD recommendation method did not beat the best-seller baseline.
- Binary purchase indicators combined with product-to-product similarity achieved a 24.74% Hit Rate@10, exceeding the baseline.
- The current recommendation prototype completed in 27.89 seconds from the saved training matrix.
- Four interpretable customer segments were identified.

## 12. Limitations and Next Steps

- The runtime measurement excludes rebuilding the matrix from the original workbook.
- The recommendation pipeline has not yet been deployed as an automated nightly job.
- Hit Rate@10 records whether at least one held-out product appears in the top ten; it does not measure ranking quality beyond this binary outcome.
- Results may vary with temporal splits, representation choices, and recommendation scoring methods.
- The customer segments are exploratory interpretations and would benefit from further validation.
- The recommendation-generation step is the primary runtime bottleneck and could be optimized in future work.

## 13. Reproducibility

The notebook contains the experiments and evaluation results, while `src/build_matrix.py` provides the data preparation and sparse-matrix construction pipeline. Generated matrices, reports, recommendation outputs, and figures are stored under `outputs/`.

Run the data preparation script before running the notebook if the generated files are missing. Ensure the original dataset and dependencies are available before reproducing the results.