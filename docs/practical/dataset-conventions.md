# Dataset Conventions – Phase 11 Practical Workflows

## Purpose

This document defines how real datasets are stored, described, processed, and used in Phase 11.

Phase 11 uses real datasets to validate that the C++ ML Core models can be used in practical workflows.

The goal is not to build a full data engineering system.

The goal is to create a simple, reproducible, and documented workflow for:

- loading real tabular datasets
- converting them into the project’s `Matrix` / `Vector` format
- running model comparisons
- exporting structured outputs
- visualizing results from Python/Jupyter
- documenting how model APIs map to the underlying math

---

## Folder Structure

Real datasets must be stored under `data/`.

```txt
data/
├── raw/
├── processed/
└── metadata/
```

### `data/raw/`

Contains original downloaded datasets, with minimal or no modification.

Rules:

- Keep filenames close to the original dataset name when possible.
- Do not manually normalize or transform values here.
- Do not overwrite raw files after processing.
- If a dataset is downloaded in multiple files, keep them grouped using a subfolder.

Example:

```txt
data/raw/stock_ohlcv/
data/raw/nasa_kc1_software_defects/
data/raw/wine/
```

### `data/processed/`

Contains cleaned CSV files ready for the C++ workflow.

Rules:

- Files should be numeric-only where possible.
- Include one header row.
- Use comma-separated CSV format.
- Keep one row per sample.
- Keep target columns explicit.
- Avoid hidden transformations.

Example:

```txt
data/processed/stock_ohlcv_engineered.csv
data/processed/nasa_kc1_software_defects.csv
data/processed/wine.csv
```

### `data/metadata/`

Contains one metadata file per dataset.

Rules:

- Use Markdown for human-readable metadata.
- Each metadata file must explain the source, task type, target column, feature columns, preprocessing assumptions, and intended Phase 11 workflows.

Example:

```txt
data/metadata/stock_ohlcv_engineered.md
data/metadata/nasa_kc1_software_defects.md
data/metadata/wine.md
```

---

## Dataset Metadata Convention

Each dataset metadata file must use the following structure.

```md
# Dataset: <dataset-name>

## Source

- Source name:
- Source URL:
- Citation / reference:
- License / usage note:

## Task Type

One of:

- regression
- binary classification
- multiclass classification
- unsupervised / visualization

## Raw Data Location

```txt
data/raw/<dataset-folder>/
```

## Processed Data Location

```txt
data/processed/<dataset-name>.csv
```

## Target Column

For supervised datasets:

```txt
<target-column-name>
```

For unsupervised workflows:

```txt
none
```

If labels are kept only for qualitative comparison, state that clearly.

## Feature Columns

```txt
feature_1
feature_2
feature_3
...
```

## Label Encoding

Explain how the target is encoded.

Examples:

- Binary labels: `0` and `1`
- Multiclass labels: `0`, `1`, `2`
- Regression target: continuous numeric value
- Unsupervised: no training target

## Preprocessing Notes

Document any preprocessing applied to create the processed CSV.

Examples:

- removed non-numeric ID column
- converted class labels to integers
- selected one target column
- rejected rows with missing values
- preserved original numeric feature values
- scaling is not stored in the processed dataset; scaling is applied only inside experiment workflows when needed

## Intended Phase 11 Workflows

List which workflows use this dataset.

Examples:

- regression comparison
- binary classification comparison
- multiclass classification comparison
- PCA projection
- KMeans clustering
- hyperparameter sweeps

## Notes

Add any dataset-specific cautions.

Examples:

- dataset is small and mainly useful for workflow validation
- features should usually be standardized for distance-based models
- target has class imbalance
- dataset is not intended for production conclusions
```

---

## Initial Dataset Selection

Phase 11 starts with datasets that are more aligned with the project owner's target domains: finance, software engineering, and practical tabular ML.

The first objective is to validate the workflow, not to maximize benchmark performance.

The goal is to use datasets that are interesting enough to support practical examples, while still remaining small and structured enough for simple C++ experiments.

---

## Dataset 1 – Stock OHLCV Engineered Returns

### Purpose

Primary dataset for regression workflows.

Preferred dataset for unsupervised PCA/KMeans workflows if the same processed finance dataset can be reused cleanly.

### Task Types

- regression
- unsupervised / visualization

### Target

For regression, use a next-period return target such as:

```txt
target_next_return
```

For unsupervised workflows, the target column must not be used during training.

### Intended Regression Models

- `LinearRegression`
- Ridge / regularized linear regression behavior, if exposed cleanly
- `DecisionTreeRegressor`
- `GradientBoostingRegressor`

### Intended Unsupervised Models

- `PCA`
- `KMeans`
- PCA + KMeans combined workflow

### Example Engineered Feature Columns

The exact feature set may be adjusted when the raw dataset is selected, but the processed dataset should follow this general idea:

```txt
return_1d
return_5d
volatility_5d
range_pct
volume_change_1d
```

### Why This Dataset

This dataset gives Phase 11 a finance-oriented workflow while still remaining compatible with simple tabular ML.

The objective is not to build a trading system or claim predictive market alpha.

The objective is to demonstrate:

- real numeric feature engineering
- regression workflow structure
- train/test discipline
- model comparison
- structured output export
- optional PCA/KMeans analysis on engineered market features

### Planned Locations

```txt
data/raw/stock_ohlcv/
data/processed/stock_ohlcv_engineered.csv
data/metadata/stock_ohlcv_engineered.md
```

---

## Dataset 2 – NASA KC1 Software Defect Prediction

### Purpose

Primary dataset for binary classification workflows.

### Task Type

Binary classification.

### Target

```txt
defect
```

Recommended encoding:

```txt
0 = non-defective
1 = defective
```

The exact target column name must be confirmed when the source dataset is added.

### Intended Models

- `LogisticRegression`
- `LinearSVM`
- `GaussianNaiveBayes`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `TinyMLPBinaryClassifier`
- optional: `Perceptron`

### Why This Dataset

This dataset connects directly to software engineering and code-quality prediction.

It is useful for demonstrating binary classification on real engineering metrics rather than purely academic examples.

Class imbalance should be checked and documented before interpreting results.

### Planned Locations

```txt
data/raw/nasa_kc1_software_defects/
data/processed/nasa_kc1_software_defects.csv
data/metadata/nasa_kc1_software_defects.md
```

---

## Dataset 3 – Wine

### Purpose

Primary dataset for multiclass classification workflows.

Fallback dataset for unsupervised PCA/KMeans workflows if the stock workflow becomes too distracting or requires too much finance-specific preprocessing.

### Task Types

- multiclass classification
- unsupervised / visualization fallback

### Target

```txt
class
```

Recommended encoding:

```txt
0
1
2
```

For unsupervised workflows, the class column should not be used for training.

It may be kept only for qualitative comparison in exported visualization files.

### Intended Multiclass Models

- `SoftmaxRegression`
- `KNNClassifier`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GaussianNaiveBayes`

### Intended Unsupervised Models

- `PCA`
- `KMeans`
- PCA + KMeans combined workflow

### Why This Dataset

The dataset has numeric chemical measurements and three known classes.

It is small enough for simple C++ experiments, but rich enough to show PCA projection, clustering behavior, distance effects, and multiclass classification.

Wine is kept as a stable classical dataset so Phase 11 does not depend entirely on custom finance preprocessing.

### Planned Locations

```txt
data/raw/wine/
data/processed/wine.csv
data/metadata/wine.md
```

---

## Processed CSV Rules

Every processed dataset CSV should follow these rules:

- One header row.
- Numeric values only.
- Comma separator.
- No index column unless it is intentionally used as a feature.
- Target column must be explicit for supervised workflows.
- Missing values should be rejected or handled before the C++ workflow.
- Scaling should not be baked into the processed CSV unless the file name clearly states it.

Preferred pattern:

```txt
feature_1,feature_2,feature_3,target
...
```

For unsupervised workflows:

```txt
feature_1,feature_2,feature_3,class_for_reference
...
```

The reference class may be included for visualization, but must not be used during unsupervised training.

---

## Data Leakage Rule

Preprocessing that learns from data must be fitted only on the training split.

Examples:

- standardization mean/std
- min-max normalization limits
- PCA fitting

Therefore:

- raw data remains unchanged
- processed data should only contain basic cleaned values
- experiment workflows apply train-only preprocessing when needed

---

## Initial Phase 11 Dataset Decision

The initial Phase 11 datasets are:

| Workflow | Dataset | Domain | Role |
|---|---|---|---|
| Regression | Stock OHLCV engineered returns | Finance | Main regression workflow |
| Binary classification | NASA KC1 Software Defect Prediction | Software engineering | Main binary classification workflow |
| Multiclass classification | UCI Wine | Classical tabular ML | Main multiclass classification workflow |
| Unsupervised / PCA / KMeans | Stock OHLCV engineered features | Finance | Preferred unsupervised workflow if it reuses the stock regression dataset cleanly |
| Unsupervised fallback | UCI Wine | Classical tabular ML | Fallback unsupervised workflow if the stock workflow becomes too distracting |

This keeps Phase 11 aligned with practical domains while avoiding unnecessary dataset sprawl.

The stock OHLCV dataset should be used carefully.

The objective is not to build a trading system or claim predictive market alpha.

The objective is to demonstrate feature engineering, regression workflow structure, train/test discipline, model comparison, structured output export, and optional PCA/KMeans analysis on engineered market features.

Additional datasets may be added later only if the first workflow proves too limited.