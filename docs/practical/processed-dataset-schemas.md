# Processed Dataset Schemas – Phase 11 Practical Workflows

## Purpose

This document defines the processed CSV schemas used in Phase 11.

The raw datasets are stored under:

```txt
data/raw/
```

The processed datasets produced from them must be stored under:

```txt
data/processed/
```

The goal of this document is to freeze the exact CSV contracts before implementing dataset conversion, CSV loading, and experiment runners.

---

## General Processed CSV Rules

All processed CSV files must follow these rules:

- One header row.
- Comma-separated values.
- Numeric model input columns only.
- No missing values.
- No string categorical values in model input columns.
- Target column must be explicit for supervised datasets.
- Target column should be the last column when possible.
- Raw source identifiers such as dates or tickers should not be used as numeric model features unless explicitly encoded.
- Scaling should not be baked into the processed CSV by default.
- Train-only preprocessing, such as standardization, must happen inside the experiment workflow after the train/test split.

---

## Data Leakage Rule

Any preprocessing step that learns from data must be fitted only on the training split.

Examples:

- standardization mean and standard deviation
- min-max scaling limits
- PCA components
- class balancing logic, if added later

Therefore, the processed CSV files should contain cleaned and engineered values, but not train-aware transformations.

Allowed in processed files:

- deterministic row filtering
- deterministic label encoding
- deterministic feature engineering from each row or from past time steps only
- removing non-numeric columns from the model-ready file

Not allowed in processed files:

- scaling using full-dataset statistics
- feature selection using target performance
- PCA projection fitted on the full dataset
- any transformation that uses future information in time-series features

---

# Dataset 1 – Stock OHLCV Engineered Returns

## Role

Primary dataset for:

- regression
- optional unsupervised PCA/KMeans workflow

## Raw Data Location

```txt
data/raw/stock_ohlcv/
```

Expected raw files:

```txt
data/raw/stock_ohlcv/aapl_us_d.csv
data/raw/stock_ohlcv/amzn_us_d.csv
data/raw/stock_ohlcv/googl_us_d.csv
data/raw/stock_ohlcv/jpm_us_d.csv
data/raw/stock_ohlcv/meta_us_d.csv
data/raw/stock_ohlcv/msft_us_d.csv
data/raw/stock_ohlcv/nvda_us_d.csv
data/raw/stock_ohlcv/xom_us_d.csv
```

Expected raw columns:

```txt
Date,Open,High,Low,Close,Volume
```

## Processed Data Location

```txt
data/processed/stock_ohlcv_engineered.csv
```

## Task Type

Regression.

Optional unsupervised workflow using the same feature columns.

## Target Column

```txt
target_next_return
```

The target should represent the next-day close-to-close return:

```txt
target_next_return = (Close[t + 1] - Close[t]) / Close[t]
```

The last available row for each ticker must be rejected because it has no next-day target.

## Feature Columns

Recommended initial feature columns:

```txt
return_1d
return_5d
volatility_5d
range_pct
volume_change_1d
```

Feature definitions:

```txt
return_1d        = (Close[t] - Close[t - 1]) / Close[t - 1]
return_5d        = (Close[t] - Close[t - 5]) / Close[t - 5]
volatility_5d    = standard deviation of daily returns over the last 5 available returns
range_pct        = (High[t] - Low[t]) / Close[t]
volume_change_1d = (Volume[t] - Volume[t - 1]) / Volume[t - 1]
```

Important:

- All features must use information available at time `t` or earlier.
- The target uses `t + 1`.
- The same row must never use future values inside its feature columns.

## Reference Columns

The processed model-ready CSV should be numeric-only, but for debugging and notebooks it is useful to preserve references.

Recommended approach:

Create one model-ready file:

```txt
data/processed/stock_ohlcv_engineered.csv
```

with only:

```txt
return_1d,return_5d,volatility_5d,range_pct,volume_change_1d,target_next_return
```

Optionally create a separate reference file later:

```txt
data/processed/stock_ohlcv_engineered_reference.csv
```

with:

```txt
ticker,date,return_1d,return_5d,volatility_5d,range_pct,volume_change_1d,target_next_return
```

Do not make the C++ model loader depend on the reference file.

## Final Processed CSV Schema

```csv
return_1d,return_5d,volatility_5d,range_pct,volume_change_1d,target_next_return
```

## Rows to Reject

Reject rows where:

- any raw numeric field is missing
- `Close[t] <= 0`
- `Close[t - 1] <= 0`
- `Close[t - 5] <= 0`
- `Volume[t - 1] <= 0`
- not enough lookback history exists
- no next-day target exists
- any engineered value is NaN or infinite

## Scaling Policy

Do not scale this dataset in the processed CSV.

Scaling should be applied inside the experiment workflow, fitted only on the training split.

Recommended:

- standardize feature columns for `LinearRegression`
- standardize feature columns for PCA/KMeans
- tree-based regressors can use raw engineered features

## Intended C++ Loading Contract

For regression:

```txt
X = columns:
  return_1d
  return_5d
  volatility_5d
  range_pct
  volume_change_1d

y = column:
  target_next_return
```

For unsupervised PCA/KMeans:

```txt
X = columns:
  return_1d
  return_5d
  volatility_5d
  range_pct
  volume_change_1d
```

The target column must not be used during unsupervised training.

---

# Dataset 2 – NASA KC1 Software Defect Prediction

## Role

Primary dataset for binary classification.

## Raw Data Location

```txt
data/raw/nasa_kc1_software_defects/kc1.arff
```

Expected ARFF relation:

```txt
@relation KC1
```

Expected target attribute:

```txt
@attribute defects {false,true}
```

## Processed Data Location

```txt
data/processed/nasa_kc1_software_defects.csv
```

## Task Type

Binary classification.

## Target Column

```txt
defects
```

Recommended encoding:

```txt
false -> 0
true  -> 1
```

## Feature Columns

The processed CSV should preserve the numeric feature columns from the ARFF file.

Recommended feature columns:

```txt
loc
v_g
ev_g
iv_g
n
v
l
d
i
e
b
t
lOCode
lOComment
lOBlank
locCodeAndComment
uniq_Op
uniq_Opnd
total_Op
total_Opnd
branchCount
```

Note about renamed columns:

The ARFF file contains names such as:

```txt
v(g)
ev(g)
iv(g)
```

For the processed CSV, rename them to safe CSV/header identifiers:

```txt
v_g
ev_g
iv_g
```

This avoids parsing issues in later C++ utilities.

## Final Processed CSV Schema

```csv
loc,v_g,ev_g,iv_g,n,v,l,d,i,e,b,t,lOCode,lOComment,lOBlank,locCodeAndComment,uniq_Op,uniq_Opnd,total_Op,total_Opnd,branchCount,defects
```

## Rows to Reject

Reject rows where:

- any feature value is missing
- any feature value is NaN or infinite
- target is not one of `false`, `true`, `0`, or `1`
- the row has the wrong number of columns

The raw metadata reports no missing attributes, so row rejection should normally be minimal.

## Class Imbalance Note

The raw dataset is imbalanced.

Reported distribution:

```txt
false / non-defective: majority class
true / defective: minority class
```

This must be considered during interpretation.

Accuracy alone should not be the only metric.

Recommended metrics:

- accuracy
- precision
- recall
- F1 score
- confusion matrix values, if already available in the project

## Scaling Policy

Do not scale this dataset in the processed CSV.

Scaling should be applied inside experiment workflows where needed.

Recommended:

- standardize feature columns for `LogisticRegression`
- standardize feature columns for `LinearSVM`
- standardize feature columns for `TinyMLPBinaryClassifier`
- scaling is less critical for trees and Random Forest
- Gaussian Naive Bayes can be tested with raw or standardized features, but interpretation should state which version was used

## Intended C++ Loading Contract

```txt
X = columns:
  loc
  v_g
  ev_g
  iv_g
  n
  v
  l
  d
  i
  e
  b
  t
  lOCode
  lOComment
  lOBlank
  locCodeAndComment
  uniq_Op
  uniq_Opnd
  total_Op
  total_Opnd
  branchCount

y = column:
  defects
```

---

# Dataset 3 – Wine

## Role

Primary dataset for multiclass classification.

Fallback dataset for unsupervised PCA/KMeans visualization.

## Raw Data Location

```txt
data/raw/wine/wine.data
data/raw/wine/wine.names
```

Expected raw data format:

```txt
class,alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280_od315_of_diluted_wines,proline
```

The raw `wine.data` file has no header.

## Processed Data Location

```txt
data/processed/wine.csv
```

## Task Type

Multiclass classification.

Fallback unsupervised / visualization dataset.

## Target Column

```txt
class
```

Recommended processed encoding:

```txt
raw class 1 -> 0
raw class 2 -> 1
raw class 3 -> 2
```

## Feature Columns

```txt
alcohol
malic_acid
ash
alcalinity_of_ash
magnesium
total_phenols
flavanoids
nonflavanoid_phenols
proanthocyanins
color_intensity
hue
od280_od315_of_diluted_wines
proline
```

## Final Processed CSV Schema

```csv
alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280_od315_of_diluted_wines,proline,class
```

## Rows to Reject

Reject rows where:

- the row does not have exactly 14 values
- any feature value is missing
- any feature value is NaN or infinite
- class is not `1`, `2`, or `3` in the raw file
- class is not `0`, `1`, or `2` after processing

The raw dataset reports no missing values, so row rejection should normally be minimal.

## Scaling Policy

Do not scale this dataset in the processed CSV.

Scaling should be applied inside experiment workflows where needed.

Recommended:

- standardize for `SoftmaxRegression`
- standardize for `KNNClassifier`
- standardize for `GaussianNaiveBayes` only if the experiment explicitly states it
- standardize for PCA/KMeans
- tree-based models can use raw feature values

## Intended C++ Loading Contract

For multiclass classification:

```txt
X = columns:
  alcohol
  malic_acid
  ash
  alcalinity_of_ash
  magnesium
  total_phenols
  flavanoids
  nonflavanoid_phenols
  proanthocyanins
  color_intensity
  hue
  od280_od315_of_diluted_wines
  proline

y = column:
  class
```

For unsupervised PCA/KMeans:

```txt
X = same feature columns
```

The `class` column may be kept only for qualitative comparison in exported notebook files.

The `class` column must not be used during unsupervised fitting.

---

## Processed Dataset Output Checklist

After processing, the following files should exist:

```txt
data/processed/stock_ohlcv_engineered.csv
data/processed/nasa_kc1_software_defects.csv
data/processed/wine.csv
```

Optional reference/debug file:

```txt
data/processed/stock_ohlcv_engineered_reference.csv
```

---

## Verification Commands

From the repository root:

```bash
find data/processed -maxdepth 1 -type f | sort
```

Expected required files:

```txt
data/processed/nasa_kc1_software_defects.csv
data/processed/stock_ohlcv_engineered.csv
data/processed/wine.csv
```

Inspect headers:

```bash
head -n 1 data/processed/stock_ohlcv_engineered.csv
head -n 1 data/processed/nasa_kc1_software_defects.csv
head -n 1 data/processed/wine.csv
```

Expected headers:

```txt
return_1d,return_5d,volatility_5d,range_pct,volume_change_1d,target_next_return
```

```txt
loc,v_g,ev_g,iv_g,n,v,l,d,i,e,b,t,lOCode,lOComment,lOBlank,locCodeAndComment,uniq_Op,uniq_Opnd,total_Op,total_Opnd,branchCount,defects
```

```txt
alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280_od315_of_diluted_wines,proline,class
```

---

## Checklist Item Addressed

This document defines the processed dataset schema needed before implementing dataset conversion and loading utilities.

It supports the following Phase 11 checklist items:

```md
- [ ] Implement or standardize CSV dataset loading for real workflows
- [ ] Implement dataset-to-`Matrix` / `Vector` conversion utilities where needed
- [ ] Define handling for:
  - headers
  - selected feature columns
  - target column
  - numeric-only datasets
  - missing-value rejection or simple preprocessing
```

After this document is added, the next concrete step is to create the processed CSV files from the raw datasets.
