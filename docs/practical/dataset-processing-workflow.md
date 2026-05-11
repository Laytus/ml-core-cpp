# Dataset Processing Workflow – Phase 11 Practical Workflows

## Purpose

This document defines the reusable dataset processing workflow for Phase 11.

The goal is to avoid hardcoded dataset conversion logic.

A new dataset should be processable by adding a processing specification file that declares:

- input format
- input file path or paths
- output file path
- column names
- selected feature columns
- target column
- label encodings
- validation rules
- optional reusable feature-engineering processor

The processing code should remain generic.

Dataset-specific decisions should live in JSON specs under:

```txt
data/metadata/processing_specs/
```

---

## Core Principle

Do not build one converter per dataset.

Avoid this:

```txt
convert_wine()
convert_kc1()
convert_stock_ohlcv()
```

Prefer this:

```txt
raw dataset + processing spec
        ↓
generic processor
        ↓
processed numeric CSV
        ↓
generic C++ CSV loader
        ↓
Matrix X / Vector y
```

The processor should not need to know what Wine, KC1, or AAPL are.

It should only read the processing spec and execute the declared operations.

---

## Folder Structure

```txt
data/
├── raw/
├── processed/
└── metadata/
    ├── stock_ohlcv_engineered.md
    ├── nasa_kc1_software_defects.md
    ├── wine.md
    └── processing_specs/
        ├── stock_ohlcv_engineered.json
        ├── nasa_kc1_software_defects.json
        └── wine.json
```

The Markdown metadata files explain datasets for humans.

The JSON processing specs define machine-readable processing instructions.

---

## Scope of the Processing System

The processing system should support clean tabular datasets.

It is not intended to solve every possible raw-data problem automatically.

A compatible dataset should be processable when its spec declares:

- how to read the raw data
- how to name the columns
- how to select feature columns
- how to identify the target column
- how to encode labels
- how to reject invalid rows
- how to write the processed CSV

---

## Supported Input Formats

Initial supported formats:

```txt
csv
delimited
arff
csv_collection
```

### `csv`

Use for a single comma-separated file.

Typical case:

```json
{
  "format": "csv",
  "path": "data/raw/example/example.csv",
  "has_header": true
}
```

### `delimited`

Use for text-like files with a known delimiter.

Typical case:

```json
{
  "format": "delimited",
  "path": "data/raw/wine/wine.data",
  "delimiter": ",",
  "has_header": false,
  "columns": ["class", "feature_1", "feature_2"]
}
```

### `arff`

Use for Weka/OpenML/PROMISE-style ARFF files.

Typical case:

```json
{
  "format": "arff",
  "path": "data/raw/nasa_kc1_software_defects/kc1.arff"
}
```

### `csv_collection`

Use for multiple CSV files with the same schema.

Typical case:

```json
{
  "format": "csv_collection",
  "paths": [
    "data/raw/stock_ohlcv/aapl_us_d.csv",
    "data/raw/stock_ohlcv/msft_us_d.csv"
  ],
  "has_header": true
}
```

---

## Processing Spec Contract

Every processing spec should follow this high-level structure:

```json
{
  "dataset_name": "dataset_name",
  "input": {},
  "output": {},
  "column_renames": {},
  "processor": {},
  "features": [],
  "target": {},
  "validation": {}
}
```

Not every field is required for every dataset.

For example:

- `column_renames` is optional.
- `processor` is optional for normal tabular conversion.
- `target` is optional for pure unsupervised datasets.

---

## Generic Processing Steps

The processing script should follow this order:

1. Load the JSON processing spec.
2. Validate required spec fields.
3. Read the raw dataset according to `input.format`.
4. Assign column names if the input has no header.
5. Apply `column_renames`.
6. Apply the declared `processor`, if any.
7. Apply target encoding, if any.
8. Select feature columns and target column.
9. Validate numeric feature columns.
10. Validate numeric target column when required.
11. Reject rows with missing, NaN, or infinite values.
12. Write the processed CSV to `output.path`.
13. Optionally write a reference/debug CSV to `output.reference_path`.

---

## Generic Tabular Processing

For normal supervised tabular datasets, the processor should:

- read the raw table
- rename columns if needed
- encode target labels if needed
- select the declared features
- append the target column as the final column
- write a numeric CSV

Example output order:

```txt
feature_1,feature_2,feature_3,target
```

The output column order should follow the order declared in the spec.

---

## Reusable Feature-Engineering Processors

Some datasets require more than column selection.

For example, OHLCV market data needs time-series feature engineering.

This should not be hardcoded for one dataset.

Instead, use named reusable processors.

Initial supported processor:

```txt
ohlcv_feature_engineering
```

This processor should work for any collection of OHLCV files that provide:

- date column
- open column
- high column
- low column
- close column
- volume column

It should produce:

- return features
- volatility features
- range features
- volume-change features
- next-period return target

---

## OHLCV Feature Engineering Contract

For each ticker/file, rows should be sorted by date ascending before feature generation.

Recommended features:

```txt
return_1d
return_5d
volatility_5d
range_pct
volume_change_1d
```

Recommended target:

```txt
target_next_return
```

Definitions:

```txt
return_1d        = (Close[t] - Close[t - 1]) / Close[t - 1]
return_5d        = (Close[t] - Close[t - 5]) / Close[t - 5]
volatility_5d    = standard deviation of daily returns over the last 5 available returns
range_pct        = (High[t] - Low[t]) / Close[t]
volume_change_1d = (Volume[t] - Volume[t - 1]) / Volume[t - 1]
target_next_return = (Close[t + 1] - Close[t]) / Close[t]
```

Rows must be rejected when there is not enough lookback or no future target.

The processor may write an optional reference file containing `ticker` and `date` for debugging and notebooks.

The model-ready output should remain numeric-only.

---

## Validation Rules

The processing specs should declare validation behavior.

Recommended validation fields:

```json
{
  "missing_values": ["", "?", "NA", "NaN", "nan"],
  "reject_missing": true,
  "require_numeric_features": true,
  "require_numeric_target": true,
  "reject_infinite": true
}
```

Dataset-specific validation fields may also exist.

Examples:

```json
{
  "reject_non_positive_close": true,
  "reject_non_positive_previous_volume": true
}
```

The processor should fail loudly if:

- required columns are missing
- declared features are not found
- target column is not found
- a row has the wrong number of columns
- values cannot be converted to numeric
- encoded labels are not covered by the target encoding map

---

## Scaling Policy

The processing workflow must not apply train-aware scaling.

Processed CSV files should contain cleaned numeric values, not standardized values.

Scaling must happen later inside C++ experiment workflows, after train/test splitting.

This prevents data leakage.

Allowed during processing:

- deterministic label encoding
- deterministic column renaming
- deterministic feature engineering from each row or valid past context
- deterministic row rejection

Not allowed during processing:

- full-dataset standardization
- full-dataset min-max scaling
- PCA projection
- feature selection based on model performance
- target leakage from future values

---

## Output Policy

Required processed outputs:

```txt
data/processed/stock_ohlcv_engineered.csv
data/processed/nasa_kc1_software_defects.csv
data/processed/wine.csv
```

Optional reference/debug outputs:

```txt
data/processed/stock_ohlcv_engineered_reference.csv
```

Reference files may include non-model columns such as:

```txt
ticker
date
```

The generic C++ model loader should not depend on reference files.

---

## Recommended Script

The recommended conversion script path is:

```txt
scripts/prepare_dataset.py
```

Recommended usage:

```bash
python3 scripts/prepare_dataset.py data/metadata/processing_specs/wine.json
python3 scripts/prepare_dataset.py data/metadata/processing_specs/nasa_kc1_software_defects.json
python3 scripts/prepare_dataset.py data/metadata/processing_specs/stock_ohlcv_engineered.json
```

The script should be generic and spec-driven.

It should not contain dataset-specific branches such as:

```python
if dataset_name == "wine":
    ...
```

Acceptable reusable branches are format/processor based:

```python
if input_format == "arff":
    ...

if processor_type == "ohlcv_feature_engineering":
    ...
```

This distinction keeps the workflow reusable.

---

## Verification Commands

After running the conversion script, verify:

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

## Checklist Items Supported

This document supports the following Phase 11 checklist items:

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

After this document and the JSON specs are added, the next implementation step is:

```txt
scripts/prepare_dataset.py
```
