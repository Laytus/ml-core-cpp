# Practical Output Schemas – Phase 11 Practical Workflows

## Purpose

This document defines the common output schemas for practical ML workflows.

All Phase 11 practical exercise outputs should be written under:

```txt
outputs/practical-exercises/
```

The goal is to make outputs:

- consistent across models
- easy to compare
- easy to inspect manually
- easy to load from Python/Jupyter notebooks
- stable enough for future real-dataset exercises

This document freezes the baseline CSV schemas for:

- metrics
- predictions
- probabilities
- decision scores
- loss histories
- hyperparameter sweeps
- dimensionality-reduction projections
- clustering assignments

---

## Output Root

All practical exercise outputs should use this root:

```txt
outputs/practical-exercises/
```

Recommended folder structure:

```txt
outputs/practical-exercises/
├── regression/
├── binary-classification/
├── multiclass-classification/
└── unsupervised/
```

Each workflow folder may contain one or more CSV files following the schemas defined below.

---

## General CSV Rules

All output CSV files must follow these rules:

- Use one header row.
- Use comma-separated values.
- Use UTF-8 text.
- Prefer lowercase `snake_case` column names.
- Use stable column order.
- Use numeric values when possible.
- Use empty fields for unavailable values instead of custom strings such as `N/A`.
- Do not include formatting symbols such as `%`.
- Do not include human-only prose in machine-readable CSV files.
- Keep output files readable from Python with `pandas.read_csv(...)`.

Example:

```python
import pandas as pd

metrics = pd.read_csv("outputs/practical-exercises/regression/metrics.csv")
```

---

## Naming Rules

Use workflow folders:

```txt
regression/
binary-classification/
multiclass-classification/
unsupervised/
```

Use canonical file names:

```txt
metrics.csv
predictions.csv
probabilities.csv
decision_scores.csv
loss_history.csv
hyperparameter_sweep.csv
projections.csv
clustering_assignments.csv
```

If multiple experiments are run for the same workflow, either:

1. append to the same canonical file using `run_id`, or
2. use a subfolder per experiment.

Preferred simple structure for Phase 11:

```txt
outputs/practical-exercises/regression/metrics.csv
outputs/practical-exercises/regression/predictions.csv
```

Preferred expanded structure for later experiments:

```txt
outputs/practical-exercises/regression/stock_ohlcv_linear_models/metrics.csv
outputs/practical-exercises/regression/stock_ohlcv_tree_models/metrics.csv
```

Both are acceptable as long as the CSV schema remains stable.

---

## Common Identifier Columns

Several schemas reuse these columns.

### `run_id`

A stable identifier for one experiment run.

Examples:

```txt
run_001
linear_regression_baseline
logistic_regression_lr_0_01
gb_regressor_depth_3_estimators_100
```

### `workflow`

One of:

```txt
regression
binary_classification
multiclass_classification
unsupervised
```

Use underscores inside CSV values even though folder names use hyphens.

### `dataset`

Short dataset identifier.

Recommended initial values:

```txt
stock_ohlcv_engineered
nasa_kc1_software_defects
wine
```

### `model`

Model name used in the experiment.

Examples:

```txt
LinearRegression
DecisionTreeRegressor
GradientBoostingRegressor
LogisticRegression
LinearSVM
GaussianNaiveBayes
DecisionTreeClassifier
RandomForestClassifier
SoftmaxRegression
KNNClassifier
PCA
KMeans
```

### `split`

One of:

```txt
train
validation
test
full
```

Use `full` only when the workflow intentionally uses the full dataset, such as exploratory PCA/KMeans visualization.

### `row_id`

Zero-based row index within the exported split or output file.

This is not necessarily the original raw dataset row.

For output files that need to map back to original references, use an additional reference file or optional columns such as `source_row_id`, `ticker`, or `date`.

---

# 1. Metrics Schema

## File Name

```txt
metrics.csv
```

## Purpose

Stores scalar evaluation metrics.

Use this file for:

- regression metrics
- binary classification metrics
- multiclass classification metrics
- clustering metrics, if available
- dimensionality-reduction summary metrics, if available

## Schema

```csv
run_id,workflow,dataset,model,split,metric,value
```

## Column Definitions

| Column | Type | Required | Description |
|---|---:|---:|---|
| `run_id` | string | yes | Experiment run identifier |
| `workflow` | string | yes | Workflow type |
| `dataset` | string | yes | Dataset identifier |
| `model` | string | yes | Model name |
| `split` | string | yes | Dataset split |
| `metric` | string | yes | Metric name |
| `value` | numeric | yes | Metric value |

## Examples

Regression:

```csv
run_id,workflow,dataset,model,split,metric,value
linear_regression_baseline,regression,stock_ohlcv_engineered,LinearRegression,test,mse,0.000123
linear_regression_baseline,regression,stock_ohlcv_engineered,LinearRegression,test,mae,0.007421
linear_regression_baseline,regression,stock_ohlcv_engineered,LinearRegression,test,r2,0.0312
```

Binary classification:

```csv
run_id,workflow,dataset,model,split,metric,value
logistic_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,accuracy,0.8500
logistic_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,precision,0.4200
logistic_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,recall,0.3100
logistic_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,f1,0.3560
```

Unsupervised:

```csv
run_id,workflow,dataset,model,split,metric,value
kmeans_baseline,unsupervised,stock_ohlcv_engineered,KMeans,full,inertia,123.45
pca_baseline,unsupervised,wine,PCA,full,explained_variance_ratio_1,0.3619
pca_baseline,unsupervised,wine,PCA,full,explained_variance_ratio_2,0.1921
```

---

# 2. Predictions Schema

## File Name

```txt
predictions.csv
```

## Purpose

Stores model predictions for supervised workflows.

Use this file for:

- regression predictions
- binary classification predicted labels
- multiclass classification predicted labels

## Regression Schema

```csv
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,error
```

Where:

```txt
error = y_pred - y_true
```

## Classification Schema

```csv
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,correct
```

Where:

```txt
correct = 1 if y_true == y_pred else 0
```

## Column Definitions

| Column | Type | Required | Description |
|---|---:|---:|---|
| `run_id` | string | yes | Experiment run identifier |
| `row_id` | integer | yes | Zero-based output row index |
| `workflow` | string | yes | Workflow type |
| `dataset` | string | yes | Dataset identifier |
| `model` | string | yes | Model name |
| `split` | string | yes | Dataset split |
| `y_true` | numeric | yes | Ground-truth target |
| `y_pred` | numeric | yes | Model prediction |
| `error` | numeric | regression only | `y_pred - y_true` |
| `correct` | integer | classification only | 1 if correct, else 0 |

## Examples

Regression:

```csv
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,error
linear_regression_baseline,0,regression,stock_ohlcv_engineered,LinearRegression,test,0.012,0.009,-0.003
linear_regression_baseline,1,regression,stock_ohlcv_engineered,LinearRegression,test,-0.006,-0.004,0.002
```

Classification:

```csv
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,correct
logistic_baseline,0,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,1,0,0
logistic_baseline,1,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,0,0,1
```

---

# 3. Probabilities Schema

## File Name

```txt
probabilities.csv
```

## Purpose

Stores predicted class probabilities for models that expose probability outputs.

Use this file for:

- binary probabilities
- multiclass probabilities

Do not create this file for models that do not expose probabilities unless a calibrated probability workflow is explicitly implemented.

## Binary Classification Schema

```csv
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1
```

## Multiclass Classification Schema

```csv
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1,probability_class_2
```

For datasets with more than three classes, extend the schema as:

```txt
probability_class_0
probability_class_1
...
probability_class_k
```

## Example

```csv
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1
logistic_baseline,0,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,1,0.266,0.734
```

Multiclass:

```csv
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1,probability_class_2
softmax_baseline,0,multiclass_classification,wine,SoftmaxRegression,test,2,0.020,0.140,0.840
```

---

# 4. Decision Scores Schema

## File Name

```txt
decision_scores.csv
```

## Purpose

Stores raw decision scores for models that expose margins or score-like outputs.

Use this file for:

- SVM decision scores
- logistic regression logits, if exposed
- perceptron raw scores, if used
- any binary margin-based classifier

Do not create this file for models where decision scores are not meaningful or not exposed.

## Binary Classification Schema

```csv
run_id,row_id,workflow,dataset,model,split,y_true,decision_score
```

## Optional Multiclass Extension

For multiclass margin-style models:

```csv
run_id,row_id,workflow,dataset,model,split,y_true,score_class_0,score_class_1,score_class_2
```

For more classes, extend with:

```txt
score_class_k
```

## Example

```csv
run_id,row_id,workflow,dataset,model,split,y_true,decision_score
linear_svm_baseline,0,binary_classification,nasa_kc1_software_defects,LinearSVM,test,1,1.472
```

---

# 5. Loss History Schema

## File Name

```txt
loss_history.csv
```

## Purpose

Stores training loss over iterations or epochs.

Use this file for:

- gradient descent linear regression
- logistic regression
- softmax regression
- linear SVM, if loss history is exposed
- TinyMLPBinaryClassifier
- gradient boosting staged loss, if exposed

## Schema

```csv
run_id,workflow,dataset,model,split,iteration,loss
```

## Column Definitions

| Column | Type | Required | Description |
|---|---:|---:|---|
| `run_id` | string | yes | Experiment run identifier |
| `workflow` | string | yes | Workflow type |
| `dataset` | string | yes | Dataset identifier |
| `model` | string | yes | Model name |
| `split` | string | yes | Usually `train`, optionally `validation` |
| `iteration` | integer | yes | Iteration or epoch index |
| `loss` | numeric | yes | Loss value |

## Example

```csv
run_id,workflow,dataset,model,split,iteration,loss
logistic_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,train,0,0.693
logistic_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,train,1,0.681
```

---

# 6. Hyperparameter Sweep Schema

## File Name

```txt
hyperparameter_sweep.csv
```

## Purpose

Stores results from hyperparameter sweeps.

This schema uses one row per:

```txt
run_id + parameter + metric
```

This makes the file easy to append and easy to group from Python.

## Schema

```csv
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

## Example

Single-parameter sweep:

```csv
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
logistic_lr_0_01,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,learning_rate,0.01,f1,0.356
logistic_lr_0_05,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,learning_rate,0.05,f1,0.371
```

Multiple parameters in one run:

```csv
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
gb_depth3_lr005_est100,regression,stock_ohlcv_engineered,GradientBoostingRegressor,test,max_depth,3,mse,0.00012
gb_depth3_lr005_est100,regression,stock_ohlcv_engineered,GradientBoostingRegressor,test,learning_rate,0.05,mse,0.00012
gb_depth3_lr005_est100,regression,stock_ohlcv_engineered,GradientBoostingRegressor,test,n_estimators,100,mse,0.00012
```

In Python, this can be pivoted by `run_id` to reconstruct full parameter combinations.

---

# 7. Dimensionality-Reduction Projections Schema

## File Name

```txt
projections.csv
```

## Purpose

Stores low-dimensional projections from PCA or future dimensionality-reduction methods.

Use this file for notebook visualization.

## 2D Schema

```csv
run_id,row_id,workflow,dataset,method,split,component_1,component_2,label_reference
```

## Optional 3D Schema

```csv
run_id,row_id,workflow,dataset,method,split,component_1,component_2,component_3,label_reference
```

## Column Definitions

| Column | Type | Required | Description |
|---|---:|---:|---|
| `run_id` | string | yes | Experiment run identifier |
| `row_id` | integer | yes | Zero-based output row index |
| `workflow` | string | yes | Usually `unsupervised` |
| `dataset` | string | yes | Dataset identifier |
| `method` | string | yes | Projection method, e.g. `PCA` |
| `split` | string | yes | Usually `full`, optionally `train`/`test` |
| `component_1` | numeric | yes | First projected coordinate |
| `component_2` | numeric | yes | Second projected coordinate |
| `component_3` | numeric | optional | Third projected coordinate |
| `label_reference` | numeric/string/empty | optional | Known label used only for visualization |

## Examples

Wine PCA with known class label:

```csv
run_id,row_id,workflow,dataset,method,split,component_1,component_2,label_reference
pca_wine_2d,0,unsupervised,wine,PCA,full,-1.230,0.480,2
```

Stock PCA without class label:

```csv
run_id,row_id,workflow,dataset,method,split,component_1,component_2,label_reference
pca_stock_2d,0,unsupervised,stock_ohlcv_engineered,PCA,full,-0.340,1.270,
```

---

# 8. Clustering Assignments Schema

## File Name

```txt
clustering_assignments.csv
```

## Purpose

Stores cluster assignments from KMeans or future clustering methods.

Use this file for notebook visualization and cluster analysis.

## Schema

```csv
run_id,row_id,workflow,dataset,method,split,cluster,label_reference
```

## Column Definitions

| Column | Type | Required | Description |
|---|---:|---:|---|
| `run_id` | string | yes | Experiment run identifier |
| `row_id` | integer | yes | Zero-based output row index |
| `workflow` | string | yes | Usually `unsupervised` |
| `dataset` | string | yes | Dataset identifier |
| `method` | string | yes | Clustering method, e.g. `KMeans` |
| `split` | string | yes | Usually `full`, optionally `train`/`test` |
| `cluster` | integer | yes | Assigned cluster index |
| `label_reference` | numeric/string/empty | optional | Known label used only for qualitative comparison |

## Examples

Wine with class label for qualitative comparison:

```csv
run_id,row_id,workflow,dataset,method,split,cluster,label_reference
kmeans_wine_k3,0,unsupervised,wine,KMeans,full,1,2
```

Stock without class label:

```csv
run_id,row_id,workflow,dataset,method,split,cluster,label_reference
kmeans_stock_k4,0,unsupervised,stock_ohlcv_engineered,KMeans,full,3,
```

---

## Python Notebook Readability

All schemas are designed to be read directly with Pandas:

```python
import pandas as pd

metrics = pd.read_csv("outputs/practical-exercises/regression/metrics.csv")
predictions = pd.read_csv("outputs/practical-exercises/regression/predictions.csv")
```

Useful notebook operations:

```python
metrics.pivot_table(
    index=["dataset", "model"],
    columns="metric",
    values="value"
)
```

```python
predictions.head()
```

```python
projections.plot.scatter(
    x="component_1",
    y="component_2"
)
```

Guidelines for notebook compatibility:

- Avoid nested JSON inside CSV cells.
- Avoid semicolon-delimited subfields.
- Prefer one observation per row.
- Use `run_id` to group rows belonging to the same experiment.
- Use empty cells for unavailable values.
- Keep labels numeric when possible.
- Keep file names stable across runs.

---

## Commit Policy

Recommended to commit:

```txt
outputs/practical-exercises/**/metrics.csv
outputs/practical-exercises/**/summary.csv
small representative prediction/projection files
```

Recommended not to commit:

```txt
huge prediction dumps
large sweep outputs
temporary debug exports
```

If an output file becomes large, store only a small representative sample or summary in Git.

---

## Checklist Item Addressed

This document completes the schema-design part of the Phase 11 practical workflow output system:

```md
- [x] Define common output schemas for:
  - metrics
  - predictions
  - probabilities
  - decision scores
  - loss histories
  - hyperparameter sweeps
  - dimensionality-reduction projections
  - clustering assignments
```

It also supports:

```md
- [ ] Ensure experiment outputs are easy to read from Python notebooks
```

That item should be marked complete only after at least one generated output file is successfully read from a notebook or Python sanity script.
