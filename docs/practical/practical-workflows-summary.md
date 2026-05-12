# Practical Workflows Summary

## Purpose

This document summarizes Phase 11 of `ml-core-cpp`: practical ML workflows with real datasets.

The goal of this phase was to prove that the implemented C++ ML Core models can be used as a small real experimentation framework:

```txt
real dataset
    ↓
CSV loading
    ↓
Matrix / Vector conversion
    ↓
preprocessing
    ↓
model training
    ↓
evaluation
    ↓
CSV export
    ↓
Python/Jupyter visualization
    ↓
interpretation
```

This phase was not intended to add many new algorithms. Instead, it focused on using, comparing, visualizing, and documenting the models already implemented in previous phases.

---

## Main workflow families

Phase 11 now contains practical workflows for:

```txt
regression
binary classification
multiclass classification
unsupervised learning
hyperparameter sweeps
```

The main output root is:

```txt
outputs/practical-exercises/
```

The main practical workflow code lives under:

```txt
include/ml/workflows/
src/workflows/
```

The sanity and workflow checks live under:

```txt
experiments/phase-11-practical-workflows/
```

The visualization notebooks live under:

```txt
notebooks/practical-workflows/
```

---

## Real datasets used

### Regression / unsupervised dataset

Dataset:

```txt
stock_ohlcv_engineered
```

Used for:

```txt
regression
unsupervised PCA/KMeans workflows
```

Regression target:

```txt
target_next_return
```

Main feature type:

```txt
engineered OHLCV-style numeric features
```

This dataset is useful because it gives a real numeric regression task, but it is also intentionally difficult: next-step financial return prediction is noisy and weakly predictable from simple features.

### Binary classification dataset

Dataset:

```txt
nasa_kc1_software_defects
```

Used for:

```txt
binary classification
binary hyperparameter sweeps
```

Target interpretation:

```txt
0 = no defect
1 = defect
```

This dataset is useful because it exposes a realistic binary classification issue: class imbalance. It showed clearly why accuracy alone is not enough.

### Multiclass classification dataset

Dataset:

```txt
wine
```

Used for:

```txt
multiclass classification
KNN hyperparameter sweep
```

Target encoding:

```txt
original labels: 1, 2, 3
workflow labels: 0, 1, 2
```

This dataset is useful because the classes are relatively clean and separable after preprocessing, so several implemented models perform strongly.

---

## Output schema conventions

Phase 11 standardized several output schemas so that C++ results can be consumed directly from Python/Jupyter.

### Metrics

Path pattern:

```txt
outputs/practical-exercises/<workflow>/metrics.csv
```

Schema:

```txt
run_id,workflow,dataset,model,split,metric,value
```

Used for:

```txt
regression metrics
binary classification metrics
multiclass classification metrics
unsupervised metrics
```

### Predictions

Path pattern:

```txt
outputs/practical-exercises/<workflow>/predictions.csv
```

Regression schema:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,error
```

Classification schema:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,correct
```

### Probabilities

Path pattern:

```txt
outputs/practical-exercises/<workflow>/probabilities.csv
```

Binary classification schema:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1
```

Multiclass classification schema:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1,probability_class_2,...
```

### Decision scores

Path pattern:

```txt
outputs/practical-exercises/binary-classification/decision_scores.csv
```

Schema:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,decision_score
```

Used by:

```txt
LogisticRegression
LinearSVM
```

### Loss history

Path pattern:

```txt
outputs/practical-exercises/<workflow>/loss_history.csv
```

Schema:

```txt
run_id,workflow,dataset,model,split,iteration,loss
```

Used by models with iterative training:

```txt
LinearRegression
RidgeRegression
GradientBoostingRegressor
LogisticRegression
LinearSVM
SoftmaxRegression
TinyMLPBinaryClassifier
```

### Hyperparameter sweeps

Path pattern:

```txt
outputs/practical-exercises/<workflow>/hyperparameter_sweep.csv
```

Schema:

```txt
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

This schema allows each run to expose one or more parameter values while repeating the corresponding metric values.

### Unsupervised projections

Path:

```txt
outputs/practical-exercises/unsupervised/projections.csv
```

Schema:

```txt
run_id,row_id,workflow,dataset,method,split,component_1,component_2,label_reference
```

Used for PCA 2D visualizations.

### Clustering assignments

Path:

```txt
outputs/practical-exercises/unsupervised/clustering_assignments.csv
```

Schema:

```txt
run_id,row_id,workflow,dataset,method,split,cluster,label_reference
```

Used for KMeans and PCA+KMeans cluster outputs.

---

## Regression workflow summary

Output folder:

```txt
outputs/practical-exercises/regression/
```

Models compared:

```txt
LinearRegression
RidgeRegression
DecisionTreeRegressor
GradientBoostingRegressor
```

Main outputs:

```txt
metrics.csv
predictions.csv
loss_history.csv
hyperparameter_sweep.csv
```

The regression workflow successfully exported real predictions, errors, metrics, and training histories.

### Main result pattern

The stock next-return target was difficult.

The models produced small numeric prediction errors in absolute terms, but R2 values were weak or negative in the tested workflow. This means the models did not clearly beat a simple mean-target baseline on the selected test split.

This result is plausible for a noisy next-day return target and should not be interpreted as an implementation failure by itself.

### Best observed model behavior

From the model-comparison outputs and sweep summaries, `GradientBoostingRegressor` was the strongest regression model among the tested configurations.

The best tested gradient boosting sweep configuration was:

```txt
run_id: gb_n5_lr0.03
n_estimators: 5
learning_rate: 0.03
mse: 0.000530764443451
rmse: 0.0230383255349
mae: 0.0170641325285
r2: -0.0114576760095
```

This shows that conservative boosting worked better than more aggressive settings in the tested sweep, but the task still remained hard.

### Practical lesson

For noisy regression tasks, especially financial next-step targets:

```txt
low training loss does not guarantee useful test performance
negative R2 can happen even with correct code
baseline comparison is essential
feature engineering matters
chronological evaluation may be more appropriate than random splitting
```

---

## Binary classification workflow summary

Output folder:

```txt
outputs/practical-exercises/binary-classification/
```

Models compared:

```txt
LogisticRegression
LinearSVM
GaussianNaiveBayes
DecisionTreeClassifier
RandomForestClassifier
TinyMLPBinaryClassifier
```

Main outputs:

```txt
metrics.csv
predictions.csv
probabilities.csv
decision_scores.csv
loss_history.csv
hyperparameter_sweep.csv
```

The binary workflow successfully exported metrics, predictions, probabilities, decision scores, and training histories.

### Main result pattern

The NASA KC1 defect dataset exposed class imbalance clearly.

Several models achieved decent accuracy, but recall and F1 were much more informative than accuracy. Some models predicted very few positive samples, producing high accuracy but weak recall.

### Model behavior observed

`GaussianNaiveBayes` performed strongly in the baseline comparison and provided probability outputs.

`RandomForestClassifier` showed useful recall/F1 behavior in the sweep, especially when increasing the number of trees and using feature subsampling.

`LogisticRegression` and `LinearSVM` provided useful linear baselines and decision scores.

`TinyMLPBinaryClassifier` demonstrated the educational neural-network bridge, but its best tested configuration still showed very low recall despite high accuracy and high precision.

Best TinyMLP sweep result:

```txt
run_id: tiny_mlp_h8_lr0.05_e50
accuracy: 0.866666666667
precision: 1.0
recall: 0.0588235294118
f1: 0.111111111111
```

This is a clear example of why accuracy alone is insufficient.

### Clearest hyperparameter effects

#### LogisticRegression

Lower learning rate gave better recall/F1 in the tested sweep, while a slightly larger learning rate improved accuracy and precision.

Best F1 observed:

```txt
run_id: logistic_lr0.005_ridge0
f1: 0.210526315789
recall: 0.235294117647
```

#### DecisionTreeClassifier

Shallow trees with depth `2` produced the strongest recall/F1 in the tested sweep, while deeper trees improved accuracy but reduced recall.

Best F1 observed:

```txt
run_id: decision_tree_depth2_leaf1
f1: 0.375
recall: 0.882352941176
```

#### RandomForestClassifier

More trees and smaller feature subsets improved the tested F1.

Best F1 observed:

```txt
run_id: random_forest_trees10_maxfeat4
f1: 0.441176470588
recall: 0.882352941176
precision: 0.294117647059
```

#### TinyMLPBinaryClassifier

More hidden units and larger learning rate improved accuracy in the tested sweep, but recall remained weak.

### Practical lesson

For imbalanced binary classification:

```txt
accuracy can be misleading
recall and F1 are essential
probability distributions should be inspected
threshold tuning is a natural next improvement
class weighting or resampling would likely help
```

---

## Multiclass classification workflow summary

Output folder:

```txt
outputs/practical-exercises/multiclass-classification/
```

Models compared:

```txt
SoftmaxRegression
KNNClassifier
DecisionTreeClassifier
RandomForestClassifier
GaussianNaiveBayes
```

Main outputs:

```txt
metrics.csv
predictions.csv
probabilities.csv
loss_history.csv
hyperparameter_sweep.csv
```

The multiclass workflow successfully exported metrics, predictions, probabilities, loss histories, and KNN sweep outputs.

### Main result pattern

The Wine dataset was clean and well-separated after preprocessing.

Most models performed strongly.

Baseline metric examples showed:

```txt
SoftmaxRegression accuracy: 0.944444444444
KNNClassifier accuracy: 0.972222222222
GaussianNaiveBayes accuracy: 0.972222222222
```

This confirms that the implemented models can work well on a clean real multiclass dataset.

### Best observed model behavior

`KNNClassifier` and `GaussianNaiveBayes` were among the strongest observed models in the exported baseline results.

`SoftmaxRegression` also performed strongly, showing that a linear multiclass decision structure works well on this dataset.

### KNN sweep behavior

The KNN sweep compared:

```txt
k = 1
k = 3
k = 5
```

All three performed strongly.

Best macro F1 observed:

```txt
run_id: knn_k5
macro_f1: 0.973909131804
```

The similar results across `k` values suggest that the Wine dataset has stable local neighborhoods after standardization.

### Practical lesson

For clean multiclass tabular data:

```txt
simple models can perform very well
standardization is important for KNN and softmax
macro metrics help check balanced class behavior
probability outputs are useful for model inspection
confusion-matrix-style visualization helps diagnose remaining mistakes
```

---

## Unsupervised workflow summary

Output folder:

```txt
outputs/practical-exercises/unsupervised/
```

Models / workflows compared:

```txt
PCA
KMeans
PCA+KMeans
```

Main outputs:

```txt
metrics.csv
projections.csv
clustering_assignments.csv
hyperparameter_sweep.csv
```

The unsupervised workflow successfully exported PCA projections, KMeans cluster assignments, and PCA+KMeans visualization-ready outputs.

### PCA behavior

The PCA workflow exported explained variance ratios.

From the sweep summary:

```txt
pca_components_2 cumulative_explained_variance_ratio: 0.629482245536
pca_components_3 cumulative_explained_variance_ratio: 0.829574241370
```

This means that using three components preserved substantially more variance than using two components.

### KMeans behavior

The KMeans workflow exported inertia and cluster assignments.

From the sweep summary:

```txt
kmeans_k2 inertia: 3161.57834899
kmeans_k4 inertia: 2373.77557625
```

The lower inertia for `k = 4` is expected because more clusters can fit the data more closely.

However, this does not automatically prove that `k = 4` is semantically better.

### PCA + KMeans behavior

The PCA+KMeans workflow made the clustering outputs visualization-ready.

This is useful for exploration:

```txt
PCA provides 2D coordinates
KMeans provides cluster IDs
the notebook overlays cluster assignments on PCA projections
```

### Practical lesson

For unsupervised workflows:

```txt
PCA is useful for visualization and variance analysis
KMeans inertia decreases as k increases
cluster IDs are arbitrary
visual clusters are exploratory, not proof of true semantic groups
feature scaling is critical
```

---

## Hyperparameter sweep summary

Hyperparameter sweeps were implemented and exported consistently.

Main files:

```txt
outputs/practical-exercises/regression/hyperparameter_sweep.csv
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
outputs/practical-exercises/multiclass-classification/hyperparameter_sweep.csv
outputs/practical-exercises/unsupervised/hyperparameter_sweep.csv
```

A helper summary script was also added:

```txt
scripts/summarize_hyperparameter_sweeps.py
```

### Covered sweep groups

Regression:

```txt
GradientBoostingRegressor:
  n_estimators
  learning_rate
```

Binary classification:

```txt
LogisticRegression:
  learning_rate
  ridge_lambda

DecisionTreeClassifier:
  max_depth
  min_samples_leaf

RandomForestClassifier:
  n_estimators
  max_features

TinyMLPBinaryClassifier:
  hidden_units
  learning_rate
  max_epochs
```

Multiclass classification:

```txt
KNNClassifier:
  k
```

Unsupervised:

```txt
PCA:
  num_components

KMeans:
  num_clusters
```

### Clearest effects observed

The clearest effects were:

```txt
GradientBoostingRegressor:
  smaller learning rate and fewer estimators worked better for the noisy regression target

DecisionTreeClassifier:
  shallower trees improved recall/F1 on the imbalanced binary task

RandomForestClassifier:
  more trees and feature subsampling improved F1 in the tested binary sweep

TinyMLPBinaryClassifier:
  accuracy improved with a larger hidden layer / learning rate, but recall remained weak

KNNClassifier:
  k had small effects because the Wine dataset was already well-separated

PCA:
  more components increased cumulative explained variance

KMeans:
  more clusters reduced inertia
```

---

## Python/Jupyter visualization summary

Phase 11 added notebooks for visual inspection of exported outputs.

Notebook folder:

```txt
notebooks/practical-workflows/
```

Implemented notebooks:

```txt
01_regression_outputs.ipynb
02_binary_classification_outputs.ipynb
03_multiclass_classification_outputs.ipynb
04_unsupervised_outputs.ipynb
05_hyperparameter_sweeps.ipynb
```

### Notebook role

The notebooks are intentionally analysis tools only.

Core model logic remains in C++.

The notebooks read CSV exports and provide:

```txt
metric comparisons
predicted-vs-true plots
residual plots
probability distributions
decision-score plots
confusion-matrix-style visualizations
PCA projections
KMeans cluster visualizations
hyperparameter sweep plots
```

### Verification scripts

Useful scripts:

```txt
scripts/verify_practical_outputs.py
scripts/summarize_hyperparameter_sweeps.py
```

`verify_practical_outputs.py` confirmed that all exported CSV files are readable from Python/Pandas and include the expected workflow families.

---

## Practical usage docs summary

Usage docs were created for the main practical models.

Path pattern:

```txt
docs/practical/models/<model-name>-usage.md
```

Created model usage docs:

```txt
docs/practical/models/linear-regression-usage.md
docs/practical/models/logistic-regression-usage.md
docs/practical/models/softmax-regression-usage.md
docs/practical/models/knn-classifier-usage.md
docs/practical/models/linear-svm-usage.md
docs/practical/models/decision-tree-usage.md
docs/practical/models/random-forest-usage.md
docs/practical/models/gradient-boosting-regressor-usage.md
docs/practical/models/gaussian-naive-bayes-usage.md
docs/practical/models/kmeans-usage.md
docs/practical/models/pca-usage.md
docs/practical/models/tiny-mlp-binary-classifier-usage.md
```

Each usage doc explains:

```txt
what the model does
what task type it supports
expected input format
preprocessing expectations
how to instantiate the model
how to call fit / predict / predict_proba / transform where applicable
how to read exported outputs
common mistakes to avoid
which workflow outputs demonstrate the model
```

---

## Math-map docs summary

Math-map docs were created for the main practical models.

Path pattern:

```txt
docs/practical/math-maps/<model-name>-math-map.md
```

Created math-map docs:

```txt
docs/practical/math-maps/linear-regression-math-map.md
docs/practical/math-maps/logistic-regression-math-map.md
docs/practical/math-maps/softmax-regression-math-map.md
docs/practical/math-maps/knn-classifier-math-map.md
docs/practical/math-maps/linear-svm-math-map.md
docs/practical/math-maps/decision-tree-math-map.md
docs/practical/math-maps/random-forest-math-map.md
docs/practical/math-maps/gradient-boosting-regressor-math-map.md
docs/practical/math-maps/gaussian-naive-bayes-math-map.md
docs/practical/math-maps/kmeans-math-map.md
docs/practical/math-maps/pca-math-map.md
docs/practical/math-maps/tiny-mlp-binary-classifier-math-map.md
```

Each math-map doc connects public methods to the mathematical operations they implement.

Examples:

```txt
fit
predict
predict_proba
decision_function
transform
loss_history
training_history
weights / bias / centroids / components
```

The docs also separate:

```txt
model math
optimization math
probability math
metric/loss math
infrastructure
```

---

## What worked best by dataset type

### Regression

Best observed practical behavior:

```txt
GradientBoostingRegressor
```

Reason:

```txt
It gave the best tested regression metrics among the compared models on the stock next-return workflow.
```

Important caveat:

```txt
The task remained hard, and R2 was still weak/negative.
```

### Binary classification

Best observed practical behavior depended on metric.

For the imbalanced KC1 dataset:

```txt
RandomForestClassifier showed strong F1/recall behavior in sweeps.
GaussianNaiveBayes was a strong simple probabilistic baseline.
LogisticRegression and LinearSVM were useful linear baselines.
TinyMLPBinaryClassifier was educational but weak on recall.
```

The key point is that no binary result should be judged by accuracy alone.

### Multiclass classification

Best observed practical behavior:

```txt
KNNClassifier
GaussianNaiveBayes
```

Both performed strongly on Wine.

`SoftmaxRegression` was also strong and useful as a linear multiclass baseline.

### Unsupervised learning

Best practical exploratory tools:

```txt
PCA for 2D projection and variance analysis
KMeans for cluster assignment
PCA+KMeans for visualization-ready clustering
```

No unsupervised result should be interpreted as ground truth without external validation.

---

## What was easy when using the C++ ML Core on real data

The following parts worked well:

```txt
numeric CSV loading into Matrix / Vector workflows
consistent train/test workflow structure
reusable model APIs
consistent output schemas
Python/Pandas reading of exported files
model comparison through shared metrics.csv format
notebook-based visualization
hyperparameter sweep export
clear separation between C++ core logic and Python visualization
```

The strongest design choice was keeping model logic in C++ and using Python only for analysis and visualization.

This preserved the learning goal of implementing ML logic directly while avoiding unnecessary plotting infrastructure in C++.

---

## What was hard when using the C++ ML Core on real data

The hardest parts were:

```txt
handling real dataset conventions consistently
keeping output schemas general enough across model families
comparing models with different capabilities
interpreting imbalanced binary classification results
making regression results meaningful on a noisy financial target
deciding what belongs in workflows versus experiments
avoiding scope creep
```

Model capability differences required special handling.

Examples:

```txt
LinearSVM has decision scores but no probabilities.
KNNClassifier has hard predictions but no probabilities.
GaussianNaiveBayes has probabilities but no loss history.
KMeans has inertia but no supervised metrics.
PCA has explained variance but no predictions.
```

This made the output writer and workflow design important.

---

## What should be improved later

These are useful future improvements, but they should not expand Phase 11 indefinitely.

### Dataset and preprocessing improvements

Potential improvements:

```txt
more robust missing-value handling
categorical encoding
train-fitted scalers with saved metadata
chronological split support for time-series-like datasets
more dataset metadata validation
larger dataset examples
```

### Model evaluation improvements

Potential improvements:

```txt
threshold tuning for imbalanced binary classification
validation-set model selection
cross-validation inside practical workflows
confusion matrix export from C++
calibration analysis for probability outputs
baseline predictors included directly in comparison files
```

### Model improvements

Potential improvements:

```txt
class weighting for LogisticRegression / LinearSVM / TinyMLP
more robust TinyMLP training options
more Random Forest sweep configurations
GradientBoostingClassifier
better KMeans initialization
multiple KMeans restarts
PCA reconstruction-error exports
```

### Output and notebook improvements

Potential improvements:

```txt
automatic plot export to image files
small HTML reports
combined model leaderboard files
automatic sweep summary CSVs
consistent README links to each notebook
```

### Project-scope warning

These improvements are valuable, but they belong to future refinement or a later applied-ML project.

The current ML Core project should not become an unlimited AutoML or production data-engineering framework.

---

## Final Phase 11 conclusion

Phase 11 successfully transformed `ml-core-cpp` from a collection of implemented algorithms into a practical experimentation system.

It now demonstrates:

```txt
real dataset loading
regression workflows
binary classification workflows
multiclass classification workflows
unsupervised workflows
model comparison
hyperparameter sweeps
structured CSV exports
Python/Jupyter visualization
model usage documentation
method-to-math mapping documentation
```

The main practical conclusion is:

```txt
ML Core can now run small real ML experiments end-to-end.
```

The main learning conclusion is:

```txt
Model implementation is only part of ML.
A serious workflow also needs data conventions, preprocessing discipline, evaluation metrics, output schemas, visualization, and interpretation.
```

This phase provides the final practical validation layer before the project wrap-up and the transition to Deep Learning.
