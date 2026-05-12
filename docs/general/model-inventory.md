# Model Inventory – ML Core

## Purpose

This document is the final model inventory for `ml-core-cpp`.

It summarizes what the project contains, how deeply each model was implemented, where each model is used, and which docs explain its theory, usage, and math mapping.

Implementation depth follows the project convention:

```txt
Level A — full implementation
Level B — partial implementation + experiments
Level C — theory + small demo only
```

## Summary

The project now contains reusable implementations for the main classical ML families needed before Deep Learning:

```txt
linear regression
binary linear classification
multiclass linear classification
margin-based classification
distance-based classification
decision trees
tree ensembles
unsupervised learning
probabilistic classification
minimal neural-network bridge
```

Most practical model families are implemented at Level A or B+ and have:

```txt
public C++ API
sanity tests / experiments
practical workflow coverage
usage documentation
math-map documentation
```

---

## Supervised regression models

| Model | Task type | Depth | Main public methods | Practical workflow coverage | Status |
|---|---|---:|---|---|---|
| `LinearRegression` | Regression | A | `fit`, `predict`, `score_mse`, `weights`, `bias`, `training_history` | Phase 11 regression workflow | Complete |
| `RidgeRegression` behavior through `LinearRegression` regularization | Regression | A | `LinearRegressionOptions::regularization`, `RegularizationConfig::ridge` | Phase 11 regression workflow | Complete |
| `DecisionTreeRegressor` | Regression | A | `fit`, `predict`, `is_fitted`, `options` | Phase 11 regression workflow | Complete |
| `GradientBoostingRegressor` | Regression | A | `fit`, `predict`, `training_loss_history`, `initial_prediction`, `num_trees` | Phase 11 regression workflow and sweep | Complete |

### Regression docs

| Model | Theory doc | Usage doc | Math-map doc |
|---|---|---|---|
| `LinearRegression` | `docs/theory/linear-models.md` | `docs/practical/models/linear-regression-usage.md` | `docs/practical/math-maps/linear-regression-math-map.md` |
| Ridge behavior | `docs/theory/linear-models.md` | `docs/practical/models/linear-regression-usage.md` | `docs/practical/math-maps/linear-regression-math-map.md` |
| `DecisionTreeRegressor` | `docs/theory/trees-and-ensembles.md` | `docs/practical/models/decision-tree-usage.md` | `docs/practical/math-maps/decision-tree-math-map.md` |
| `GradientBoostingRegressor` | `docs/theory/tree-ensembles.md` | `docs/practical/models/gradient-boosting-regressor-usage.md` | `docs/practical/math-maps/gradient-boosting-regressor-math-map.md` |

---

## Binary classification models

| Model | Task type | Depth | Main public methods | Practical workflow coverage | Status |
|---|---|---:|---|---|---|
| `LogisticRegression` | Binary classification | A | `fit`, `logits`, `predict_proba`, `predict_classes`, `binary_cross_entropy`, `weights`, `bias`, `training_history` | Phase 11 binary workflow and sweep | Complete |
| `LinearSVM` | Binary classification | B+ | `fit`, `decision_function`, `predict`, `training_loss_history`, `weights`, `bias` | Phase 11 binary workflow | Complete as primal linear SVM |
| `GaussianNaiveBayes` | Binary/multiclass classification | A | `fit`, `predict`, `predict_proba`, `predict_log_proba`, `classes`, `class_priors`, `means`, `variances` | Phase 11 binary and multiclass workflows | Complete |
| `DecisionTreeClassifier` | Binary/multiclass classification | A | `fit`, `predict`, `is_fitted`, `options` | Phase 11 binary and multiclass workflows and binary sweep | Complete |
| `RandomForestClassifier` | Binary/multiclass classification | A | `fit`, `predict`, `predict_proba`, `num_classes`, `num_trees` | Phase 11 binary and multiclass workflows and binary sweep | Complete |
| `TinyMLPBinaryClassifier` | Binary classification | A as DL bridge | `fit`, `forward`, `predict_proba`, `predict`, `loss_history`, `W1`, `b1`, `W2`, `b2` | Phase 11 binary workflow and sweep | Complete as educational neural bridge |
| `Perceptron` | Binary classification | B | `fit`, `predict`, learned parameters where exposed | Phase 10 bridge experiments | Complete as educational baseline |

### Binary classification docs

| Model | Theory doc | Usage doc | Math-map doc |
|---|---|---|---|
| `LogisticRegression` | `docs/theory/logistic-regression.md` | `docs/practical/models/logistic-regression-usage.md` | `docs/practical/math-maps/logistic-regression-math-map.md` |
| `LinearSVM` | `docs/theory/distance-and-kernel-thinking.md` | `docs/practical/models/linear-svm-usage.md` | `docs/practical/math-maps/linear-svm-math-map.md` |
| `GaussianNaiveBayes` | `docs/theory/probabilistic-ml.md` | `docs/practical/models/gaussian-naive-bayes-usage.md` | `docs/practical/math-maps/gaussian-naive-bayes-math-map.md` |
| `DecisionTreeClassifier` | `docs/theory/trees-and-ensembles.md` | `docs/practical/models/decision-tree-usage.md` | `docs/practical/math-maps/decision-tree-math-map.md` |
| `RandomForestClassifier` | `docs/theory/tree-ensembles.md` | `docs/practical/models/random-forest-usage.md` | `docs/practical/math-maps/random-forest-math-map.md` |
| `TinyMLPBinaryClassifier` | `docs/theory/dl-bridge.md` | `docs/practical/models/tiny-mlp-binary-classifier-usage.md` | `docs/practical/math-maps/tiny-mlp-binary-classifier-math-map.md` |

---

## Multiclass classification models

| Model | Task type | Depth | Main public methods | Practical workflow coverage | Status |
|---|---|---:|---|---|---|
| `SoftmaxRegression` | Multiclass classification | A | `fit`, `logits`, `predict_proba`, `predict_classes`, `categorical_cross_entropy`, `weights`, `bias`, `training_history` | Phase 11 multiclass workflow | Complete |
| `KNNClassifier` | Binary/multiclass classification | A | `fit`, `predict`, `num_train_samples`, `num_features`, `options` | Phase 11 multiclass workflow and KNN sweep | Complete |
| `GaussianNaiveBayes` | Binary/multiclass classification | A | `fit`, `predict`, `predict_proba`, `predict_log_proba` | Phase 11 multiclass workflow | Complete |
| `DecisionTreeClassifier` | Binary/multiclass classification | A | `fit`, `predict` | Phase 11 multiclass workflow | Complete |
| `RandomForestClassifier` | Binary/multiclass classification | A | `fit`, `predict`, `predict_proba` | Phase 11 multiclass workflow | Complete |

### Multiclass docs

| Model | Theory doc | Usage doc | Math-map doc |
|---|---|---|---|
| `SoftmaxRegression` | `docs/theory/logistic-regression.md` | `docs/practical/models/softmax-regression-usage.md` | `docs/practical/math-maps/softmax-regression-math-map.md` |
| `KNNClassifier` | `docs/theory/distance-and-kernel-thinking.md` | `docs/practical/models/knn-classifier-usage.md` | `docs/practical/math-maps/knn-classifier-math-map.md` |
| `GaussianNaiveBayes` | `docs/theory/probabilistic-ml.md` | `docs/practical/models/gaussian-naive-bayes-usage.md` | `docs/practical/math-maps/gaussian-naive-bayes-math-map.md` |
| `DecisionTreeClassifier` | `docs/theory/trees-and-ensembles.md` | `docs/practical/models/decision-tree-usage.md` | `docs/practical/math-maps/decision-tree-math-map.md` |
| `RandomForestClassifier` | `docs/theory/tree-ensembles.md` | `docs/practical/models/random-forest-usage.md` | `docs/practical/math-maps/random-forest-math-map.md` |

---

## Unsupervised models

| Model | Task type | Depth | Main public methods | Practical workflow coverage | Status |
|---|---|---:|---|---|---|
| `PCA` | Dimensionality reduction | A | `fit`, `transform`, `fit_transform`, `inverse_transform`, `components`, `explained_variance_ratio` | Phase 11 unsupervised workflow and sweep | Complete |
| `KMeans` | Clustering | A | `fit`, `predict`, `fit_predict`, `centroids`, `labels`, `inertia`, `inertia_history` | Phase 11 unsupervised workflow and sweep | Complete |

### Unsupervised docs

| Model | Theory doc | Usage doc | Math-map doc |
|---|---|---|---|
| `PCA` | `docs/theory/unsupervised-learning.md` | `docs/practical/models/pca-usage.md` | `docs/practical/math-maps/pca-math-map.md` |
| `KMeans` | `docs/theory/unsupervised-learning.md` | `docs/practical/models/kmeans-usage.md` | `docs/practical/math-maps/kmeans-math-map.md` |

---

## Supporting model-level utilities

| Utility / module | Role | Depth | Status |
|---|---|---:|---|
| `RegularizationConfig` | Ridge/Lasso configuration for linear models | A | Complete |
| `classification_metrics` | Binary accuracy, precision, recall, F1, confusion matrix | A | Complete |
| `multiclass_metrics` | Multiclass accuracy and macro metrics | A | Complete |
| `regression_metrics` | MAE, RMSE, R2 | A | Complete |
| `math_ops` | Dot product, matrix-vector operations, residuals, MSE | A | Complete |
| `data_split` | Train/test and train/validation/test splits | A | Complete |
| `cross_validation` | K-fold evaluation utilities | A | Complete |
| `preprocessing` / `preprocessing_pipeline` | Standardization, min-max normalization, fitted-transform discipline | A | Complete |
| `distance_metrics` | Euclidean, squared Euclidean, Manhattan | A | Complete |
| `kernels` | Linear, polynomial, RBF kernel utilities | B | Complete as utilities; no full kernel SVM |
| `bootstrap` | Bootstrap sampling support for tree ensembles | A | Complete |
| `csv_dataset_loader` / workflow loading utilities | Numeric CSV loading into `Matrix` / `Vector` | A | Complete for practical workflows |
| `OutputWriter` | Structured CSV export for practical workflows | A | Complete |

---

## Implementation depth boundaries

## Fully implemented and reusable

The following are considered reusable ML Core models:

```txt
LinearRegression
LogisticRegression
SoftmaxRegression
DecisionTreeClassifier
DecisionTreeRegressor
RandomForestClassifier
GradientBoostingRegressor
KNNClassifier
LinearSVM
PCA
KMeans
GaussianNaiveBayes
TinyMLPBinaryClassifier
```

## Implemented as educational bridge / baseline

```txt
Perceptron
TinyMLPBinaryClassifier
```

`TinyMLPBinaryClassifier` is reusable in the project, but its main purpose is conceptual: it bridges classical ML to Deep Learning.

## Implemented as utilities, not full model families

```txt
kernel functions
distance metrics
bootstrap sampling
preprocessing helpers
evaluation harnesses
```

## Explicitly deferred

The following are intentionally not part of this repo’s final scope:

```txt
full kernel SVM
SMO / dual SVM optimization
GradientBoostingClassifier
advanced production-grade missing-value routing in trees
full cost-complexity pruning workflow
automatic model selection / AutoML
full deep-learning framework
```

---

## Practical workflow coverage matrix

| Model | Regression workflow | Binary workflow | Multiclass workflow | Unsupervised workflow | Hyperparameter sweep |
|---|---:|---:|---:|---:|---:|
| `LinearRegression` | Yes | No | No | No | No |
| Ridge behavior | Yes | No | No | No | No |
| `DecisionTreeRegressor` | Yes | No | No | No | No |
| `GradientBoostingRegressor` | Yes | No | No | No | Yes |
| `LogisticRegression` | No | Yes | No | No | Yes |
| `LinearSVM` | No | Yes | No | No | No |
| `GaussianNaiveBayes` | No | Yes | Yes | No | No |
| `DecisionTreeClassifier` | No | Yes | Yes | No | Yes |
| `RandomForestClassifier` | No | Yes | Yes | No | Yes |
| `TinyMLPBinaryClassifier` | No | Yes | No | No | Yes |
| `SoftmaxRegression` | No | No | Yes | No | No |
| `KNNClassifier` | No | No | Yes | No | Yes |
| `PCA` | No | No | No | Yes | Yes |
| `KMeans` | No | No | No | Yes | Yes |

---

## Final model inventory conclusion

`ml-core-cpp` now contains a broad and coherent classical ML model inventory.

The repo is not a toy implementation of one or two algorithms. It now covers the central model families needed before a serious Deep Learning project:

```txt
linear models
probabilistic classifiers
distance-based models
tree-based models
tree ensembles
unsupervised representations
optimization-aware training
minimal neural-network bridge
```

The model layer is sufficiently complete for the project’s purpose.

Future work should not expand this repo indefinitely. New deep-learning-specific work should move into a dedicated DL project.
