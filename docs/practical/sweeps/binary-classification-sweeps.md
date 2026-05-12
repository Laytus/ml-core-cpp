# Binary Classification Sweeps

## Setup

This document summarizes the binary classification hyperparameter sweeps on the real binary dataset:

```txt
data/processed/nasa_kc1_software_defects.csv
```

The target is:

```txt
defects
```

The sweep results were exported to:

```txt
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
```

The dataset is imbalanced, so interpretation should not rely on accuracy alone. The main metrics used were:

```txt
accuracy
precision
recall
f1
```

The sweeps covered:

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

## LogisticRegression

### What changed

The `LogisticRegression` sweep varied:

```txt
learning_rate: 0.005, 0.01
ridge_lambda: 0.0, 0.01
```

### What improved

The best F1 and recall were obtained with:

```txt
run_id: logistic_lr0.005_ridge0
learning_rate: 0.005
ridge_lambda: 0.0
f1: 0.210526315789
recall: 0.235294117647
```

The best precision and accuracy were obtained with:

```txt
run_id: logistic_lr0.01_ridge0
learning_rate: 0.01
ridge_lambda: 0.0
precision: 0.25
accuracy: 0.808333333333
```

### What worsened

The higher learning rate improved accuracy and precision, but it reduced recall and F1 compared with the lower learning rate.

The tested ridge value:

```txt
ridge_lambda: 0.01
```

did not materially improve the results in this small sweep.

### Conceptual takeaway

`LogisticRegression` illustrates a trade-off between conservative positive predictions and minority-class detection. On this imbalanced defect dataset, accuracy can improve while recall and F1 remain weak.

The main lesson is:

```txt
For imbalanced binary classification, accuracy alone can hide poor positive-class detection.
Learning rate and regularization should be evaluated using precision, recall, and F1.
```

## DecisionTreeClassifier

### What changed

The `DecisionTreeClassifier` sweep varied:

```txt
max_depth: 2, 4
min_samples_leaf: 1, 10
```

Balanced class weights were enabled in the workflow.

### What improved

The best F1 and recall were obtained with the shallow tree:

```txt
run_id: decision_tree_depth2_leaf1
max_depth: 2
min_samples_leaf: 1
f1: 0.375
recall: 0.882352941176
```

The same F1 and recall were observed with:

```txt
run_id: decision_tree_depth2_leaf10
max_depth: 2
min_samples_leaf: 10
f1: 0.375
recall: 0.882352941176
```

The best accuracy was obtained with:

```txt
run_id: decision_tree_depth4_leaf1
max_depth: 4
min_samples_leaf: 1
accuracy: 0.691666666667
```

### What worsened

Increasing depth improved accuracy but reduced recall and F1.

The shallow tree caught many positive cases, but it also generated many false positives, which limited precision:

```txt
precision: 0.238095238095
```

### Conceptual takeaway

The decision tree sweep shows how tree depth controls the balance between simple broad rules and more specific decision regions.

In this dataset, shallow trees with balanced class weights strongly favored recall. Deeper trees became more selective and improved accuracy, but at the cost of detecting fewer defective modules.

The main lesson is:

```txt
Tree depth is not simply a “more is better” parameter.
For imbalanced data, shallow trees may improve recall while deeper trees may improve accuracy.
```

## RandomForestClassifier

### What changed

The `RandomForestClassifier` sweep varied:

```txt
n_estimators: 5, 10
max_features: 4, 8
```

The forest used balanced class-weighted decision trees.

### What improved

The best overall configuration was:

```txt
run_id: random_forest_trees10_maxfeat4
n_estimators: 10
max_features: 4
f1: 0.441176470588
recall: 0.882352941176
precision: 0.294117647059
accuracy: 0.683333333333
```

This configuration improved F1, precision, and accuracy compared with the smaller forest.

### What worsened

Using more features per split did not improve the observed results in this sweep. For example:

```txt
random_forest_trees10_maxfeat8
```

had lower accuracy than:

```txt
random_forest_trees10_maxfeat4
```

The forest still had relatively low precision, meaning many predicted positives were false positives.

### Conceptual takeaway

The random forest sweep shows the effect of ensembling and feature subsampling. More trees improved stability, while limiting `max_features` helped the ensemble maintain diversity.

The main lesson is:

```txt
Random forests can improve over a single tree by averaging multiple trees.
Feature subsampling can improve diversity, but precision/recall trade-offs remain important on imbalanced data.
```

## TinyMLPBinaryClassifier

### What changed

The `TinyMLPBinaryClassifier` sweep varied:

```txt
hidden_units: 4, 8
learning_rate: 0.01, 0.05
max_epochs: 50
```

### What improved

The best observed configuration was:

```txt
run_id: tiny_mlp_h8_lr0.05_e50
hidden_units: 8
learning_rate: 0.05
max_epochs: 50
accuracy: 0.866666666667
precision: 1.0
recall: 0.0588235294118
f1: 0.111111111111
```

This configuration had the highest accuracy and precision among the tested TinyMLP settings.

### What worsened

Most TinyMLP configurations had zero recall and zero F1, meaning they failed to identify positive defect cases.

Even the best TinyMLP configuration had very low recall:

```txt
recall: 0.0588235294118
```

The high precision value should be interpreted carefully because it likely came from predicting very few positive cases.

### Conceptual takeaway

The TinyMLP sweep illustrates a common behavior on imbalanced classification problems: a model can achieve high accuracy by mostly predicting the majority class.

The main lesson is:

```txt
Neural models are not automatically better.
On imbalanced datasets, a model can achieve high accuracy while failing to detect the minority class.
Recall and F1 are essential for interpreting binary defect prediction.
```

## Overall interpretation

Across the binary classification sweeps, the most important observation is that accuracy was not enough.

For this dataset:

```txt
LogisticRegression:
  stable but low recall/F1

DecisionTreeClassifier:
  high recall with shallow class-balanced trees, but low precision

RandomForestClassifier:
  best F1 among the swept models, with high recall and improved precision

TinyMLPBinaryClassifier:
  high accuracy and precision, but very low recall
```

The best F1 in the sweep came from:

```txt
run_id: random_forest_trees10_maxfeat4
model: RandomForestClassifier
f1: 0.441176470588
```

The highest recall was shared by tree-based models:

```txt
recall: 0.882352941176
```

The highest accuracy came from TinyMLP:

```txt
accuracy: 0.866666666667
```

but this was not the best model if the goal is to detect defective modules.

## Conceptual takeaway

The KC1 software defect dataset is a practical example of imbalanced binary classification.

The main practical lesson is:

```txt
When the positive class is rare, accuracy can be misleading.
Precision, recall, and F1 must be inspected together.
Different model families optimize different trade-offs:
  - logistic models give stable linear baselines
  - trees can increase recall with class weighting
  - forests improve robustness through ensembling
  - neural models may need additional imbalance handling to detect positives
```

## Files

Input sweep output:

```txt
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
```

Generated by:

```txt
src/workflows/hyperparameter_sweeps.cpp
```

Summarized with:

```txt
scripts/summarize_hyperparameter_sweeps.py
```
