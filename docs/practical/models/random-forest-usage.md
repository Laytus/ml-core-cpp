# RandomForestClassifier Usage

## What the model does

`RandomForestClassifier` is an ensemble classification model built from many decision trees.

Each tree is trained on a sampled version of the training data, and predictions are combined through voting.

Conceptually:

```txt
training:
  build many decision trees
  each tree sees a bootstrap sample of the data
  each tree can use a random subset of features when splitting

prediction:
  each tree predicts a class
  the forest aggregates votes
  final prediction = majority vote
```

The model can also estimate class probabilities by averaging tree votes:

```txt
probability_class_j = votes_for_class_j / number_of_trees
```

In practical terms, `RandomForestClassifier` is useful when you want:

```txt
a stronger tree-based classifier
nonlinear tabular modeling
reduced instability compared with a single tree
class-probability estimates
feature-subsampling behavior
a robust baseline for binary or multiclass classification
```

## Supported task type

`RandomForestClassifier` supports:

```txt
classification
```

It can be used for:

```txt
binary classification
multiclass classification
```

It does not support:

```txt
regression
gradient boosting
dimensionality reduction
clustering
```

For regression tree ensembles, use:

```txt
GradientBoostingRegressor
```

in the current project.

## Expected input format

The model expects:

```cpp
Matrix X;
Vector y;
```

where:

```txt
X.rows() == y.size()
X.cols() == number of numeric features
y contains numeric class labels
```

Typical shapes:

```txt
X: n_samples x n_features
y: n_samples
```

Example:

```cpp
ml::Matrix X(6, 2);
X << 1.0, 2.0,
     1.2, 1.8,
     5.0, 8.0,
     6.0, 9.0,
     9.0, 1.0,
     9.5, 1.2;

ml::Vector y(6);
y << 0.0, 0.0, 1.0, 1.0, 2.0, 2.0;
```

For real dataset workflows, use the CSV loader to create `X` and `y` from a processed CSV file.

## Target encoding

Targets should be encoded as numeric class labels.

Recommended convention:

```txt
0
1
2
...
num_classes - 1
```

For binary classification:

```txt
0 = negative class
1 = positive class
```

For the Phase 11 NASA KC1 workflow:

```txt
0 = no defect
1 = defect
```

For the Phase 11 Wine workflow:

```txt
0, 1, 2
```

## Preprocessing usually needed

Random forests are less sensitive to feature scaling than gradient-based or distance-based models because trees split on thresholds.

However, preprocessing is still important.

Recommended preprocessing:

```txt
reject missing values or preprocess them before loading
avoid non-numeric feature columns
ensure class labels are numeric and integer-valued
check class imbalance
use consistent train/test split discipline
```

Feature standardization is not strictly required for random forests, but practical workflows may still standardize features for consistent comparisons across model families.

## How to instantiate the model

Basic model:

```cpp
#include "ml/trees/random_forest.hpp"

ml::DecisionTreeOptions tree_options;
tree_options.max_depth = 4;
tree_options.min_samples_split = 2;
tree_options.min_samples_leaf = 1;
tree_options.use_balanced_class_weight = false;
tree_options.random_seed = 42;

ml::RandomForestOptions options;
options.n_estimators = 50;
options.bootstrap = true;
options.max_features = std::nullopt;
options.random_seed = 42;
options.tree_options = tree_options;

ml::RandomForestClassifier model(options);
```

Default constructor is also available:

```cpp
ml::RandomForestClassifier model;
```

## Main options

`RandomForestOptions` contains:

```txt
n_estimators
bootstrap
max_features
random_seed
tree_options
```

### `n_estimators`

Number of trees in the forest.

Interpretation:

```txt
more trees:
  usually more stable predictions
  higher compute cost
```

### `bootstrap`

Whether each tree is trained on a bootstrap sample.

Interpretation:

```txt
bootstrap = true:
  each tree sees a sampled dataset with replacement
  increases diversity between trees

bootstrap = false:
  each tree sees the same full dataset
  less bagging diversity
```

### `max_features`

Optional number of features considered at each tree split.

Interpretation:

```txt
smaller max_features:
  increases tree diversity
  can reduce correlation between trees

larger max_features:
  each tree can choose from more features
  may reduce diversity
```

### `tree_options`

Configuration passed to each underlying `DecisionTreeClassifier`.

Important fields include:

```txt
max_depth
min_samples_split
min_samples_leaf
min_impurity_decrease
max_leaf_nodes
max_features
use_balanced_class_weight
random_seed
```

## How to call `fit`

```cpp
model.fit(X_train, y_train);
```

Expected behavior:

```txt
- validates forest options
- validates tree options
- validates non-empty matrix/vector inputs
- validates matching number of rows
- validates class labels
- estimates number of classes
- trains n_estimators decision trees
- applies bootstrap sampling if enabled
- applies feature subsampling through tree options / max_features
- marks the forest as fitted
```

## How to call `predict`

```cpp
ml::Vector predicted_classes = model.predict(X_test);
```

Expected output:

```txt
predicted_classes.size() == X_test.rows()
```

Each prediction is produced by aggregating tree votes.

Calling `predict` before `fit` should be rejected.

## How to call `predict_proba`

```cpp
ml::Matrix probabilities = model.predict_proba(X_test);
```

Expected output shape:

```txt
probabilities.rows() == X_test.rows()
probabilities.cols() == number of classes
```

Each row contains class probabilities estimated from tree votes.

For binary classification:

```txt
probability_class_0
probability_class_1
```

For multiclass classification:

```txt
probability_class_0
probability_class_1
probability_class_2
...
```

Each probability row should sum approximately to:

```txt
1.0
```

## How to inspect fitted state

After fitting:

```cpp
bool fitted = model.is_fitted();
std::size_t class_count = model.num_classes();
std::size_t tree_count = model.num_trees();
const ml::RandomForestOptions& used_options = model.options();
```

These are useful for verifying that the model was trained and that the expected forest configuration was used.

## How to evaluate predictions

For binary classification:

```cpp
#include "ml/common/classification_metrics.hpp"

double accuracy = ml::accuracy_score(predicted_classes, y_test);
double precision = ml::precision_score(predicted_classes, y_test);
double recall = ml::recall_score(predicted_classes, y_test);
double f1 = ml::f1_score(predicted_classes, y_test);
```

For multiclass classification:

```cpp
#include "ml/common/multiclass_metrics.hpp"

double accuracy = ml::multiclass_accuracy_score(
    predicted_classes,
    y_test,
    num_classes
);

double macro_precision = ml::macro_precision(
    predicted_classes,
    y_test,
    num_classes
);

double macro_recall = ml::macro_recall(
    predicted_classes,
    y_test,
    num_classes
);

double macro_f1 = ml::macro_f1(
    predicted_classes,
    y_test,
    num_classes
);
```

For imbalanced binary datasets, accuracy can be misleading. Inspect:

```txt
precision
recall
f1
```

## How to read outputs

In Phase 11 practical workflows, `RandomForestClassifier` writes results through the binary and multiclass classification comparison workflows.

Main binary classification output files:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/probabilities.csv
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
```

Main multiclass classification output files:

```txt
outputs/practical-exercises/multiclass-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/predictions.csv
outputs/practical-exercises/multiclass-classification/probabilities.csv
```

### `metrics.csv`

Binary example rows:

```txt
run_id,workflow,dataset,model,split,metric,value
random_forest_classifier_baseline,binary_classification,nasa_kc1_software_defects,RandomForestClassifier,test,accuracy,...
random_forest_classifier_baseline,binary_classification,nasa_kc1_software_defects,RandomForestClassifier,test,precision,...
random_forest_classifier_baseline,binary_classification,nasa_kc1_software_defects,RandomForestClassifier,test,recall,...
random_forest_classifier_baseline,binary_classification,nasa_kc1_software_defects,RandomForestClassifier,test,f1,...
```

Multiclass example rows:

```txt
run_id,workflow,dataset,model,split,metric,value
random_forest_classifier_baseline,multiclass_classification,wine,RandomForestClassifier,test,accuracy,...
random_forest_classifier_baseline,multiclass_classification,wine,RandomForestClassifier,test,macro_precision,...
random_forest_classifier_baseline,multiclass_classification,wine,RandomForestClassifier,test,macro_recall,...
random_forest_classifier_baseline,multiclass_classification,wine,RandomForestClassifier,test,macro_f1,...
```

### `predictions.csv`

Contains:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,correct
```

where:

```txt
correct = 1 if y_pred == y_true else 0
```

### `probabilities.csv`

For binary classification:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1
```

For multiclass classification:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1,probability_class_2
```

The number of probability columns depends on the number of classes.

### `hyperparameter_sweep.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

For Random Forest sweeps, parameters include:

```txt
n_estimators
max_features
```

Example:

```txt
random_forest_trees10_maxfeat4,binary_classification,nasa_kc1_software_defects,RandomForestClassifier,test,n_estimators,10,f1,...
random_forest_trees10_maxfeat4,binary_classification,nasa_kc1_software_defects,RandomForestClassifier,test,max_features,4,f1,...
```

## Practical workflow examples

`RandomForestClassifier` is used in:

```txt
include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp

include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp

include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

Related notebooks:

```txt
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The notebooks visualize:

```txt
metric comparison
prediction correctness
probability distributions
confusion-matrix-style tables
parameter-vs-metric sweep behavior
```

## Hyperparameter behavior

### `n_estimators`

Increasing the number of trees often improves stability.

However:

```txt
more trees:
  higher compute cost
  not always better on small sanity splits
```

### `max_features`

Feature subsampling controls tree diversity.

Interpretation:

```txt
smaller max_features:
  more diverse trees
  potentially better generalization

larger max_features:
  stronger individual trees
  potentially more correlated trees
```

### Tree depth and leaf constraints

The forest also depends on the underlying tree settings:

```txt
max_depth
min_samples_split
min_samples_leaf
```

Interpretation:

```txt
deeper trees:
  more flexible
  can overfit

larger min_samples_leaf:
  smoother trees
  more regularization
```

### Class weighting

For imbalanced binary classification, `use_balanced_class_weight` can help the forest detect minority-class samples.

In the Phase 11 KC1 workflow, class imbalance is important, so recall and F1 should be inspected alongside accuracy.

## Common mistakes to avoid

### 1. Assuming more trees always improves every metric

More trees usually reduce variance, but a small sweep may show mixed behavior depending on data, split, and tree settings.

Always compare metrics.

### 2. Ignoring feature subsampling

Random Forest is not only “many trees”.

The diversity from feature subsampling is part of what makes the ensemble useful.

### 3. Interpreting accuracy alone on imbalanced data

On imbalanced datasets, high accuracy can hide poor minority-class detection.

Inspect:

```txt
precision
recall
f1
```

### 4. Expecting probability estimates to be calibrated

`predict_proba` uses vote proportions.

These are useful probability-like estimates, but they are not guaranteed to be calibrated probabilities.

### 5. Forgetting that tree options matter

The forest is only as useful as the trees it trains.

Poor tree settings can make the forest underfit or overfit.

### 6. Treating forest predictions as interpretable as a single tree

A single tree can be interpreted as a sequence of threshold rules.

A random forest is less directly interpretable because prediction comes from many trees.

### 7. Calling prediction methods before `fit`

The model must be trained before calling:

```txt
predict
predict_proba
num_classes
num_trees
```

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating `RandomForestClassifier` are:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/probabilities.csv
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv

outputs/practical-exercises/multiclass-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/predictions.csv
outputs/practical-exercises/multiclass-classification/probabilities.csv
```

The main visualization notebooks are:

```txt
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The main workflow implementations are:

```txt
include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp
include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

## When to use this model

Use `RandomForestClassifier` when:

```txt
the task is binary or multiclass classification
you want a strong nonlinear tabular baseline
you want more stability than a single decision tree
you want vote-based class probabilities
you want to reduce tree instability through ensembling
you want to study effects of n_estimators and max_features
```

Avoid relying on it alone when:

```txt
you need very compact or highly interpretable rules
you need calibrated probabilities
prediction speed or memory is very constrained
you need regression in the current project
you need sequence/time-aware modeling
the dataset has missing values that have not been handled
```
