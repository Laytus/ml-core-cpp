# DecisionTree Usage

## What the model does

The Decision Tree models learn prediction rules by recursively splitting the feature space.

A tree asks a sequence of questions of the form:

```txt
feature_j <= threshold
```

Each split sends samples to a left or right child node. At prediction time, a sample follows the learned path until it reaches a leaf.

This project includes two decision-tree model types:

```txt
DecisionTreeClassifier
DecisionTreeRegressor
```

`DecisionTreeClassifier` predicts class labels.

`DecisionTreeRegressor` predicts continuous numeric values.

In practical terms, decision trees are useful when you want:

```txt
nonlinear tabular models
interpretable rule-based behavior
models that can capture feature thresholds
strong baselines before ensemble methods
building blocks for RandomForestClassifier and GradientBoostingRegressor
```

## Supported task types

### `DecisionTreeClassifier`

Supports:

```txt
classification
```

It can be used for binary or multiclass classification, as long as class labels are numeric.

### `DecisionTreeRegressor`

Supports:

```txt
regression
```

It predicts continuous target values.

## Expected input format

Both tree models expect:

```cpp
Matrix X;
Vector y;
```

where:

```txt
X.rows() == y.size()
X.cols() == number of numeric features
```

Typical shapes:

```txt
X: n_samples x n_features
y: n_samples
```

Classification target example:

```cpp
ml::Matrix X(5, 2);
X << 1.0, 2.0,
     1.5, 1.8,
     5.0, 8.0,
     6.0, 9.0,
     9.0, 1.0;

ml::Vector y(5);
y << 0.0, 0.0, 1.0, 1.0, 2.0;
```

Regression target example:

```cpp
ml::Matrix X(4, 2);
X << 1.0, 2.0,
     2.0, 3.0,
     3.0, 4.0,
     4.0, 5.0;

ml::Vector y(4);
y << 10.0, 15.0, 21.0, 28.0;
```

For real dataset workflows, use the CSV loader to create `X` and `y` from a processed CSV file.

## Target encoding

### Classification

Targets should be numeric class labels.

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

For multiclass classification, labels should also be integer-valued.

### Regression

Targets should be continuous numeric values.

Examples:

```txt
price
return
score
measurement
```

## Preprocessing usually needed

Decision trees are less sensitive to feature scaling than gradient-based linear models.

A split such as:

```txt
feature_j <= threshold
```

is not affected by monotonic scaling in the same way as distance-based or gradient-based models.

However, preprocessing is still important.

Recommended preprocessing:

```txt
reject missing values or preprocess them before loading
avoid non-numeric feature columns
ensure target labels are correctly encoded
remove obvious invalid values
keep train/test split discipline
```

Feature standardization is not strictly required for decision trees, but the Phase 11 workflow may still standardize features for consistency across models.

## DecisionTreeClassifier options

`DecisionTreeClassifier` is configured through `DecisionTreeOptions`.

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

Example:

```cpp
#include "ml/trees/decision_tree.hpp"
#include "ml/trees/split_scoring.hpp"

ml::DecisionTreeOptions options;
options.max_depth = 4;
options.min_samples_split = 2;
options.min_samples_leaf = 1;
options.min_impurity_decrease = 0.0;
options.use_balanced_class_weight = false;
options.random_seed = 42;

ml::DecisionTreeClassifier model(options);
```

## DecisionTreeRegressor options

`DecisionTreeRegressor` is configured through `RegressionTreeOptions`.

Important fields include:

```txt
max_depth
min_samples_split
min_samples_leaf
min_error_decrease
```

Example:

```cpp
#include "ml/trees/regression_tree.hpp"

ml::RegressionTreeOptions options;
options.max_depth = 4;
options.min_samples_split = 2;
options.min_samples_leaf = 1;
options.min_error_decrease = 0.0;

ml::DecisionTreeRegressor model(options);
```

## How to call `fit`

Classifier:

```cpp
model.fit(X_train, y_train);
```

Regressor:

```cpp
model.fit(X_train, y_train);
```

Expected behavior:

```txt
- validates non-empty matrix/vector inputs
- validates matching number of rows
- validates options
- rejects unsupported invalid/missing values
- recursively searches for useful feature thresholds
- creates leaf nodes when stopping rules are reached
- marks the model as fitted
```

For classification, splits are selected to reduce class impurity.

For regression, splits are selected to reduce prediction error.

## How to call `predict`

Classifier:

```cpp
ml::Vector predicted_classes = model.predict(X_test);
```

Regressor:

```cpp
ml::Vector predictions = model.predict(X_test);
```

Expected output:

```txt
output.size() == X_test.rows()
```

Calling `predict` before `fit` should be rejected.

## Classification split intuition

`DecisionTreeClassifier` uses class impurity ideas.

The project includes split scoring utilities such as:

```txt
gini_impurity
entropy
weighted_gini_impurity
weighted_entropy
impurity_reduction
```

A good classification split reduces class mixing.

Example intuition:

```txt
Before split:
  node contains mixed classes

After split:
  left child is mostly one class
  right child is mostly another class

Result:
  impurity decreases
```

## Regression split intuition

`DecisionTreeRegressor` predicts continuous values.

Each leaf typically predicts an average target value for the samples that reach that leaf.

A good regression split reduces target variance or squared error inside child nodes.

Example intuition:

```txt
Before split:
  node contains targets with high variation

After split:
  left and right child nodes contain targets with lower variation

Result:
  prediction error decreases
```

## How to evaluate predictions

### Classification metrics

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

### Regression metrics

For regression:

```cpp
#include "ml/common/math_ops.hpp"
#include "ml/common/regression_metrics.hpp"

double mse = ml::mean_squared_error(predictions, y_test);
double rmse = ml::root_mean_squared_error(predictions, y_test);
double mae = ml::mean_absolute_error(predictions, y_test);
double r2 = ml::r2_score(predictions, y_test);
```

## How to read outputs

In Phase 11 practical workflows, decision trees appear in regression, binary classification, and multiclass classification outputs.

### Regression outputs

`DecisionTreeRegressor` writes through:

```txt
outputs/practical-exercises/regression/
```

Main files:

```txt
metrics.csv
predictions.csv
```

Example rows:

```txt
run_id,workflow,dataset,model,split,metric,value
decision_tree_regressor_baseline,regression,stock_ohlcv_engineered,DecisionTreeRegressor,test,mse,...
decision_tree_regressor_baseline,regression,stock_ohlcv_engineered,DecisionTreeRegressor,test,rmse,...
decision_tree_regressor_baseline,regression,stock_ohlcv_engineered,DecisionTreeRegressor,test,mae,...
decision_tree_regressor_baseline,regression,stock_ohlcv_engineered,DecisionTreeRegressor,test,r2,...
```

### Binary classification outputs

`DecisionTreeClassifier` writes through:

```txt
outputs/practical-exercises/binary-classification/
```

Main files:

```txt
metrics.csv
predictions.csv
hyperparameter_sweep.csv
```

Example rows:

```txt
run_id,workflow,dataset,model,split,metric,value
decision_tree_classifier_baseline,binary_classification,nasa_kc1_software_defects,DecisionTreeClassifier,test,accuracy,...
decision_tree_classifier_baseline,binary_classification,nasa_kc1_software_defects,DecisionTreeClassifier,test,precision,...
decision_tree_classifier_baseline,binary_classification,nasa_kc1_software_defects,DecisionTreeClassifier,test,recall,...
decision_tree_classifier_baseline,binary_classification,nasa_kc1_software_defects,DecisionTreeClassifier,test,f1,...
```

### Multiclass classification outputs

`DecisionTreeClassifier` also writes through:

```txt
outputs/practical-exercises/multiclass-classification/
```

Main files:

```txt
metrics.csv
predictions.csv
```

### Predictions output

For classification:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,correct
```

For regression:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,error
```

where:

```txt
error = y_pred - y_true
```

## Hyperparameter sweep outputs

Decision tree sweeps are exported to:

```txt
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
```

The current Phase 11 sweep studies:

```txt
max_depth
min_samples_leaf
```

Schema:

```txt
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

Example:

```txt
decision_tree_depth2_leaf1,binary_classification,nasa_kc1_software_defects,DecisionTreeClassifier,test,max_depth,2,f1,...
decision_tree_depth2_leaf1,binary_classification,nasa_kc1_software_defects,DecisionTreeClassifier,test,min_samples_leaf,1,f1,...
```

## Practical workflow examples

Decision trees are used in these workflows:

```txt
include/ml/workflows/regression_comparison.hpp
src/workflows/regression_comparison.cpp

include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp

include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp

include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

Related notebooks:

```txt
notebooks/practical-workflows/01_regression_outputs.ipynb
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The notebooks visualize:

```txt
metric comparison
predictions
residuals for regression
confusion-matrix-style tables for classification
parameter-vs-metric sweep behavior
```

## Hyperparameter behavior

### `max_depth`

Controls the maximum depth of the tree.

Interpretation:

```txt
small max_depth:
  simpler tree
  less variance
  may underfit

large max_depth:
  more complex tree
  can capture detailed rules
  may overfit
```

### `min_samples_split`

Controls the minimum number of samples required to split a node.

Interpretation:

```txt
larger value:
  fewer splits
  simpler tree
  more regularization
```

### `min_samples_leaf`

Controls the minimum number of samples required in a leaf.

Interpretation:

```txt
larger value:
  smoother predictions
  fewer tiny leaves
  less overfitting
```

### `min_impurity_decrease` / `min_error_decrease`

Controls how much a split must improve the objective before being accepted.

Interpretation:

```txt
larger value:
  stricter split acceptance
  simpler tree
```

### `max_features`

For classification trees, controls how many features are considered when searching for splits.

This is especially important for Random Forest, where feature subsampling increases tree diversity.

### `use_balanced_class_weight`

For classification, this helps adjust split scoring when classes are imbalanced.

Useful when the positive or minority class is rare.

## Common mistakes to avoid

### 1. Assuming deeper is always better

Deeper trees can fit training data more closely, but they can overfit.

Use validation or sweep results to compare depths.

### 2. Ignoring class imbalance

For imbalanced classification datasets, accuracy alone can be misleading.

Use:

```txt
precision
recall
f1
```

and consider balanced class weights when appropriate.

### 3. Expecting smooth regression predictions

A single regression tree predicts constant values inside each leaf.

This often creates step-like or banded prediction behavior.

That is normal for tree regressors.

### 4. Forgetting that trees are sensitive to small data changes

Single trees can be unstable. Small changes in data can produce different splits.

Use Random Forest or Gradient Boosting when stronger ensemble behavior is needed.

### 5. Treating feature scaling as required

Trees usually do not require feature scaling in the same way as KNN, SVM, or gradient-based models.

However, this project may still standardize features for consistent workflow comparisons.

### 6. Expecting probabilities from the current `DecisionTreeClassifier`

The current practical workflow uses hard class predictions for `DecisionTreeClassifier`.

If probability estimates are needed, use:

```txt
RandomForestClassifier
GaussianNaiveBayes
LogisticRegression
TinyMLPBinaryClassifier
```

or later extend the tree API to expose leaf class distributions.

### 7. Calling `predict` before `fit`

The model must be trained before prediction.

Calling `predict` before `fit` should be rejected.

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating decision trees are:

```txt
outputs/practical-exercises/regression/metrics.csv
outputs/practical-exercises/regression/predictions.csv

outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv

outputs/practical-exercises/multiclass-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/predictions.csv
```

The main visualization notebooks are:

```txt
notebooks/practical-workflows/01_regression_outputs.ipynb
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The main workflow implementations are:

```txt
include/ml/workflows/regression_comparison.hpp
src/workflows/regression_comparison.cpp
include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp
include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

## When to use this model

Use a decision tree when:

```txt
you want a simple nonlinear tabular model
you want interpretable threshold-based behavior
you want a baseline before Random Forest or Gradient Boosting
you want to study split criteria and stopping rules
the dataset contains numeric tabular features
```

Avoid relying on a single decision tree alone when:

```txt
the dataset is noisy
you need stable predictions
you need strong generalization
you need probability estimates from the current API
you need smoother regression behavior
the tree overfits unless heavily constrained
```

For stronger tree-based performance, compare with:

```txt
RandomForestClassifier
GradientBoostingRegressor
```
