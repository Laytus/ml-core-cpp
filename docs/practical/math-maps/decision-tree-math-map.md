# DecisionTree Math Map

## Purpose

This document maps the public Decision Tree APIs and their important behavior to the mathematical processes implemented by the models.

The goal is to make clear which parts of the tree classes correspond to model mathematics and which parts are infrastructure.

This document covers:

```txt
DecisionTreeClassifier
DecisionTreeRegressor
```

Related files:

```txt
include/ml/trees/decision_tree.hpp
src/trees/decision_tree.cpp
include/ml/trees/regression_tree.hpp
src/trees/regression_tree.cpp
include/ml/trees/split_scoring.hpp
src/trees/split_scoring.cpp
include/ml/trees/tree_node.hpp
include/ml/trees/tree_builder.hpp
src/trees/tree_builder.cpp
```

Related practical usage doc:

```txt
docs/practical/models/decision-tree-usage.md
```

Related theory doc:

```txt
docs/theory/trees-and-ensembles.md
```

## Model idea

Decision Trees learn a sequence of feature-threshold rules.

Each internal node asks a question:

```txt
feature_j <= threshold
```

A sample moves:

```txt
left  if feature_j <= threshold
right if feature_j > threshold
```

The sample keeps moving through the tree until it reaches a leaf.

At a leaf:

```txt
DecisionTreeClassifier:
  predicts a class label

DecisionTreeRegressor:
  predicts a continuous numeric value
```

Mathematically, a decision tree partitions the feature space into regions:

```txt
R_1, R_2, ..., R_m
```

and assigns one prediction to each region.

For classification:

```txt
prediction in region R_m = majority class in that region
```

For regression:

```txt
prediction in region R_m = mean target value in that region
```

## Shared tree structure

A tree is composed of nodes.

Each node can be:

```txt
internal node:
  contains split feature and threshold

leaf node:
  contains final prediction
```

Internal node rule:

```txt
x_j <= threshold
```

Leaf prediction:

```txt
classification:
  class label

regression:
  mean target value
```

## Public API to math mapping

## `DecisionTreeOptions`

Used by:

```txt
DecisionTreeClassifier
```

Important fields:

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

### Mathematical role

These options control the recursive tree-building process.

Math meaning:

```txt
max_depth:
  maximum number of split levels

min_samples_split:
  minimum samples required to split a node

min_samples_leaf:
  minimum samples required in each child leaf

min_impurity_decrease:
  minimum required impurity reduction to accept a split

max_leaf_nodes:
  maximum number of terminal regions

max_features:
  number of features considered when searching for a split

use_balanced_class_weight:
  adjusts class contributions for imbalanced classification

random_seed:
  controls deterministic feature subsampling behavior
```

These options do not define the prediction equation directly, but they strongly affect the learned partition of feature space.

## `RegressionTreeOptions`

Used by:

```txt
DecisionTreeRegressor
```

Important fields:

```txt
max_depth
min_samples_split
min_samples_leaf
min_error_decrease
```

### Mathematical role

These options control the recursive regression-tree-building process.

Math meaning:

```txt
max_depth:
  maximum number of split levels

min_samples_split:
  minimum samples required to split a node

min_samples_leaf:
  minimum samples required in each child leaf

min_error_decrease:
  minimum required squared-error reduction to accept a split
```

## `validate_decision_tree_options`

```cpp
void validate_decision_tree_options(
    const DecisionTreeOptions& options,
    const std::string& context
);
```

### Mathematical role

Ensures the classification tree configuration is meaningful.

Examples of mathematical requirements:

```txt
max_depth >= 1
min_samples_split >= 2
min_samples_leaf >= 1
min_impurity_decrease >= 0
max_leaf_nodes, if set, must be valid
max_features, if set, must be valid
```

Invalid options would produce impossible or meaningless tree-growing behavior.

### Infrastructure role

It also provides consistent validation and clear error messages.

## `validate_regression_tree_options`

```cpp
void validate_regression_tree_options(
    const RegressionTreeOptions& options,
    const std::string& context
);
```

### Mathematical role

Ensures the regression tree configuration is meaningful.

Examples:

```txt
max_depth >= 1
min_samples_split >= 2
min_samples_leaf >= 1
min_error_decrease >= 0
```

### Infrastructure role

It provides early failure for invalid training configuration.

## `DecisionTreeClassifier::fit`

```cpp
void fit(const Matrix& X, const Vector& y);
```

### Mathematical role

`fit` builds a classification tree by recursively choosing feature-threshold splits that reduce class impurity.

At a node containing labels `y_node`, the algorithm searches over candidate splits:

```txt
feature_j <= threshold
```

Each candidate creates:

```txt
left child labels
right child labels
```

The split quality is measured by impurity reduction:

```txt
impurity_decrease = parent_impurity - weighted_child_impurity
```

For Gini impurity:

```txt
Gini(node) = 1 - sum_c p_c^2
```

where:

```txt
p_c = fraction of samples in the node belonging to class c
```

Weighted child impurity:

```txt
weighted_child_impurity =
  (n_left / n_parent) * impurity(left)
  + (n_right / n_parent) * impurity(right)
```

The best split is the split with the largest impurity decrease.

### What `fit` does mathematically

`fit` implements:

```txt
1. start with all training samples at root
2. compute parent class impurity
3. evaluate candidate feature-threshold splits
4. choose split with maximum impurity decrease
5. create left/right child nodes
6. recurse until stopping rules are reached
7. assign class prediction to each leaf
```

Leaf class prediction is usually:

```txt
majority class among samples in the leaf
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
input validation
target validation
missing-value rejection where implemented
shape checking
option validation
feature-count tracking
fitted-state tracking
tree storage
```

## `DecisionTreeRegressor::fit`

```cpp
void fit(const Matrix& X, const Vector& y);
```

### Mathematical role

`fit` builds a regression tree by recursively choosing feature-threshold splits that reduce squared prediction error.

At a node containing target values `y_node`, the leaf prediction would be:

```txt
mean(y_node)
```

The squared error of that node is:

```txt
SSE(node) = sum_i (y_i - mean(y_node))^2
```

A candidate split creates:

```txt
left targets
right targets
```

The error decrease is:

```txt
error_decrease =
  parent_squared_error
  - left_squared_error
  - right_squared_error
```

The best split is the one with the largest error decrease.

### What `fit` does mathematically

`fit` implements:

```txt
1. start with all training samples at root
2. compute parent target mean and squared error
3. evaluate candidate feature-threshold splits
4. choose split with maximum error decrease
5. create left/right child nodes
6. recurse until stopping rules are reached
7. assign mean target value to each leaf
```

Leaf regression prediction:

```txt
prediction = mean target value of samples in the leaf
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
input validation
target validation
missing-value rejection where implemented
shape checking
option validation
feature-count tracking
fitted-state tracking
tree storage
```

## `DecisionTreeClassifier::predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` applies the learned tree rules to each sample.

For one sample:

```txt
node = root

while node is not leaf:
  if x[feature_index] <= threshold:
      node = left child
  else:
      node = right child

return node.prediction
```

The final output is a class label.

### Infrastructure role

`predict` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output vector allocation
looping over samples
```

## `DecisionTreeRegressor::predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` applies the learned regression-tree rules to each sample.

The traversal rule is the same as classification:

```txt
feature_j <= threshold -> left
feature_j > threshold  -> right
```

The final output is the numeric value stored at the leaf.

For regression:

```txt
leaf prediction = mean training target in that leaf
```

### Infrastructure role

`predict` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output vector allocation
looping over samples
```

## `is_fitted`

```cpp
bool is_fitted() const;
```

### Mathematical role

None directly.

This is state-management infrastructure.

It answers whether the tree structure has been built and can be used for prediction.

## `options`

```cpp
const DecisionTreeOptions& options() const;
const RegressionTreeOptions& options() const;
```

### Mathematical role

Returns the tree-growing configuration used by the model.

These options define the constraints under which the learned recursive partition was constructed.

## Split-scoring math

## `gini_impurity`

```cpp
double gini_impurity(const Vector& y);
```

### Mathematical role

Computes class impurity:

```txt
Gini = 1 - sum_c p_c^2
```

Interpretation:

```txt
Gini = 0:
  pure node, all samples have same class

higher Gini:
  more class mixing
```

## `entropy`

```cpp
double entropy(const Vector& y);
```

### Mathematical role

Computes entropy impurity:

```txt
Entropy = -sum_c p_c log2(p_c)
```

Interpretation:

```txt
Entropy = 0:
  pure node

higher entropy:
  more uncertainty / class mixing
```

## `weighted_gini_impurity`

```cpp
double weighted_gini_impurity(
    const Vector& y,
    const Vector& sample_weight
);
```

### Mathematical role

Computes Gini impurity when samples have weights.

Weighted class probability:

```txt
p_c = weight_of_class_c / total_weight
```

Weighted Gini:

```txt
Gini = 1 - sum_c p_c^2
```

This supports class/sample weighting behavior.

## `weighted_entropy`

```cpp
double weighted_entropy(
    const Vector& y,
    const Vector& sample_weight
);
```

### Mathematical role

Computes entropy using weighted class probabilities.

Weighted entropy:

```txt
Entropy = -sum_c p_c log2(p_c)
```

where `p_c` is computed from sample weights.

## `weighted_child_impurity`

```cpp
double weighted_child_impurity(
    double left_impurity,
    double right_impurity,
    std::size_t left_count,
    std::size_t right_count
);
```

### Mathematical role

Combines child impurities according to child sizes:

```txt
weighted_child_impurity =
  (left_count / total_count) * left_impurity
  + (right_count / total_count) * right_impurity
```

This is used to evaluate how good a split is.

## `weighted_child_impurity_by_weight`

```cpp
double weighted_child_impurity_by_weight(
    double left_impurity,
    double right_impurity,
    double left_weight,
    double right_weight
);
```

### Mathematical role

Weighted version of child impurity aggregation:

```txt
weighted_child_impurity =
  (left_weight / total_weight) * left_impurity
  + (right_weight / total_weight) * right_impurity
```

This is used when sample weights are active.

## `impurity_reduction`

```cpp
double impurity_reduction(
    double parent_impurity,
    double weighted_child_impurity
);
```

### Mathematical role

Computes how much a split improves node purity:

```txt
impurity_reduction = parent_impurity - weighted_child_impurity
```

Larger is better.

A useful split should have:

```txt
impurity_reduction > 0
```

or at least exceed the configured minimum impurity decrease.

## `evaluate_candidate_threshold`

```cpp
SplitCandidate evaluate_candidate_threshold(
    const Matrix& X,
    const Vector& y,
    Eigen::Index feature_index,
    double threshold,
    const DecisionTreeOptions& options
);
```

### Mathematical role

Evaluates one possible split:

```txt
feature_index <= threshold
```

It computes:

```txt
left samples
right samples
parent impurity
left impurity
right impurity
weighted child impurity
impurity decrease
```

The result tells whether this threshold is a valid and useful split.

### Infrastructure role

Also checks constraints such as:

```txt
min_samples_leaf
valid left/right partitions
```

## `find_best_split`

```cpp
SplitCandidate find_best_split(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options
);
```

### Mathematical role

Searches over candidate features and thresholds to find the best split.

Mathematically:

```txt
best_split = argmax_split impurity_reduction(split)
```

subject to constraints:

```txt
min_samples_leaf
min_samples_split
max_features
minimum impurity decrease
```

## Regression split math

For `DecisionTreeRegressor`, split evaluation is based on squared error rather than class impurity.

At a node:

```txt
prediction = mean(y_node)
SSE = sum_i (y_i - prediction)^2
```

For a candidate split:

```txt
error_decrease =
  SSE(parent) - SSE(left) - SSE(right)
```

Larger error decrease means the split better separates target values into more homogeneous regions.

## Important tree math concepts

## Recursive partitioning

Decision trees recursively partition the feature space.

Each path from root to leaf defines a region:

```txt
R_m = {x : all split conditions along path are satisfied}
```

Prediction is constant within each region.

Classification:

```txt
class prediction is constant in region R_m
```

Regression:

```txt
numeric prediction is constant in region R_m
```

## Impurity

Impurity measures class mixing.

Pure node:

```txt
all samples have same class
impurity = 0
```

Mixed node:

```txt
multiple classes present
impurity > 0
```

## Error reduction

For regression, the analogue of impurity reduction is reduction in squared error.

A good regression split creates child nodes where target values are closer to their child means.

## Stopping rules

Tree growth stops when one or more conditions hold:

```txt
max_depth reached
too few samples to split
split would violate min_samples_leaf
best split does not improve impurity/error enough
node is already pure
max_leaf_nodes reached, if supported
```

Stopping rules are regularization mechanisms.

They prevent the tree from growing until it memorizes the training data.

## Leaf prediction

Classification leaf:

```txt
prediction = majority class
```

Regression leaf:

```txt
prediction = mean target value
```

This is why single regression trees produce piecewise-constant predictions.

## Feature subsampling

When `max_features` is used, the tree considers only a subset of features at each split.

This changes the split search from:

```txt
all features
```

to:

```txt
selected feature subset
```

This is important for Random Forest, where feature subsampling increases diversity across trees.

## Class and sample weighting

When class/sample weights are used, impurity calculations use weighted class proportions instead of raw counts.

This changes the split objective to give more influence to selected classes or samples.

For imbalanced classification, this can help the tree pay more attention to minority classes.

## Method classification

| Method / Struct | Classification math | Regression math | Split math | Optimization/search math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `DecisionTreeOptions` | Partial | No | Yes | Yes | Yes |
| `RegressionTreeOptions` | No | Partial | Yes | Yes | Yes |
| `validate_decision_tree_options` | Partial | No | Partial | Partial | Yes |
| `validate_regression_tree_options` | No | Partial | Partial | Partial | Yes |
| `DecisionTreeClassifier::fit` | Yes | No | Yes | Yes | Yes |
| `DecisionTreeRegressor::fit` | No | Yes | Yes | Yes | Yes |
| `DecisionTreeClassifier::predict` | Yes | No | Yes | No | Yes |
| `DecisionTreeRegressor::predict` | No | Yes | Yes | No | Yes |
| `gini_impurity` | Yes | No | Yes | No | No |
| `entropy` | Yes | No | Yes | No | No |
| `weighted_child_impurity` | Yes | No | Yes | No | No |
| `impurity_reduction` | Yes | No | Yes | No | No |
| `evaluate_candidate_threshold` | Yes | No | Yes | Yes | Yes |
| `find_best_split` | Yes | No | Yes | Yes | Yes |

## Output files and math meaning

## Regression outputs

Relevant file:

```txt
outputs/practical-exercises/regression/metrics.csv
```

Relevant rows:

```txt
model = DecisionTreeRegressor
metric = mse
metric = rmse
metric = mae
metric = r2
```

Math meaning:

```txt
mse:
  mean squared prediction error

rmse:
  square root of MSE

mae:
  mean absolute prediction error

r2:
  explained variance relative to a mean baseline
```

Relevant file:

```txt
outputs/practical-exercises/regression/predictions.csv
```

Relevant columns:

```txt
y_true
y_pred
error
```

Math meaning:

```txt
y_pred:
  leaf mean reached by the sample

error:
  y_pred - y_true
```

## Classification outputs

Relevant files:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/metrics.csv
```

Relevant metrics:

```txt
binary:
  accuracy
  precision
  recall
  f1

multiclass:
  accuracy
  macro_precision
  macro_recall
  macro_f1
```

Relevant predictions file:

```txt
predictions.csv
```

Relevant columns:

```txt
y_true
y_pred
correct
```

Math meaning:

```txt
y_pred:
  majority-class leaf prediction

correct:
  1 if y_pred == y_true else 0
```

## Hyperparameter sweep outputs

Relevant file:

```txt
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
```

Relevant rows:

```txt
model = DecisionTreeClassifier
param_name = max_depth
param_name = min_samples_leaf
```

Math meaning:

```txt
max_depth:
  controls tree complexity

min_samples_leaf:
  controls minimum region size

metric value:
  classification performance produced by that tree configuration
```

## Practical interpretation

Decision Trees are nonlinear, rule-based models.

They are useful because they can capture feature thresholds and interactions without requiring explicit feature engineering.

However, single trees can be unstable.

Common practical behavior:

```txt
shallow tree:
  easier to interpret
  may underfit

deep tree:
  more flexible
  may overfit

large min_samples_leaf:
  smoother, more regularized tree

small min_samples_leaf:
  more detailed regions, higher overfitting risk
```

In regression, a single tree predicts constant values in each leaf. This can produce banded predicted-vs-true plots.

In classification, a tree can capture nonlinear boundaries but may be sensitive to small changes in data.

## Summary

Decision Trees map to the following mathematical pipeline:

```txt
data matrix X
target vector y
        ↓
start at root node
        ↓
search feature-threshold splits
        ↓
score splits by impurity reduction or error reduction
        ↓
choose best valid split
        ↓
recursively partition feature space
        ↓
assign leaf predictions
        ↓
predict by traversing split rules
        ↓
evaluate with classification or regression metrics
```

The core math lives in:

```txt
fit
predict
gini_impurity
entropy
weighted_child_impurity
impurity_reduction
evaluate_candidate_threshold
find_best_split
regression squared-error split logic
leaf prediction logic
```

The supporting infrastructure lives in:

```txt
options
validation
tree node storage
is_fitted
feature-count tracking
missing-value rejection
output export workflows
```
