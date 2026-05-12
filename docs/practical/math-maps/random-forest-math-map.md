# RandomForestClassifier Math Map

## Purpose

This document maps the public `RandomForestClassifier` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/trees/random_forest.hpp
src/trees/random_forest.cpp
include/ml/trees/decision_tree.hpp
src/trees/decision_tree.cpp
include/ml/trees/bootstrap.hpp
src/trees/bootstrap.cpp
include/ml/trees/split_scoring.hpp
src/trees/split_scoring.cpp
```

Related practical usage doc:

```txt
docs/practical/models/random-forest-usage.md
```

Related theory docs:

```txt
docs/theory/trees-and-ensembles.md
docs/theory/tree-ensembles.md
```

## Model idea

`RandomForestClassifier` is an ensemble classifier made of many decision trees.

The core idea is:

```txt
train many diverse trees
combine their predictions by voting
```

Each tree is trained with some randomness, usually from:

```txt
bootstrap sampling of rows
feature subsampling during split search
random seeds controlling reproducibility
```

For one input sample `x`, each tree predicts a class:

```txt
tree_1(x), tree_2(x), ..., tree_T(x)
```

The forest prediction is the majority vote:

```txt
y_pred = mode({tree_t(x) : t = 1..T})
```

The forest can also estimate class probabilities from vote proportions:

```txt
P(class = c | x) ≈ number_of_trees_voting_for_c / T
```

where:

```txt
T = number of trees
```

## Why randomness helps

A single decision tree can be unstable.

Small changes in the training data can change the selected splits.

Random Forest reduces this instability by averaging many trees.

The forest benefits when trees are:

```txt
individually useful
not perfectly correlated with each other
```

Bootstrap sampling and feature subsampling make the trees different from one another.

This reduces variance compared with a single tree.

## Public API to math mapping

## `RandomForestOptions`

```cpp
struct RandomForestOptions {
    std::size_t n_estimators;
    bool bootstrap;
    std::optional<std::size_t> max_features;
    unsigned int random_seed;
    DecisionTreeOptions tree_options;
};
```

### Mathematical role

`RandomForestOptions` controls the ensemble construction process.

Important fields:

```txt
n_estimators
bootstrap
max_features
random_seed
tree_options
```

Math meaning:

```txt
n_estimators:
  number of trees in the ensemble

bootstrap:
  whether each tree is trained on a resampled dataset

max_features:
  number of features considered by each tree or split

random_seed:
  reproducible randomness for sampling/subsampling

tree_options:
  configuration of each base DecisionTreeClassifier
```

These options do not define a single equation. They define the sampling and aggregation process used to build the forest.

## `validate_random_forest_options`

```cpp
void validate_random_forest_options(
    const RandomForestOptions& options,
    const std::string& context
);
```

### Mathematical role

Ensures the ensemble configuration is meaningful.

Examples of mathematical requirements:

```txt
n_estimators >= 1
max_features, if set, must be at least 1
tree options must be valid
```

A forest with zero trees cannot vote.

A feature subset of size zero cannot produce meaningful splits.

### Infrastructure role

It also provides:

```txt
consistent error messages
context-specific validation
early failure for invalid configuration
```

## Constructor

```cpp
RandomForestClassifier();
explicit RandomForestClassifier(RandomForestOptions options);
```

### Mathematical role

The constructor does not train trees.

It stores the ensemble configuration:

```txt
number of trees
bootstrap behavior
feature subsampling behavior
tree options
random seed
```

Math implemented:

```txt
none directly
```

Infrastructure role:

```txt
configure ensemble behavior
validate options
```

## `fit`

```cpp
void fit(const Matrix& X, const Vector& y);
```

### Mathematical role

`fit` is the core training method.

It builds an ensemble of decision trees.

For each tree `t`:

```txt
1. sample training rows, usually by bootstrap sampling
2. configure tree-level feature subsampling
3. train a DecisionTreeClassifier on the sampled dataset
4. store the trained tree
```

If bootstrap is enabled, each tree receives a sampled dataset:

```txt
D_t = bootstrap_sample(D_train)
```

where sampling is done with replacement.

If feature subsampling is enabled, each split considers only a subset of features:

```txt
F_t or F_node ⊂ {1, 2, ..., d}
```

depending on the exact implementation path through tree options.

Each tree learns its own recursive partition of feature space.

The forest stores all trained trees:

```txt
forest = {tree_1, tree_2, ..., tree_T}
```

### What `fit` does mathematically

`fit` implements:

```txt
1. validate training data and labels
2. determine number of classes
3. for each estimator:
   - draw bootstrap sample if enabled
   - configure feature subsampling
   - train a DecisionTreeClassifier
   - store the fitted tree
4. define the ensemble as the collection of fitted trees
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
input validation
target validation
option validation
feature-count tracking
class-count tracking
random-seed management
tree vector storage
fitted-state tracking
```

These are necessary for correctness but are not the voting rule itself.

## `predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` applies the forest majority-vote rule.

For each sample `x_i`:

```txt
votes = [tree_1(x_i), tree_2(x_i), ..., tree_T(x_i)]
y_pred_i = majority_vote(votes)
```

For class `c`, the vote count is:

```txt
votes_c = sum_t 1(tree_t(x_i) = c)
```

The predicted class is:

```txt
argmax_c votes_c
```

### What `predict` does mathematically

`predict` implements:

```txt
tree-level prediction
vote counting
majority-vote aggregation
tie-breaking where needed
```

### What `predict` does as infrastructure

`predict` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output vector allocation
looping over samples
```

## `predict_proba`

```cpp
Matrix predict_proba(const Matrix& X) const;
```

### Mathematical role

`predict_proba` converts tree votes into class probability estimates.

For each sample `x_i` and class `c`:

```txt
P_hat(y = c | x_i) = votes_c / T
```

where:

```txt
votes_c = number of trees predicting class c
T = total number of trees
```

Expected output shape:

```txt
probabilities.rows() == X.rows()
probabilities.cols() == num_classes
```

Each row should sum approximately to:

```txt
1.0
```

### Interpretation

These are vote proportions.

They are probability-like, but they are not guaranteed to be calibrated probabilities.

### Infrastructure role

`predict_proba` also handles:

```txt
fitted-state validation
input validation
feature-count validation
probability matrix allocation
normalizing vote counts by number of trees
```

## `is_fitted`

```cpp
bool is_fitted() const;
```

### Mathematical role

None directly.

This is state-management infrastructure.

It answers whether the ensemble has trained trees available for prediction.

## `options`

```cpp
const RandomForestOptions& options() const;
```

### Mathematical role

Returns the ensemble training configuration.

This configuration determines:

```txt
how many trees exist
how bootstrap samples were drawn
how feature subsampling was configured
how each base tree was constrained
```

### Infrastructure role

Useful for diagnostics and reproducibility.

## `num_classes`

```cpp
std::size_t num_classes() const;
```

### Mathematical role

Returns the number of classes learned during training.

This controls:

```txt
number of probability columns
valid vote bins
valid predicted labels
```

### Infrastructure role

Also validates probability output shape and model state.

## `num_trees`

```cpp
std::size_t num_trees() const;
```

### Mathematical role

Returns the number of trained trees:

```txt
T
```

This directly affects:

```txt
vote aggregation
probability normalization
ensemble stability
```

### Infrastructure role

Useful for checking that the expected number of estimators was trained.

## Important internal math concepts

## Bootstrap sampling

Bootstrap sampling creates a dataset of the same size as the original training set by sampling rows with replacement.

If the original dataset is:

```txt
D = {(x_1, y_1), ..., (x_n, y_n)}
```

a bootstrap dataset is:

```txt
D_t = sample_with_replacement(D, n)
```

Some original samples may appear multiple times.

Some may not appear at all.

The samples not selected for a tree are called out-of-bag samples.

In this project, bootstrap utilities may expose out-of-bag indices, even if full OOB scoring is deferred.

## Bagging

Bagging means:

```txt
bootstrap aggregating
```

The idea is:

```txt
train multiple models on different bootstrap samples
aggregate their predictions
```

For classification, aggregation is usually majority voting.

Bagging reduces variance when base learners are unstable, which decision trees often are.

## Feature subsampling

Random Forest also introduces randomness by limiting the number of features considered during split search.

Instead of searching all features:

```txt
{1, 2, ..., d}
```

a tree split may search only:

```txt
random subset of features
```

This makes trees less correlated with one another.

Less correlated trees improve the variance-reduction effect of ensembling.

## Majority vote

For a sample `x`, each tree predicts a class.

Vote count for class `c`:

```txt
votes_c = sum_t 1(tree_t(x) = c)
```

Final prediction:

```txt
argmax_c votes_c
```

If there is a tie, the implementation should use deterministic tie-breaking.

Common deterministic tie-breaking is choosing the smallest class label among tied classes.

## Vote probabilities

The forest probability estimate is:

```txt
P_hat(c | x) = votes_c / T
```

Example:

```txt
T = 10
votes for class 1 = 7
P_hat(class 1 | x) = 0.7
```

These probabilities are based on votes, not on a calibrated likelihood model.

## Bias-variance intuition

Random Forest primarily reduces variance.

A single tree can overfit and vary strongly with data changes.

An ensemble of many diverse trees averages out some of this instability.

However:

```txt
if all trees make the same mistake:
  the forest also makes that mistake

if trees are too shallow:
  the forest may underfit

if trees are too correlated:
  variance reduction is weaker
```

## Relationship to DecisionTreeClassifier

A random forest is an orchestrator around many decision trees.

The base learner math comes from:

```txt
DecisionTreeClassifier
```

The forest-level math is:

```txt
bootstrap sampling
feature subsampling
vote aggregation
vote-proportion probabilities
```

The forest should not duplicate tree split logic.

It should reuse the tree learner.

## Method classification

| Method / Struct | Tree math | Ensemble math | Probability math | Sampling math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `RandomForestOptions` | Partial | Yes | No | Yes | Yes |
| `validate_random_forest_options` | Partial | Partial | No | Partial | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Yes | Yes | No | Yes | Yes |
| `predict` | Yes | Yes | No | No | Yes |
| `predict_proba` | Yes | Yes | Yes | No | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `options` | Partial | Partial | No | Partial | Yes |
| `num_classes` | Partial | Partial | Yes | No | Yes |
| `num_trees` | No | Yes | Yes | No | Yes |

## Output files and math meaning

## Binary classification `metrics.csv`

Relevant rows:

```txt
model = RandomForestClassifier
metric = accuracy
metric = precision
metric = recall
metric = f1
```

Math meaning:

```txt
accuracy:
  fraction of correct majority-vote predictions

precision:
  true positives / predicted positives

recall:
  true positives / actual positives

f1:
  harmonic mean of precision and recall
```

For imbalanced binary datasets, recall and F1 are usually more informative than accuracy alone.

## Multiclass classification `metrics.csv`

Relevant rows:

```txt
model = RandomForestClassifier
metric = accuracy
metric = macro_precision
metric = macro_recall
metric = macro_f1
```

Math meaning:

```txt
accuracy:
  fraction of correct predictions

macro_precision:
  average precision across classes

macro_recall:
  average recall across classes

macro_f1:
  average F1 score across classes
```

## `predictions.csv`

Relevant columns:

```txt
y_true
y_pred
correct
```

Math meaning:

```txt
y_true:
  actual class label

y_pred:
  majority-vote forest prediction

correct:
  1 if y_pred == y_true else 0
```

## `probabilities.csv`

Relevant columns:

```txt
probability_class_0
probability_class_1
probability_class_2
...
```

Math meaning:

```txt
probability_class_c:
  fraction of trees voting for class c
```

Each row should sum approximately to:

```txt
1.0
```

## `hyperparameter_sweep.csv`

Relevant rows:

```txt
model = RandomForestClassifier
param_name = n_estimators
param_name = max_features
metric = accuracy
metric = precision
metric = recall
metric = f1
```

Math meaning:

```txt
n_estimators:
  number of trees used in the ensemble

max_features:
  number of features considered during tree split search

metric value:
  classification performance produced by that forest configuration
```

## Why Random Forest does not export loss history

Random Forest is not trained by iterative gradient descent.

It does not minimize a differentiable loss over epochs.

Instead, it trains independent decision trees and aggregates their votes.

Therefore, there is no training loss curve analogous to:

```txt
LogisticRegression
LinearSVM
TinyMLPBinaryClassifier
```

## Why Random Forest probabilities are different from LogisticRegression probabilities

Logistic regression probability:

```txt
sigmoid(w^T x + b)
```

Random Forest probability:

```txt
fraction of trees voting for a class
```

Both produce values in `[0, 1]`, but they come from different mathematical sources.

Random Forest probabilities are vote proportions and should not automatically be treated as calibrated probabilities.

## Practical interpretation

`RandomForestClassifier` is a strong nonlinear tabular baseline.

It is especially useful when:

```txt
single trees are unstable
feature interactions matter
nonlinear boundaries matter
probability-like vote outputs are useful
```

If Random Forest performs better than a single tree, it usually indicates that ensembling reduced variance or improved robustness.

If it performs poorly, possible causes include:

```txt
too few trees
trees too shallow
poor feature set
severe class imbalance
too much or too little feature subsampling
noisy target labels
small training subset
```

For Phase 11 binary KC1 defect classification, Random Forest should be interpreted using:

```txt
precision
recall
f1
```

not accuracy alone.

For Phase 11 Wine multiclass classification, Random Forest is expected to be strong because the classes are fairly separable with numeric features.

## Summary

`RandomForestClassifier` maps to the following mathematical pipeline:

```txt
training matrix X
target labels y
        ↓
for each tree:
    bootstrap sample rows
    configure feature subsampling
    train DecisionTreeClassifier
        ↓
store all trained trees
        ↓
for each test sample:
    collect tree predictions
    count votes per class
        ↓
majority-vote class prediction
vote-proportion class probabilities
        ↓
classification metrics
```

The core forest-level math lives in:

```txt
fit
predict
predict_proba
num_classes
num_trees
bootstrap sampling
feature subsampling
majority voting
vote-proportion probabilities
```

The base tree math lives in:

```txt
DecisionTreeClassifier
split scoring
recursive partitioning
leaf majority prediction
```

The supporting infrastructure lives in:

```txt
options
validate_random_forest_options
is_fitted
input validation
random-seed management
tree storage
output export workflows
```
