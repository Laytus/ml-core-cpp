# KNNClassifier Math Map

## Purpose

This document maps the public `KNNClassifier` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/distance/knn_classifier.hpp
src/distance/knn_classifier.cpp
include/ml/distance/distance_metrics.hpp
src/distance/distance_metrics.cpp
include/ml/common/classification_metrics.hpp
include/ml/common/multiclass_metrics.hpp
```

Related practical usage doc:

```txt
docs/practical/models/knn-classifier-usage.md
```

Related theory doc:

```txt
docs/theory/distance-and-kernel-thinking.md
```

## Model idea

`KNNClassifier` is a distance-based classification model.

KNN means:

```txt
k-nearest neighbors
```

Unlike linear models or trees, KNN does not learn weights, thresholds, or probability parameters during training.

Instead, it stores the training dataset and predicts a new sample by comparing it to the stored training samples.

For a query sample `x`, the prediction process is:

```txt
1. compute distance from x to every training sample
2. select the k closest samples
3. collect their labels
4. return the most common label
```

Mathematically:

```txt
N_k(x) = set of k training samples closest to x
```

Prediction:

```txt
y_pred = mode({ y_i : x_i in N_k(x) })
```

where:

```txt
x_i = training sample
y_i = training label
N_k(x) = k-nearest-neighbor set
mode = most frequent class label
```

## Supported distance functions

The model supports distance metrics through `DistanceMetric`.

Common formulas:

### Euclidean distance

```txt
d(a, b) = sqrt(sum_j (a_j - b_j)^2)
```

### Squared Euclidean distance

```txt
d(a, b) = sum_j (a_j - b_j)^2
```

This gives the same neighbor ordering as Euclidean distance, but avoids the square root.

### Manhattan distance

```txt
d(a, b) = sum_j |a_j - b_j|
```

The selected distance function defines what “nearest” means.

## Public API to math mapping

## `DistanceMetric`

```cpp
enum class DistanceMetric {
    Euclidean,
    SquaredEuclidean,
    Manhattan
};
```

### Mathematical role

`DistanceMetric` selects the distance function used to compare samples.

Math implemented:

```txt
Euclidean:
  sqrt(sum squared coordinate differences)

SquaredEuclidean:
  sum squared coordinate differences

Manhattan:
  sum absolute coordinate differences
```

This directly affects neighbor ordering and therefore predictions.

## `distance_metric_name`

```cpp
std::string distance_metric_name(DistanceMetric metric);
```

### Mathematical role

None directly.

This is naming/diagnostic infrastructure.

It maps a distance metric enum value to a readable string.

## `KNNClassifierOptions`

```cpp
struct KNNClassifierOptions {
    std::size_t k;
    DistanceMetric distance_metric;
};
```

### Mathematical role

`KNNClassifierOptions` controls the KNN prediction rule.

Important fields:

```txt
k
distance_metric
```

Math meaning:

```txt
k:
  number of neighbors used for voting

distance_metric:
  function used to measure closeness between samples
```

Effect of `k`:

```txt
small k:
  highly local decision rule
  more sensitive to noise

large k:
  smoother decision rule
  less sensitive to individual samples
  may underfit local structure
```

## `validate_knn_classifier_options`

```cpp
void validate_knn_classifier_options(
    const KNNClassifierOptions& options,
    const std::string& context
);
```

### Mathematical role

Ensures the KNN rule is valid.

The key mathematical requirement is:

```txt
k >= 1
```

A classifier cannot use zero neighbors.

### Infrastructure role

This function also provides:

```txt
consistent error messages
context-specific validation
early failure for invalid configuration
```

## Constructor

```cpp
KNNClassifier();
explicit KNNClassifier(KNNClassifierOptions options);
```

### Mathematical role

The constructor does not perform neighbor search.

It only stores the prediction-rule configuration:

```txt
k
distance metric
```

Math implemented:

```txt
none directly
```

Infrastructure role:

```txt
configure k
configure distance metric
validate options
```

## `fit`

```cpp
void fit(const Matrix& X, const Vector& y);
```

### Mathematical role

`fit` stores the reference set used for nearest-neighbor prediction.

KNN has no parameter optimization step.

There is no gradient descent, impurity minimization, likelihood fitting, or loss minimization in this model.

Mathematically, `fit` defines the stored training set:

```txt
D_train = {(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}
```

Prediction later depends entirely on this stored set.

### What `fit` does mathematically

`fit` implements:

```txt
1. store training feature matrix X_train
2. store training target labels y_train
3. define the reference set for future nearest-neighbor search
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
input validation
shape checking
empty dataset rejection
matching row/target size validation
feature-count tracking
fitted-state tracking
```

These are necessary for correctness but are not model optimization.

## `predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` applies the KNN decision rule to each query sample.

For each query sample `x`:

```txt
1. compute d(x, x_i) for every training sample x_i
2. sort or rank training samples by distance
3. select the k nearest labels
4. return the majority-vote class
```

Mathematically:

```txt
distances_i = d(x, x_i)
N_k(x) = indices of k smallest distances_i
y_pred = mode({y_i : i in N_k(x)})
```

### What `predict` does mathematically

`predict` implements:

```txt
distance computation
nearest-neighbor selection
majority voting
deterministic tie-breaking where implemented
```

### What `predict` does as infrastructure

`predict` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output vector allocation
looping over query rows
```

## `is_fitted`

```cpp
bool is_fitted() const;
```

### Mathematical role

None directly.

This is state-management infrastructure.

It answers whether the stored reference set:

```txt
X_train
y_train
```

is available and valid for prediction.

## `options`

```cpp
const KNNClassifierOptions& options() const;
```

### Mathematical role

Exposes the KNN rule configuration:

```txt
k
distance metric
```

These values determine the mathematical prediction rule.

### Infrastructure role

Also provides access to the configuration used during model construction.

## `num_train_samples`

```cpp
std::size_t num_train_samples() const;
```

### Mathematical role

Returns the number of stored training samples:

```txt
n_train
```

This matters because KNN prediction compares each query sample against all stored training samples.

Prediction cost scales with:

```txt
n_train
```

### Infrastructure role

Useful for diagnostics and sanity checks.

## `num_features`

```cpp
Eigen::Index num_features() const;
```

### Mathematical role

Returns the dimensionality of the feature space:

```txt
d
```

Distance computations are performed across these `d` coordinates:

```txt
d(a, b) = distance over j = 1..d
```

### Infrastructure role

Used to validate that prediction data has the same number of features as training data.

## Important internal math concepts

## Distance computation

The most important mathematical operation in KNN is computing distances between vectors.

For two samples:

```txt
a = [a_1, a_2, ..., a_d]
b = [b_1, b_2, ..., b_d]
```

A distance metric produces a scalar:

```txt
d(a, b)
```

The smaller the distance, the more similar the samples are under that metric.

## Neighbor ranking

For a query sample `x`, KNN computes:

```txt
d(x, x_1)
d(x, x_2)
...
d(x, x_n)
```

Then it ranks the training samples by distance:

```txt
nearest first
farthest last
```

Only the first `k` labels are used for prediction.

## Majority vote

Given the neighbor labels:

```txt
[y_a, y_b, ..., y_k]
```

The prediction is:

```txt
most frequent label
```

Example:

```txt
neighbors: [0, 1, 1, 1, 2]
prediction: 1
```

## Tie-breaking

A tie can occur when multiple labels receive the same number of votes.

Example:

```txt
neighbors: [0, 0, 1, 1]
```

A deterministic implementation should break ties consistently.

Common strategies include:

```txt
choose the smaller label
choose the label whose nearest neighbor is closest
choose the first label after sorted neighbor order
```

The exact behavior depends on the implementation. The important practical point is that tie-breaking should be deterministic.

## Feature scaling

KNN depends directly on distances, so feature scaling is part of the mathematical behavior.

If one feature has a much larger numeric range than others, it can dominate the distance:

```txt
large-scale feature difference >> small-scale feature difference
```

This is why standardization is usually required.

For standardized features:

```txt
each feature contributes more comparably to distance
```

## Curse of dimensionality

As dimensionality grows, distances can become less informative.

In high-dimensional spaces:

```txt
nearest and farthest samples may become less distinguishable
local neighborhoods may become noisy
irrelevant features can dominate similarity
```

This affects KNN more strongly than many parametric models.

## Method classification

| Method / Struct | Distance math | Voting math | Optimization math | Metrics math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `DistanceMetric` | Yes | No | No | No | Partial |
| `distance_metric_name` | No | No | No | No | Yes |
| `KNNClassifierOptions` | Yes | Yes | No | No | Yes |
| `validate_knn_classifier_options` | Partial | Partial | No | No | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Partial | No | No | No | Yes |
| `predict` | Yes | Yes | No | No | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `options` | Partial | Partial | No | No | Yes |
| `num_train_samples` | Partial | No | No | No | Yes |
| `num_features` | Partial | No | No | No | Yes |

## Output files and math meaning

## `metrics.csv`

Relevant rows:

```txt
model = KNNClassifier
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

KNN predictions are hard class labels, so these metrics are computed from predicted labels and true labels.

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
  majority-vote KNN class prediction

correct:
  1 if y_pred == y_true else 0
```

## `hyperparameter_sweep.csv`

Relevant rows:

```txt
model = KNNClassifier
param_name = k
metric = accuracy
metric = macro_precision
metric = macro_recall
metric = macro_f1
```

Math meaning:

```txt
k:
  number of neighbors used in majority voting

metric value:
  performance produced by that k value
```

The sweep helps show how neighborhood size affects classification behavior.

## Why KNN does not export loss history

KNN has no iterative training objective.

There is no training loss curve because `fit` does not optimize parameters.

The model stores the data and delays computation until prediction time.

## Why KNN does not export decision scores

The current `KNNClassifier` API exposes hard class predictions only.

It does not expose:

```txt
decision_function
predict_proba
```

A future extension could compute probability-like outputs from neighbor vote proportions:

```txt
P(class_j | x) ≈ votes_for_class_j / k
```

But that is not part of the current public API.

## Practical interpretation

`KNNClassifier` is a local, distance-based classifier.

It is most useful when:

```txt
nearby samples tend to share labels
features are scaled
dimensionality is moderate
the dataset is not too large
```

If KNN performs well, the dataset likely has meaningful local neighborhoods.

If KNN performs poorly, possible causes include:

```txt
features are not scaled
irrelevant features dominate distances
classes overlap heavily
k is too small or too large
dimensionality is too high
training set is too small or noisy
```

In the Phase 11 Wine workflow, KNN performs strongly because the standardized Wine features produce meaningful class neighborhoods.

## Summary

`KNNClassifier` maps to the following mathematical pipeline:

```txt
training feature matrix X_train
training labels y_train
        ↓
store reference dataset
        ↓
for each query sample x:
        ↓
compute distances to all training samples
        ↓
select k nearest neighbors
        ↓
majority vote over neighbor labels
        ↓
predicted class
        ↓
classification metrics
```

The core math lives in:

```txt
distance metric selection
fit as reference-set storage
predict
nearest-neighbor search
majority voting
```

The supporting infrastructure lives in:

```txt
options
validation
is_fitted
num_train_samples
num_features
output export workflows
```
