# KNNClassifier Usage

## What the model does

`KNNClassifier` is a distance-based multiclass classification model.

KNN means:

```txt
k-nearest neighbors
```

The model does not learn weights during training. Instead, it stores the training data and predicts a new sample by:

```txt
1. computing distances from the new sample to all training samples
2. selecting the k closest training samples
3. returning the most common class among those neighbors
```

In practical terms, `KNNClassifier` is useful when you want:

```txt
a simple non-parametric classifier
a distance-based baseline
a model that can handle nonlinear decision boundaries
an interpretable local-neighborhood method
```

## Supported task type

`KNNClassifier` supports:

```txt
classification
```

It can be used for binary or multiclass classification as long as labels are numeric class IDs.

It does not support:

```txt
regression
probability prediction
decision scores
loss history
dimensionality reduction
clustering
```

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
y contains class labels
```

Typical shapes:

```txt
X: n_samples x n_features
y: n_samples
```

Example:

```cpp
ml::Matrix X(5, 2);
X << 1.0, 2.0,
     1.2, 1.8,
     5.0, 8.0,
     6.0, 9.0,
     9.0, 1.0;

ml::Vector y(5);
y << 0.0, 0.0, 1.0, 1.0, 2.0;
```

For real dataset workflows, use the CSV loader to create `X` and `y` from a processed CSV file.

## Target encoding

Targets should be encoded as numeric class labels.

For multiclass workflows, the recommended convention is:

```txt
0
1
2
...
num_classes - 1
```

For the Phase 11 Wine workflow, original labels are converted from:

```txt
1, 2, 3
```

to:

```txt
0, 1, 2
```

## Preprocessing usually needed

KNN is highly sensitive to feature scale because predictions depend directly on distances.

Recommended preprocessing:

```txt
standardize numeric features
reject missing values or preprocess them before loading
avoid non-numeric columns
use train-fitted preprocessing statistics only
choose distance metric deliberately
```

Feature scaling is especially important. Without scaling, a feature with a large numeric range can dominate the distance calculation.

For practical workflows in this project, feature standardization is handled in the workflow layer before training.

## Distance metrics

`KNNClassifier` supports the distance metric configured through `KNNClassifierOptions`.

Available options include:

```txt
Euclidean
SquaredEuclidean
Manhattan
```

Example:

```cpp
ml::KNNClassifierOptions options;
options.k = 5;
options.distance_metric = ml::DistanceMetric::Euclidean;
```

Use cases:

```txt
Euclidean:
  common default for standardized numeric features

SquaredEuclidean:
  similar neighbor ordering to Euclidean, avoids square root

Manhattan:
  can be useful when absolute coordinate differences are more appropriate
```

## How to instantiate the model

Basic model:

```cpp
#include "ml/distance/knn_classifier.hpp"

ml::KNNClassifierOptions options;
options.k = 5;
options.distance_metric = ml::DistanceMetric::Euclidean;

ml::KNNClassifier model(options);
```

Default constructor is also available:

```cpp
ml::KNNClassifier model;
```

## How to call `fit`

```cpp
model.fit(X_train, y_train);
```

Expected behavior:

```txt
- validates non-empty matrix/vector inputs
- validates matching number of rows
- validates options
- stores the training matrix
- stores the training labels
- marks the model as fitted
```

Unlike gradient-based models, KNN does not optimize weights during `fit`.

The training step is mostly data storage and validation.

## How to call `predict`

```cpp
ml::Vector predicted_classes = model.predict(X_test);
```

Expected output:

```txt
predicted_classes.size() == X_test.rows()
```

Each prediction is the majority class among the `k` nearest training samples.

Calling `predict` before `fit` should be rejected.

## How to inspect fitted state and training shape

After fitting:

```cpp
bool fitted = model.is_fitted();
std::size_t train_samples = model.num_train_samples();
Eigen::Index feature_count = model.num_features();
```

These are useful for validating that the model stored the expected training data.

## How to evaluate predictions

For multiclass classification, use:

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

For binary classification, use the binary metrics if labels are encoded as `0/1`:

```cpp
#include "ml/common/classification_metrics.hpp"

double accuracy = ml::accuracy_score(predicted_classes, y_test);
double precision = ml::precision_score(predicted_classes, y_test);
double recall = ml::recall_score(predicted_classes, y_test);
double f1 = ml::f1_score(predicted_classes, y_test);
```

## How to read outputs

In Phase 11 practical workflows, `KNNClassifier` writes results through the multiclass classification comparison workflow.

Main output files:

```txt
outputs/practical-exercises/multiclass-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/predictions.csv
outputs/practical-exercises/multiclass-classification/hyperparameter_sweep.csv
```

KNN does not currently export:

```txt
probabilities
decision scores
loss history
```

because the implemented `KNNClassifier` API exposes hard class prediction only.

### `metrics.csv`

Contains rows such as:

```txt
run_id,workflow,dataset,model,split,metric,value
knn_classifier_baseline,multiclass_classification,wine,KNNClassifier,test,accuracy,...
knn_classifier_baseline,multiclass_classification,wine,KNNClassifier,test,macro_precision,...
knn_classifier_baseline,multiclass_classification,wine,KNNClassifier,test,macro_recall,...
knn_classifier_baseline,multiclass_classification,wine,KNNClassifier,test,macro_f1,...
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

### `hyperparameter_sweep.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

For KNN sweeps, the main parameter is:

```txt
k
```

Example:

```txt
knn_k5,multiclass_classification,wine,KNNClassifier,test,k,5,macro_f1,...
```

## Practical workflow example

`KNNClassifier` is used in the Phase 11 multiclass classification workflow:

```txt
include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp
```

It is also used in the hyperparameter sweep workflow:

```txt
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

The corresponding notebooks are:

```txt
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The notebooks visualize:

```txt
metric comparison
prediction correctness
confusion-matrix-style tables
confusion-matrix-style plots
k vs metric behavior
```

## Hyperparameter behavior

The most important KNN hyperparameter is:

```txt
k
```

Interpretation:

```txt
small k:
  more local
  more sensitive to individual samples
  can overfit noise

larger k:
  smoother decision boundary
  less sensitive to individual samples
  can underfit if too large
```

In the Phase 11 Wine sweep, `k = 5` produced the best macro F1 among the tested values, while `k = 1`, `k = 3`, and `k = 5` all performed strongly.

This suggests that the Wine classes are fairly well-separated after standardization.

## Common mistakes to avoid

### 1. Forgetting feature scaling

This is the most important KNN mistake.

KNN relies directly on distances, so unscaled features can dominate the neighbor search.

Always standardize numeric features unless there is a deliberate reason not to.

### 2. Choosing `k` without validation

Do not assume one value of `k` is universally best.

Use validation or sweep outputs to compare several values.

### 3. Using KNN on very large datasets without considering cost

Prediction requires distance calculations against stored training samples.

Naive KNN prediction cost grows with:

```txt
number of test samples
number of training samples
number of features
```

This implementation is educational and direct, not optimized with approximate nearest-neighbor indexing.

### 4. Expecting probabilities from the current API

The current `KNNClassifier` API exposes:

```txt
predict
```

It does not expose:

```txt
predict_proba
```

If probability-like behavior is needed later, it could be added by converting neighbor vote counts into class proportions.

### 5. Ignoring dimensionality

In high-dimensional spaces, distances can become less informative.

This is part of the curse of dimensionality.

KNN is often more effective when:

```txt
features are meaningful
features are scaled
dimensionality is moderate
irrelevant features are removed
```

### 6. Calling `predict` before `fit`

The model must store training data before prediction.

Calling `predict` before `fit` should be rejected.

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating this model are:

```txt
outputs/practical-exercises/multiclass-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/predictions.csv
outputs/practical-exercises/multiclass-classification/hyperparameter_sweep.csv
```

The main visualization notebooks are:

```txt
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The main workflow implementations are:

```txt
include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

## When to use this model

Use `KNNClassifier` when:

```txt
the task is classification
you want a simple distance-based baseline
you suspect local neighborhoods are meaningful
the dataset is not too large
features can be standardized cleanly
you want to study the effect of k
```

Avoid relying on it alone when:

```txt
the dataset is very large
the feature space is high-dimensional and noisy
features have incompatible scales
prediction speed is critical
you need calibrated probabilities
you need a compact trained model with learned parameters
```
