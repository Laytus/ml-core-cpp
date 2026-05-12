# GaussianNaiveBayes Usage

## What the model does

`GaussianNaiveBayes` is a probabilistic classification model.

It applies Bayes' theorem with the Naive Bayes conditional-independence assumption:

```txt
P(class | features) ∝ P(class) * Π P(feature_j | class)
```

The Gaussian version assumes that each numeric feature follows a Gaussian distribution within each class:

```txt
feature_j | class_k ~ Normal(mean_kj, variance_kj)
```

During training, the model estimates:

```txt
class priors
per-class feature means
per-class feature variances
```

During prediction, it computes class log-likelihoods and converts them into normalized class probabilities.

In practical terms, `GaussianNaiveBayes` is useful when you want:

```txt
a fast probabilistic classifier
class probability outputs
a simple likelihood-based baseline
a model that works for binary and multiclass classification
a clear bridge between probability theory and ML predictions
```

## Supported task type

`GaussianNaiveBayes` supports:

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
decision scores / margins
clustering
dimensionality reduction
gradient-based optimization
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

`GaussianNaiveBayes` expects numeric features.

Recommended preprocessing:

```txt
reject missing values or preprocess them before loading
avoid non-numeric feature columns
ensure class labels are numeric and integer-valued
standardize features if comparing with scale-sensitive models
check for features with near-zero variance
```

Naive Bayes itself does not require gradient-based feature scaling in the same way as logistic regression or SVM. However, in Phase 11 workflows, features may be standardized for consistent comparison across model families.

## Gaussian assumption

For each class and each feature, the model estimates:

```txt
mean
variance
```

Then it uses the Gaussian likelihood:

```txt
P(x_j | class_k)
```

This works best when feature distributions are reasonably separated by class and not extremely non-Gaussian.

The model can still be useful even when the assumption is imperfect, because it often provides a strong simple probabilistic baseline.

## Naive independence assumption

The model assumes that features are conditionally independent given the class:

```txt
P(x_1, x_2, ..., x_d | class) = Π P(x_j | class)
```

This assumption is usually not exactly true in real datasets.

However, Naive Bayes can still work well as a classifier even when the independence assumption is only approximate.

## How to instantiate the model

Basic model:

```cpp
#include "ml/probabilistic/naive_bayes.hpp"

ml::GaussianNaiveBayesOptions options;
options.variance_smoothing = 1e-9;

ml::GaussianNaiveBayes model(options);
```

Default constructor is also available:

```cpp
ml::GaussianNaiveBayes model;
```

## Main options

`GaussianNaiveBayesOptions` contains:

```txt
variance_smoothing
```

### `variance_smoothing`

Adds a small positive value to variances for numerical stability.

Purpose:

```txt
avoid division by zero
avoid extremely sharp Gaussian likelihoods
stabilize probability computation
```

Example:

```cpp
ml::GaussianNaiveBayesOptions options;
options.variance_smoothing = 1e-9;
```

A larger value can make the model less sensitive to very small feature variance.

## How to call `fit`

```cpp
model.fit(X_train, y_train);
```

Expected behavior:

```txt
- validates options
- validates non-empty matrix/vector inputs
- validates matching number of rows
- validates class labels
- estimates class priors
- estimates per-class feature means
- estimates per-class feature variances
- applies variance smoothing
- marks the model as fitted
```

The target vector must contain numeric class labels.

## How to call `predict`

```cpp
ml::Vector predicted_classes = model.predict(X_test);
```

Expected output:

```txt
predicted_classes.size() == X_test.rows()
```

Each prediction is the class with the highest posterior probability.

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

Each row contains normalized class probabilities.

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

## How to call `predict_log_proba`

```cpp
ml::Matrix log_probabilities = model.predict_log_proba(X_test);
```

Expected output shape:

```txt
log_probabilities.rows() == X_test.rows()
log_probabilities.cols() == number of classes
```

Log probabilities are useful for numerical stability and for inspecting relative class likelihoods without underflow.

## How to inspect learned statistics

After fitting:

```cpp
const ml::Vector& classes = model.classes();
const ml::Vector& class_priors = model.class_priors();
const ml::Matrix& means = model.means();
const ml::Matrix& variances = model.variances();
Eigen::Index feature_count = model.num_features();
```

Expected shapes:

```txt
classes.size() == number of classes
class_priors.size() == number of classes
means.rows() == number of classes
means.cols() == number of features
variances.rows() == number of classes
variances.cols() == number of features
```

Interpretation:

```txt
class_priors(k):
  estimated P(class_k)

means(k, j):
  estimated mean of feature j for class k

variances(k, j):
  estimated variance of feature j for class k
```

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

For imbalanced binary datasets, inspect:

```txt
precision
recall
f1
```

rather than accuracy alone.

## How to read outputs

In Phase 11 practical workflows, `GaussianNaiveBayes` writes results through the binary and multiclass classification comparison workflows.

Main binary classification output files:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/probabilities.csv
```

Main multiclass classification output files:

```txt
outputs/practical-exercises/multiclass-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/predictions.csv
outputs/practical-exercises/multiclass-classification/probabilities.csv
```

`GaussianNaiveBayes` does not currently export:

```txt
decision_scores
loss_history
```

because it is not a margin-based or iterative gradient-trained model.

### `metrics.csv`

Binary example rows:

```txt
run_id,workflow,dataset,model,split,metric,value
gaussian_naive_bayes_baseline,binary_classification,nasa_kc1_software_defects,GaussianNaiveBayes,test,accuracy,...
gaussian_naive_bayes_baseline,binary_classification,nasa_kc1_software_defects,GaussianNaiveBayes,test,precision,...
gaussian_naive_bayes_baseline,binary_classification,nasa_kc1_software_defects,GaussianNaiveBayes,test,recall,...
gaussian_naive_bayes_baseline,binary_classification,nasa_kc1_software_defects,GaussianNaiveBayes,test,f1,...
```

Multiclass example rows:

```txt
run_id,workflow,dataset,model,split,metric,value
gaussian_naive_bayes_baseline,multiclass_classification,wine,GaussianNaiveBayes,test,accuracy,...
gaussian_naive_bayes_baseline,multiclass_classification,wine,GaussianNaiveBayes,test,macro_precision,...
gaussian_naive_bayes_baseline,multiclass_classification,wine,GaussianNaiveBayes,test,macro_recall,...
gaussian_naive_bayes_baseline,multiclass_classification,wine,GaussianNaiveBayes,test,macro_f1,...
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

## Practical workflow examples

`GaussianNaiveBayes` is used in:

```txt
include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp

include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp
```

Related notebooks:

```txt
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
```

The notebooks visualize:

```txt
metric comparison
prediction correctness
probability distributions
probability distributions by true class
confusion-matrix-style tables
```

## Probability interpretation

`GaussianNaiveBayes` produces probability-like outputs using class priors and Gaussian likelihoods.

However, these probabilities depend strongly on the assumptions:

```txt
features are conditionally independent given class
features are Gaussian within each class
estimated variances are stable
```

The probabilities are useful for comparison and ranking, but they should not be treated as guaranteed calibrated probabilities unless calibration is evaluated separately.

## Hyperparameter behavior

The primary hyperparameter is:

```txt
variance_smoothing
```

Interpretation:

```txt
small variance_smoothing:
  closer to raw estimated variances
  can be sensitive to near-zero variance features

larger variance_smoothing:
  more numerical stability
  smoother likelihood estimates
  can reduce overconfidence
```

The current Phase 11 hyperparameter sweep does not focus on Gaussian Naive Bayes, but it is included in model comparison workflows.

## Common mistakes to avoid

### 1. Expecting feature independence to be literally true

The Naive Bayes assumption is rarely exactly true.

The model can still be useful, but correlated features can affect probability estimates.

### 2. Treating probabilities as perfectly calibrated

`predict_proba` returns normalized posterior estimates under the model assumptions.

They are not guaranteed to be calibrated probabilities.

### 3. Ignoring zero or tiny variances

Very small variances can produce unstable likelihoods.

Use `variance_smoothing` to stabilize computation.

### 4. Using non-numeric features directly

This implementation expects numeric matrices.

Categorical features must be encoded before loading.

### 5. Ignoring class imbalance

Class priors reflect the training distribution.

On imbalanced datasets, priors can strongly influence predictions.

Always inspect precision, recall, and F1 for binary imbalanced tasks.

### 6. Expecting iterative loss history

Gaussian Naive Bayes is fitted by estimating statistics, not by gradient descent.

It does not expose a training loss history.

### 7. Calling prediction methods before `fit`

The model must be trained before calling:

```txt
predict
predict_proba
predict_log_proba
classes
class_priors
means
variances
num_features
```

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating `GaussianNaiveBayes` are:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/probabilities.csv

outputs/practical-exercises/multiclass-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/predictions.csv
outputs/practical-exercises/multiclass-classification/probabilities.csv
```

The main visualization notebooks are:

```txt
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
```

The main workflow implementations are:

```txt
include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp
include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp
```

The model implementation files are:

```txt
include/ml/probabilistic/naive_bayes.hpp
src/probabilistic/naive_bayes.cpp
```

## When to use this model

Use `GaussianNaiveBayes` when:

```txt
the task is binary or multiclass classification
features are numeric
you want a fast probabilistic baseline
you want class probability outputs
you want to study priors and likelihoods
you want a simple model with no iterative optimization
```

Avoid relying on it alone when:

```txt
features are strongly correlated
feature distributions are far from Gaussian
probability calibration is critical
categorical variables are not encoded
the task requires nonlinear boundary modeling beyond Gaussian class assumptions
the dataset has missing values that have not been handled
```
