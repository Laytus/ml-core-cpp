# GaussianNaiveBayes Math Map

## Purpose

This document maps the public `GaussianNaiveBayes` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/probabilistic/naive_bayes.hpp
src/probabilistic/naive_bayes.cpp
include/ml/common/classification_metrics.hpp
include/ml/common/multiclass_metrics.hpp
```

Related practical usage doc:

```txt
docs/practical/models/gaussian-naive-bayes-usage.md
```

Related theory doc:

```txt
docs/theory/probabilistic-ml.md
```

## Model idea

`GaussianNaiveBayes` is a probabilistic classification model.

It uses Bayes' theorem:

```txt
P(class | features) = P(features | class) * P(class) / P(features)
```

For classification, the denominator is the same for every class, so the model compares:

```txt
P(class) * P(features | class)
```

Naive Bayes assumes that features are conditionally independent given the class:

```txt
P(x_1, x_2, ..., x_d | class) = Π_j P(x_j | class)
```

Gaussian Naive Bayes additionally assumes each numeric feature follows a Gaussian distribution within each class:

```txt
x_j | class_k ~ Normal(mean_kj, variance_kj)
```

So the model estimates:

```txt
class priors
per-class feature means
per-class feature variances
```

Then it predicts by choosing the class with the largest posterior probability.

## Classification equation

For a sample `x` and class `c`, the unnormalized posterior is:

```txt
score_c = P(c) * Π_j P(x_j | c)
```

Using Gaussian likelihoods:

```txt
P(x_j | c) =
  1 / sqrt(2π * variance_cj)
  * exp(-((x_j - mean_cj)^2) / (2 * variance_cj))
```

The predicted class is:

```txt
y_pred = argmax_c score_c
```

In practice, log-probabilities are used for numerical stability:

```txt
log_score_c = log P(c) + sum_j log P(x_j | c)
```

The predicted class is equivalently:

```txt
y_pred = argmax_c log_score_c
```

## Why log-probabilities are used

Multiplying many probabilities can underflow numerically:

```txt
P(x_1 | c) * P(x_2 | c) * ... * P(x_d | c)
```

can become extremely small.

Taking logs changes products into sums:

```txt
log Π_j P(x_j | c) = sum_j log P(x_j | c)
```

This is more numerically stable.

## Public API to math mapping

## `GaussianNaiveBayesOptions`

```cpp
struct GaussianNaiveBayesOptions {
    double variance_smoothing;
};
```

### Mathematical role

`GaussianNaiveBayesOptions` controls numerical stabilization of Gaussian variance estimates.

Important field:

```txt
variance_smoothing
```

Math meaning:

```txt
variance_cj := variance_cj + variance_smoothing
```

or an equivalent stabilized variance adjustment.

This prevents division by zero or extremely sharp likelihoods when a feature has near-zero variance inside a class.

## `validate_gaussian_naive_bayes_options`

```cpp
void validate_gaussian_naive_bayes_options(
    const GaussianNaiveBayesOptions& options,
    const std::string& context
);
```

### Mathematical role

Ensures the variance smoothing value is meaningful.

Typical requirement:

```txt
variance_smoothing >= 0
```

and finite.

Negative variance smoothing would be mathematically invalid because variances must remain non-negative.

### Infrastructure role

It also provides:

```txt
consistent error messages
context-specific validation
early failure for invalid configuration
```

## Constructor

```cpp
GaussianNaiveBayes();
explicit GaussianNaiveBayes(GaussianNaiveBayesOptions options);
```

### Mathematical role

The constructor does not estimate probabilities.

It only stores the model configuration:

```txt
variance_smoothing
```

Math implemented:

```txt
none directly
```

Infrastructure role:

```txt
configure variance stabilization
validate options
```

## `fit`

```cpp
void fit(const Matrix& X, const Vector& y);
```

### Mathematical role

`fit` estimates the parameters of the probabilistic model.

Given training data:

```txt
X = feature matrix
y = class labels
```

the model identifies each class:

```txt
classes = unique(y)
```

For each class `c`, it estimates the class prior:

```txt
P(c) = number_of_samples_in_class_c / total_number_of_samples
```

For each class `c` and feature `j`, it estimates:

```txt
mean_cj = mean of feature j among samples with class c
```

and:

```txt
variance_cj = variance of feature j among samples with class c
```

Then it applies variance smoothing:

```txt
variance_cj := variance_cj + variance_smoothing
```

### What `fit` does mathematically

`fit` implements:

```txt
1. find the unique class labels
2. count samples per class
3. compute class priors
4. compute per-class feature means
5. compute per-class feature variances
6. apply variance smoothing
7. store all estimated probability parameters
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
option validation
input validation
shape checking
empty dataset rejection
target validation
finite-value validation where implemented
feature-count tracking
fitted-state tracking
```

These are necessary for correctness but are not the probability equations themselves.

## `predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` selects the class with the highest posterior probability.

It computes class log-probabilities:

```txt
log_score_c = log P(c) + sum_j log P(x_j | c)
```

Then returns:

```txt
argmax_c log_score_c
```

for each sample.

### What `predict` does mathematically

`predict` implements:

```txt
Gaussian likelihood evaluation
class prior addition in log-space
posterior score comparison
argmax class selection
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

## `predict_log_proba`

```cpp
Matrix predict_log_proba(const Matrix& X) const;
```

### Mathematical role

`predict_log_proba` returns normalized log posterior probabilities.

The model first computes joint log-likelihoods:

```txt
joint_log_likelihood_c = log P(c) + sum_j log P(x_j | c)
```

Then it normalizes them across classes so they represent log probabilities:

```txt
log P(c | x)
```

The normalization uses log-sum-exp:

```txt
log P(c | x) =
  joint_log_likelihood_c
  - log(sum_k exp(joint_log_likelihood_k))
```

Expected output shape:

```txt
rows = number of samples
cols = number of classes
```

### Why log-sum-exp matters

Directly computing:

```txt
sum_k exp(log_score_k)
```

can overflow or underflow.

The stable version subtracts the maximum log-score first:

```txt
log_sum_exp(scores) =
  max_score + log(sum_k exp(score_k - max_score))
```

### Infrastructure role

`predict_log_proba` also handles:

```txt
fitted-state validation
input validation
feature-count validation
matrix allocation
stable normalization
```

## `predict_proba`

```cpp
Matrix predict_proba(const Matrix& X) const;
```

### Mathematical role

`predict_proba` converts normalized log probabilities into probabilities:

```txt
P(c | x) = exp(log P(c | x))
```

Expected output shape:

```txt
probabilities.rows() == X.rows()
probabilities.cols() == number_of_classes
```

Each row should sum approximately to:

```txt
1.0
```

### Infrastructure role

`predict_proba` also handles:

```txt
calling predict_log_proba
exponentiating log probabilities
probability matrix allocation
```

## `is_fitted`

```cpp
bool is_fitted() const;
```

### Mathematical role

None directly.

This is state-management infrastructure.

It answers whether the model has estimated:

```txt
classes
class priors
means
variances
```

and can use them for prediction.

## `options`

```cpp
const GaussianNaiveBayesOptions& options() const;
```

### Mathematical role

Returns the variance-smoothing configuration.

This affects the Gaussian likelihood through the stabilized variance values.

## `classes`

```cpp
const Vector& classes() const;
```

### Mathematical role

Returns the class labels learned during fitting.

These labels define:

```txt
the order of probability columns
the labels used by predict
the class index for priors, means, and variances
```

If:

```txt
classes = [0, 1, 2]
```

then column `0` of probability outputs corresponds to class `0`, column `1` to class `1`, and so on.

## `class_priors`

```cpp
const Vector& class_priors() const;
```

### Mathematical role

Returns estimated class prior probabilities:

```txt
P(c)
```

Each prior is:

```txt
count(class c) / total samples
```

Expected property:

```txt
sum_c P(c) = 1
```

Priors affect predictions, especially when classes are imbalanced.

## `means`

```cpp
const Matrix& means() const;
```

### Mathematical role

Returns per-class feature means:

```txt
mean_cj
```

Expected shape:

```txt
means.rows() == number_of_classes
means.cols() == number_of_features
```

Each row corresponds to a class.

Each column corresponds to a feature.

## `variances`

```cpp
const Matrix& variances() const;
```

### Mathematical role

Returns per-class feature variances:

```txt
variance_cj
```

Expected shape:

```txt
variances.rows() == number_of_classes
variances.cols() == number_of_features
```

These variances are used in Gaussian likelihood evaluation.

They should include variance smoothing if that is applied during fitting.

## `num_features`

```cpp
Eigen::Index num_features() const;
```

### Mathematical role

Returns the number of features used during fitting.

This controls:

```txt
number of Gaussian likelihood terms per class
number of columns in means
number of columns in variances
valid feature count for prediction
```

### Infrastructure role

Also supports shape validation for prediction.

## Important internal math concepts

## Class prior

The class prior is:

```txt
P(c)
```

It measures how common class `c` is in the training data.

In imbalanced datasets, priors can strongly influence predictions.

Example:

```txt
if class 0 is much more common than class 1,
P(class 0) > P(class 1)
```

This makes class `0` more likely before looking at features.

## Gaussian likelihood

For each feature and class, the model assumes:

```txt
x_j | class_c ~ Normal(mean_cj, variance_cj)
```

The likelihood is:

```txt
P(x_j | c) =
  1 / sqrt(2π variance_cj)
  * exp(-((x_j - mean_cj)^2) / (2 variance_cj))
```

This gives higher probability to values close to the class-specific mean.

## Naive independence assumption

The model assumes:

```txt
P(x_1, x_2, ..., x_d | c) = Π_j P(x_j | c)
```

This is called “naive” because it assumes conditional independence between features.

This assumption is often false in real data, but the classifier can still work well.

## Joint log-likelihood

The joint log-likelihood for class `c` is:

```txt
log P(c) + sum_j log P(x_j | c)
```

This combines:

```txt
prior probability
feature likelihoods
```

in log-space.

## Posterior normalization

To convert joint log-likelihoods into posterior probabilities, normalize across all classes:

```txt
P(c | x) = exp(score_c) / sum_k exp(score_k)
```

This is structurally similar to softmax over class scores, except the scores come from Gaussian likelihoods and priors rather than learned linear logits.

## Variance smoothing

Variance smoothing prevents unstable likelihoods.

If variance is extremely small:

```txt
variance_cj ≈ 0
```

then the Gaussian density can become extremely sharp and numerically unstable.

Smoothing changes the variance to something like:

```txt
variance_cj + epsilon
```

where:

```txt
epsilon = variance_smoothing
```

## Gaussian Naive Bayes vs Softmax Regression

Both models produce class probabilities.

Softmax regression:

```txt
learns linear logits using gradient descent
probabilities = softmax(XW + b)
```

Gaussian Naive Bayes:

```txt
estimates class priors, means, variances
probabilities come from Gaussian likelihoods + Bayes rule
```

Softmax regression is discriminative.

Gaussian Naive Bayes is generative.

## Method classification

| Method / Struct | Probability math | Distribution math | Parameter estimation | Metrics math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `GaussianNaiveBayesOptions` | Partial | Partial | No | No | Yes |
| `validate_gaussian_naive_bayes_options` | Partial | Partial | No | No | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Yes | Yes | Yes | No | Yes |
| `predict` | Yes | Yes | No | No | Yes |
| `predict_log_proba` | Yes | Yes | No | No | Yes |
| `predict_proba` | Yes | Yes | No | No | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `options` | Partial | Partial | No | No | Yes |
| `classes` | Partial | No | Yes | No | Yes |
| `class_priors` | Yes | No | Yes | No | Yes |
| `means` | No | Yes | Yes | No | Yes |
| `variances` | No | Yes | Yes | No | Yes |
| `num_features` | Partial | Partial | No | No | Yes |

## Output files and math meaning

## Binary classification `metrics.csv`

Relevant rows:

```txt
model = GaussianNaiveBayes
metric = accuracy
metric = precision
metric = recall
metric = f1
```

Math meaning:

```txt
accuracy:
  fraction of correct predictions

precision:
  true positives / predicted positives

recall:
  true positives / actual positives

f1:
  harmonic mean of precision and recall
```

These metrics are computed from hard class predictions.

## Multiclass classification `metrics.csv`

Relevant rows:

```txt
model = GaussianNaiveBayes
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
  class with maximum posterior probability

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
  posterior probability P(class = c | x)
```

These probabilities are produced from:

```txt
class priors
Gaussian likelihoods
posterior normalization
```

Each row should sum approximately to:

```txt
1.0
```

## Why Gaussian Naive Bayes does not export loss history

Gaussian Naive Bayes is not trained by iterative gradient descent.

It estimates statistics directly:

```txt
class priors
means
variances
```

Therefore, there is no training loss curve analogous to:

```txt
LogisticRegression
SoftmaxRegression
LinearSVM
TinyMLPBinaryClassifier
```

## Why Gaussian Naive Bayes does not export decision scores

The model internally computes likelihood-based scores, usually in log-space.

However, the public practical workflow focuses on:

```txt
hard predictions
posterior probabilities
```

rather than margin-style decision scores.

If needed later, a workflow could export joint log-likelihoods or log posteriors as diagnostic scores.

## Practical interpretation

`GaussianNaiveBayes` is useful when:

```txt
features are numeric
class-conditional feature distributions are reasonably separated
a fast probabilistic baseline is needed
probability outputs are useful
```

If it performs well, the class-specific Gaussian statistics are informative enough for classification.

If it performs poorly, possible causes include:

```txt
features are strongly correlated
feature distributions are far from Gaussian
classes overlap heavily
variance estimates are unstable
class priors dominate minority-class prediction
```

For imbalanced datasets, inspect:

```txt
precision
recall
f1
```

instead of relying only on accuracy.

For multiclass datasets such as Wine, Gaussian Naive Bayes can perform strongly when class feature distributions are well separated.

## Summary

`GaussianNaiveBayes` maps to the following mathematical pipeline:

```txt
training matrix X
class labels y
        ↓
estimate class priors P(c)
estimate per-class means mean_cj
estimate per-class variances variance_cj
        ↓
for each test sample:
    compute Gaussian likelihoods P(x_j | c)
    add log priors
    sum log likelihoods across features
    normalize posterior probabilities across classes
        ↓
predict class with highest posterior probability
        ↓
evaluate classification metrics
```

The core math lives in:

```txt
fit
predict
predict_log_proba
predict_proba
class_priors
means
variances
classes
```

The supporting infrastructure lives in:

```txt
options
validate_gaussian_naive_bayes_options
is_fitted
num_features
input validation
output export workflows
```
