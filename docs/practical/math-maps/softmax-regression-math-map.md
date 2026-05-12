# SoftmaxRegression Math Map

## Purpose

This document maps the public `SoftmaxRegression` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/linear_models/softmax_regression.hpp
src/linear_models/softmax_regression.cpp
include/ml/linear_models/regularization.hpp
include/ml/common/multiclass_metrics.hpp
```

Related practical usage doc:

```txt
docs/practical/models/softmax-regression-usage.md
```

Related theory docs:

```txt
docs/theory/logistic-regression.md
docs/theory/linear-models.md
```

## Model equation

`SoftmaxRegression` is a multiclass linear classification model.

It starts with one linear score per class:

```txt
logits = X * W + b
```

For one sample and one class:

```txt
z_ij = w_j^T x_i + b_j
```

where:

```txt
x_i = feature vector for sample i
w_j = weight vector for class j
b_j = bias for class j
z_ij = logit for sample i and class j
```

Then it converts logits into probabilities using the softmax function:

```txt
p_ij = exp(z_ij) / sum_k exp(z_ik)
```

The probability is interpreted as:

```txt
p_ij = P(y_i = class_j | x_i)
```

The predicted class is the class with the highest probability:

```txt
y_pred_i = argmax_j p_ij
```

## Training objective

The training objective is categorical cross-entropy:

```txt
CCE = -(1 / n) * sum_i log(p_i,true_class)
```

Equivalently, using one-hot encoded targets:

```txt
CCE = -(1 / n) * sum_i sum_j Y_ij * log(P_ij)
```

where:

```txt
Y_ij = 1 if sample i belongs to class j, else 0
P_ij = predicted probability for class j
```

With Ridge regularization enabled, the objective becomes:

```txt
loss = CCE + lambda * ||W||^2
```

The bias vector is usually not regularized.

## Public API to math mapping

## `SoftmaxRegressionOptions`

### Mathematical role

`SoftmaxRegressionOptions` controls the optimization process and optional regularization.

Important fields:

```txt
learning_rate
max_iterations
tolerance
regularization
store_loss_history
```

Math meaning:

```txt
learning_rate:
  gradient descent step size

max_iterations:
  maximum number of optimization updates

tolerance:
  convergence threshold

regularization:
  penalty added to the objective

store_loss_history:
  whether losses are recorded for analysis
```

This struct is mostly optimization configuration, not the classifier equation itself.

## `SoftmaxRegressionTrainingHistory`

### Mathematical role

`SoftmaxRegressionTrainingHistory` records the behavior of the optimization process.

Important fields:

```txt
losses
iterations_run
converged
```

Math meaning:

```txt
losses:
  objective value across iterations

iterations_run:
  number of gradient descent steps actually executed

converged:
  whether the stopping criterion was reached
```

This is diagnostic infrastructure connected to the categorical cross-entropy objective.

It does not change the model mathematics.

## Constructor

```cpp
SoftmaxRegression();
explicit SoftmaxRegression(SoftmaxRegressionOptions options);
```

### Mathematical role

The constructor does not perform learning.

It only stores the optimization and regularization configuration.

Math implemented:

```txt
none directly
```

Infrastructure role:

```txt
configure learning rate
configure max iterations
configure tolerance
configure regularization behavior
configure loss-history recording
```

## `fit`

```cpp
void fit(const Matrix& X, const Vector& y, Eigen::Index num_classes);
```

### Mathematical role

`fit` is the core training method.

It estimates:

```txt
weight matrix W
bias vector b
```

by minimizing categorical cross-entropy.

Forward computation:

```txt
Z = XW + b
P = softmax(Z)
```

Target encoding:

```txt
Y = one_hot(y)
```

Loss:

```txt
CCE = -(1 / n) * sum_i sum_j Y_ij * log(P_ij)
```

The main gradient signal is:

```txt
P - Y
```

The gradients are:

```txt
dL/dW = (1 / n) * X^T (P - Y)
dL/db = column_mean(P - Y)
```

Gradient descent updates:

```txt
W := W - learning_rate * dL/dW
b := b - learning_rate * dL/db
```

With Ridge regularization:

```txt
L(W, b) = CCE + lambda * ||W||^2
```

The weight gradient receives an additional term:

```txt
2 * lambda * W
```

### What `fit` does mathematically

`fit` implements:

```txt
1. validate class count and target labels
2. initialize W and b
3. compute class logits
4. apply softmax to obtain class probabilities
5. build or use one-hot target representation
6. compute categorical cross-entropy or regularized loss
7. compute gradients with respect to W and b
8. update W and b through gradient descent
9. repeat until convergence or max_iterations
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
input validation
shape checking
empty dataset rejection
class-label validation
finite value checks where implemented
loss-history storage
convergence flag updates
fitted-state tracking
```

These are necessary for correctness but are not part of the model equation.

## `logits`

```cpp
Matrix logits(const Matrix& X) const;
```

### Mathematical role

`logits` computes one raw linear score per class.

Math implemented:

```txt
Z = XW + b
```

Expected shape:

```txt
Z.rows() == X.rows()
Z.cols() == num_classes
```

For one sample:

```txt
z_i = x_i^T W + b
```

where `z_i` is a row vector containing one score per class.

### Interpretation

```txt
higher logit for class j:
  stronger linear evidence for class j before softmax

logits are not probabilities:
  they can be negative, positive, or any real value
```

### Infrastructure role

`logits` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output matrix allocation
```

## `predict_proba`

```cpp
Matrix predict_proba(const Matrix& X) const;
```

### Mathematical role

`predict_proba` converts logits into class probabilities with softmax.

Math implemented:

```txt
Z = logits(X)
P = softmax(Z)
```

For each sample:

```txt
p_ij = exp(z_ij) / sum_k exp(z_ik)
```

Expected shape:

```txt
P.rows() == X.rows()
P.cols() == num_classes
```

Each row should sum approximately to:

```txt
1.0
```

### Numerical stability

Softmax is usually implemented with a stability adjustment:

```txt
softmax(z)_j = exp(z_j - max(z)) / sum_k exp(z_k - max(z))
```

This avoids overflow when logits are large.

### Infrastructure role

`predict_proba` also handles:

```txt
calling logits
input validation through logits
probability matrix allocation
row-wise probability normalization
```

## `predict_classes`

```cpp
Vector predict_classes(const Matrix& X) const;
```

### Mathematical role

`predict_classes` converts class probabilities into hard class labels.

Math implemented:

```txt
P = predict_proba(X)
y_pred_i = argmax_j P_ij
```

Expected output:

```txt
predicted_classes.size() == X.rows()
predicted_classes contains labels in [0, num_classes)
```

### Infrastructure role

`predict_classes` also handles:

```txt
probability computation
row-wise argmax
output vector allocation
```

## `categorical_cross_entropy`

```cpp
double categorical_cross_entropy(const Matrix& X, const Vector& y) const;
```

### Mathematical role

`categorical_cross_entropy` evaluates predicted class probabilities against multiclass targets.

Math implemented:

```txt
P = predict_proba(X)
CCE = -(1 / n) * sum_i log(P_i,true_class)
```

Conceptually:

```txt
high probability assigned to true class:
  low loss

low probability assigned to true class:
  high loss
```

To avoid numerical instability, implementations often clamp probabilities away from exactly `0`.

### Infrastructure role

It also handles:

```txt
fitted-state validation
target validation
matching row/target size validation
class-index validation
numeric-stability safeguards
```

## `weights`

```cpp
const Matrix& weights() const;
```

### Mathematical role

Returns the learned weight matrix:

```txt
W
```

Expected shape:

```txt
W.rows() == number of features
W.cols() == num_classes
```

Interpretation:

```txt
W.col(j):
  linear weights for class j
```

For standardized features:

```txt
positive W(feature, class):
  increasing that feature increases the logit for that class

negative W(feature, class):
  increasing that feature decreases the logit for that class
```

Weights are class-specific and should be interpreted relative to the other class logits.

### Infrastructure role

Also checks that the model has been fitted before exposing learned parameters.

## `bias`

```cpp
const Vector& bias() const;
```

### Mathematical role

Returns the learned class-intercept vector:

```txt
b
```

Expected shape:

```txt
b.size() == num_classes
```

Each bias term shifts the logit for one class:

```txt
z_ij = w_j^T x_i + b_j
```

### Infrastructure role

Also checks fitted state.

## `num_classes`

```cpp
Eigen::Index num_classes() const;
```

### Mathematical role

Returns the number of classes used by the model.

This controls:

```txt
number of logits per sample
number of probability columns
number of weight columns
valid target label range
```

### Infrastructure role

Also helps validate output shape and prediction compatibility.

## `is_fitted`

```cpp
bool is_fitted() const;
```

### Mathematical role

None directly.

This is state-management infrastructure.

It answers whether the learned parameters:

```txt
W
b
```

are available and valid for prediction.

## `training_history`

```cpp
const SoftmaxRegressionTrainingHistory& training_history() const;
```

### Mathematical role

Returns optimization diagnostics.

The most mathematical field is:

```txt
losses
```

which tracks categorical cross-entropy or regularized categorical cross-entropy across iterations.

### Infrastructure role

Also exposes:

```txt
iterations_run
converged
```

which are optimization-process metadata.

## `options`

```cpp
const SoftmaxRegressionOptions& options() const;
```

### Mathematical role

Exposes the configuration used for optimization and regularization.

Math-related options include:

```txt
learning_rate
tolerance
regularization
```

Infrastructure-related options include:

```txt
max_iterations
store_loss_history
```

## Important internal math concepts

## Logits

The logits are raw class scores:

```txt
Z = XW + b
```

Each sample receives one score per class.

Logits are useful because they allow the model to represent class preference before normalization.

## Softmax

Softmax maps class logits into a probability distribution:

```txt
softmax(z)_j = exp(z_j) / sum_k exp(z_k)
```

Properties:

```txt
all probabilities are positive
each row sums to 1
larger logits produce larger probabilities
adding the same constant to all logits in a row does not change probabilities
```

## One-hot targets

Softmax training uses one-hot class indicators:

```txt
Y_ij = 1 if y_i == j else 0
```

This lets the multiclass loss be written compactly as:

```txt
CCE = -(1 / n) * sum_i sum_j Y_ij log(P_ij)
```

## Categorical cross-entropy

Categorical cross-entropy measures how much probability the model assigns to the true class.

For one sample:

```txt
loss_i = -log(P_i,true_class)
```

Effects:

```txt
true class probability near 1:
  low loss

true class probability near 0:
  high loss
```

## Gradient signal

For softmax regression with categorical cross-entropy, the main gradient signal simplifies to:

```txt
P - Y
```

This makes the vectorized gradient:

```txt
dL/dW = (1 / n) * X^T (P - Y)
dL/db = column_mean(P - Y)
```

This is the multiclass analogue of logistic regression's `p - y`.

## Decision boundaries

With softmax regression, the predicted class is:

```txt
argmax_j (w_j^T x + b_j)
```

The boundary between two classes `a` and `b` occurs where:

```txt
w_a^T x + b_a = w_b^T x + b_b
```

Rearranged:

```txt
(w_a - w_b)^T x + (b_a - b_b) = 0
```

So softmax regression learns linear boundaries between pairs of classes.

## Ridge regularization

Ridge adds an L2 penalty to discourage large class-weight values:

```txt
lambda * ||W||^2
```

Effect:

```txt
shrinks weights
reduces sensitivity to noisy/correlated features
may improve generalization
```

It changes the weight gradient by adding:

```txt
2 * lambda * W
```

## Method classification

| Method / Struct | Model math | Optimization math | Probability math | Metrics/loss math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `SoftmaxRegressionOptions` | Partial | Yes | No | No | Yes |
| `SoftmaxRegressionTrainingHistory` | No | Diagnostic | No | Diagnostic | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Yes | Yes | Yes | Yes | Yes |
| `logits` | Yes | No | No | No | Yes |
| `predict_proba` | Yes | No | Yes | No | Yes |
| `predict_classes` | Yes | No | Yes | No | Yes |
| `categorical_cross_entropy` | Yes | No | Yes | Yes | Yes |
| `weights` | Yes | No | No | No | Yes |
| `bias` | Yes | No | No | No | Yes |
| `num_classes` | Partial | No | No | No | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `training_history` | No | Diagnostic | No | Diagnostic | Yes |
| `options` | Partial | Partial | No | No | Yes |

## Output files and math meaning

## `metrics.csv`

Relevant rows:

```txt
model = SoftmaxRegression
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

Macro metrics help evaluate class-balanced behavior.

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
  argmax softmax class prediction

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
probability_class_j:
  softmax probability assigned to class j
```

Each row should sum approximately to:

```txt
1.0
```

## `loss_history.csv`

Relevant columns:

```txt
iteration
loss
```

Math meaning:

```txt
iteration:
  gradient descent step index

loss:
  categorical cross-entropy or regularized categorical cross-entropy
```

This is used to inspect optimization behavior.

## Practical interpretation

`SoftmaxRegression` is the natural multiclass extension of logistic regression.

It is most useful when:

```txt
features have roughly linear class-separation structure
class probabilities are useful
a simple multiclass baseline is needed
the user wants to understand neural-network output layers
```

For the Phase 11 Wine workflow, strong softmax performance is expected because the dataset is relatively well-separated after feature standardization.

If softmax regression performs poorly, possible causes include:

```txt
nonlinear class boundaries
overlapping classes
poor feature scaling
incorrect class encoding
insufficient optimization
class imbalance
```

## Summary

`SoftmaxRegression` maps to the following mathematical pipeline:

```txt
data matrix X
multiclass target vector y
        ↓
class logits XW + b
        ↓
softmax probabilities
        ↓
categorical cross-entropy loss
        ↓
gradient descent updates for W and b
        ↓
learned multiclass linear classifier
        ↓
argmax predictions and multiclass metrics
```

The core math lives in:

```txt
fit
logits
predict_proba
predict_classes
categorical_cross_entropy
weights
bias
num_classes
```

The supporting infrastructure lives in:

```txt
options
training_history
is_fitted
input validation
output export workflows
```
