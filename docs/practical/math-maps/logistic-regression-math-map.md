# LogisticRegression Math Map

## Purpose

This document maps the public `LogisticRegression` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/linear_models/logistic_regression.hpp
src/linear_models/logistic_regression.cpp
include/ml/linear_models/regularization.hpp
include/ml/common/classification_metrics.hpp
```

Related practical usage doc:

```txt
docs/practical/models/logistic-regression-usage.md
```

Related theory doc:

```txt
docs/theory/logistic-regression.md
```

## Model equation

`LogisticRegression` is a binary classification model.

It starts with a linear score:

```txt
z = X * w + b
```

For one sample:

```txt
z_i = w_1 x_i1 + w_2 x_i2 + ... + w_d x_id + b
```

Then it converts the score into a probability using the sigmoid function:

```txt
p_i = sigmoid(z_i) = 1 / (1 + exp(-z_i))
```

The probability is interpreted as:

```txt
p_i = P(y_i = 1 | x_i)
```

Class prediction is obtained by thresholding:

```txt
y_pred_i = 1 if p_i >= threshold else 0
```

The default threshold is usually:

```txt
threshold = 0.5
```

## Training objective

The training objective is binary cross-entropy:

```txt
BCE = -(1 / n) * sum_i [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
```

where:

```txt
p_i = sigmoid(w^T x_i + b)
```

With Ridge regularization enabled, the objective becomes:

```txt
loss = BCE + lambda * ||w||^2
```

The bias is usually not regularized.

## Public API to math mapping

## `LogisticRegressionOptions`

### Mathematical role

`LogisticRegressionOptions` controls the optimization process and optional regularization.

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

## `LogisticRegressionTrainingHistory`

### Mathematical role

`LogisticRegressionTrainingHistory` records the behavior of the optimization process.

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

This is diagnostic infrastructure connected to the binary cross-entropy objective.

It does not change the model mathematics.

## Constructor

```cpp
LogisticRegression();
explicit LogisticRegression(LogisticRegressionOptions options);
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
void fit(const Matrix& X, const Vector& y);
```

### Mathematical role

`fit` is the core training method.

It estimates:

```txt
weights w
bias b
```

by minimizing binary cross-entropy.

Forward computation:

```txt
z = Xw + b
p = sigmoid(z)
```

Loss:

```txt
BCE = -(1 / n) * sum_i [y_i log(p_i) + (1 - y_i) log(1 - p_i)]
```

For logistic regression, the residual-like training signal is:

```txt
p - y
```

The gradients are:

```txt
dL/dw = (1 / n) * X^T (p - y)
dL/db = (1 / n) * sum_i (p_i - y_i)
```

Gradient descent updates:

```txt
w := w - learning_rate * dL/dw
b := b - learning_rate * dL/db
```

With Ridge regularization:

```txt
L(w, b) = BCE + lambda * ||w||^2
```

The weight gradient receives an additional term:

```txt
2 * lambda * w
```

### What `fit` does mathematically

`fit` implements:

```txt
1. initialize weights and bias
2. compute logits
3. apply sigmoid to obtain probabilities
4. compute binary cross-entropy or regularized loss
5. compute gradients with respect to weights and bias
6. update weights and bias through gradient descent
7. repeat until convergence or max_iterations
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
input validation
shape checking
empty dataset rejection
binary target validation
finite value checks where implemented
loss-history storage
convergence flag updates
fitted-state tracking
```

These are necessary for correctness but are not part of the model equation.

## `logits`

```cpp
Vector logits(const Matrix& X) const;
```

### Mathematical role

`logits` computes the raw linear score before sigmoid.

Math implemented:

```txt
z = Xw + b
```

For one sample:

```txt
z_i = dot(x_i, w) + b
```

### Interpretation

```txt
z_i > 0:
  model leans toward class 1

z_i < 0:
  model leans toward class 0

z_i = 0:
  model predicts probability 0.5

larger |z_i|:
  stronger confidence before sigmoid thresholding
```

### Infrastructure role

`logits` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output vector allocation
```

## `predict_proba`

```cpp
Vector predict_proba(const Matrix& X) const;
```

### Mathematical role

`predict_proba` converts logits into probabilities with the sigmoid function.

Math implemented:

```txt
z = Xw + b
p = sigmoid(z)
```

where:

```txt
p_i = 1 / (1 + exp(-z_i))
```

The result is:

```txt
P(y = 1 | x)
```

For binary output exports:

```txt
probability_class_1 = p
probability_class_0 = 1 - p
```

### Infrastructure role

`predict_proba` also handles:

```txt
calling logits
input validation through logits
probability vector allocation
```

## `predict_classes`

```cpp
Vector predict_classes(const Matrix& X, const double threshold = 0.5) const;
```

### Mathematical role

`predict_classes` thresholds predicted probabilities into hard binary labels.

Math implemented:

```txt
p = predict_proba(X)
y_pred_i = 1 if p_i >= threshold else 0
```

Default decision boundary:

```txt
p_i >= 0.5
```

Since:

```txt
sigmoid(0) = 0.5
```

the default class boundary is equivalent to:

```txt
logit >= 0
```

### Threshold interpretation

```txt
lower threshold:
  predicts more positives
  may increase recall
  may reduce precision

higher threshold:
  predicts fewer positives
  may increase precision
  may reduce recall
```

### Infrastructure role

`predict_classes` also handles:

```txt
probability computation
threshold application
output vector allocation
```

If threshold validation exists, that is infrastructure.

## `binary_cross_entropy`

```cpp
double binary_cross_entropy(const Matrix& X, const Vector& y) const;
```

### Mathematical role

`binary_cross_entropy` evaluates the probability predictions against binary targets.

Math implemented:

```txt
p = predict_proba(X)
BCE = -(1 / n) * sum_i [y_i log(p_i) + (1 - y_i) log(1 - p_i)]
```

To avoid numerical instability, implementations often clamp probabilities away from exactly `0` and `1`.

Conceptually:

```txt
p_i close to y_i:
  low loss

p_i confidently wrong:
  high loss
```

### Infrastructure role

It also handles:

```txt
fitted-state validation
target validation
matching row/target size validation
numeric-stability safeguards
```

## `weights`

```cpp
const Vector& weights() const;
```

### Mathematical role

Returns the learned parameter vector:

```txt
w
```

Each element corresponds to one feature.

For standardized features:

```txt
positive weight:
  increasing the feature increases the log-odds of class 1

negative weight:
  increasing the feature decreases the log-odds of class 1

larger absolute weight:
  stronger linear influence on the logit
```

The relationship is on the logit/log-odds scale, not directly on the probability scale.

### Infrastructure role

Also checks that the model has been fitted before exposing learned parameters.

## `bias`

```cpp
double bias() const;
```

### Mathematical role

Returns the learned intercept:

```txt
b
```

The bias shifts the decision boundary.

For one sample:

```txt
z_i = dot(x_i, w) + b
p_i = sigmoid(z_i)
```

### Infrastructure role

Also checks fitted state.

## `is_fitted`

```cpp
bool is_fitted() const;
```

### Mathematical role

None directly.

This is state-management infrastructure.

It answers whether the learned parameters:

```txt
w
b
```

are available and valid for prediction.

## `training_history`

```cpp
const LogisticRegressionTrainingHistory& training_history() const;
```

### Mathematical role

Returns optimization diagnostics.

The most mathematical field is:

```txt
losses
```

which tracks binary cross-entropy or regularized binary cross-entropy across iterations.

### Infrastructure role

Also exposes:

```txt
iterations_run
converged
```

which are optimization-process metadata.

## `options`

```cpp
const LogisticRegressionOptions& options() const;
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

## Logit

The logit is the raw linear score:

```txt
z = Xw + b
```

It is also the log-odds of the positive class:

```txt
z = log(p / (1 - p))
```

This is why logistic regression is still a linear model: the log-odds are linear in the features.

## Sigmoid

The sigmoid maps any real-valued logit to a probability:

```txt
sigmoid(z) = 1 / (1 + exp(-z))
```

Properties:

```txt
z = 0:
  sigmoid(z) = 0.5

z large positive:
  sigmoid(z) approaches 1

z large negative:
  sigmoid(z) approaches 0
```

## Binary cross-entropy

Binary cross-entropy measures how well predicted probabilities match binary targets.

For one sample:

```txt
loss_i = -[y_i log(p_i) + (1 - y_i) log(1 - p_i)]
```

Effects:

```txt
confident correct prediction:
  low loss

uncertain prediction:
  moderate loss

confident wrong prediction:
  high loss
```

## Gradient signal

For logistic regression with BCE, the main gradient signal simplifies to:

```txt
p - y
```

This makes the vectorized gradient:

```txt
dL/dw = (1 / n) * X^T (p - y)
dL/db = mean(p - y)
```

## Decision boundary

With threshold `0.5`, the decision boundary is:

```txt
sigmoid(Xw + b) = 0.5
```

which is equivalent to:

```txt
Xw + b = 0
```

So logistic regression learns a linear decision boundary in feature space.

## Ridge regularization

Ridge adds an L2 penalty to discourage large weights:

```txt
lambda * ||w||^2
```

Effect:

```txt
shrinks weights
reduces sensitivity to noisy/correlated features
may improve generalization
```

It changes the weight gradient by adding:

```txt
2 * lambda * w
```

## Method classification

| Method / Struct | Model math | Optimization math | Probability math | Metrics/loss math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `LogisticRegressionOptions` | Partial | Yes | No | No | Yes |
| `LogisticRegressionTrainingHistory` | No | Diagnostic | No | Diagnostic | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Yes | Yes | Yes | Yes | Yes |
| `logits` | Yes | No | No | No | Yes |
| `predict_proba` | Yes | No | Yes | No | Yes |
| `predict_classes` | Yes | No | Yes | No | Yes |
| `binary_cross_entropy` | Yes | No | Yes | Yes | Yes |
| `weights` | Yes | No | No | No | Yes |
| `bias` | Yes | No | No | No | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `training_history` | No | Diagnostic | No | Diagnostic | Yes |
| `options` | Partial | Partial | No | No | Yes |

## Output files and math meaning

## `metrics.csv`

Relevant rows:

```txt
model = LogisticRegression
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
  actual binary label

y_pred:
  thresholded class prediction

correct:
  1 if y_pred == y_true else 0
```

## `probabilities.csv`

Relevant columns:

```txt
probability_class_0
probability_class_1
```

Math meaning:

```txt
probability_class_1:
  sigmoid(Xw + b)

probability_class_0:
  1 - sigmoid(Xw + b)
```

## `decision_scores.csv`

Relevant column:

```txt
decision_score
```

Math meaning:

```txt
decision_score = logit = Xw + b
```

The default threshold decision is:

```txt
decision_score >= 0 -> class 1
decision_score < 0  -> class 0
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
  binary cross-entropy or regularized binary cross-entropy
```

This is used to inspect optimization behavior.

## Practical interpretation

`LogisticRegression` is a simple but powerful binary classification baseline.

It is most useful when:

```txt
features have a roughly linear relationship with log-odds
probability outputs are useful
decision threshold analysis matters
interpretability is important
```

In imbalanced datasets, the default threshold may produce poor recall or poor precision depending on the score distribution.

For the Phase 11 NASA KC1 software-defect workflow, logistic regression provides a stable linear baseline, but accuracy alone can be misleading. Precision, recall, and F1 are more informative for understanding defect detection.

## Summary

`LogisticRegression` maps to the following mathematical pipeline:

```txt
data matrix X
binary target vector y
        ↓
linear logits Xw + b
        ↓
sigmoid probabilities
        ↓
binary cross-entropy loss
        ↓
gradient descent updates for w and b
        ↓
learned probabilistic binary classifier
        ↓
thresholded predictions and classification metrics
```

The core math lives in:

```txt
fit
logits
predict_proba
predict_classes
binary_cross_entropy
weights
bias
```

The supporting infrastructure lives in:

```txt
options
training_history
is_fitted
input validation
output export workflows
```
