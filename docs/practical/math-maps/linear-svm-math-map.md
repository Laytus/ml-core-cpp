# LinearSVM Math Map

## Purpose

This document maps the public `LinearSVM` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/linear_models/linear_svm.hpp
src/linear_models/linear_svm.cpp
include/ml/common/classification_metrics.hpp
```

Related practical usage doc:

```txt
docs/practical/models/linear-svm-usage.md
```

Related theory doc:

```txt
docs/theory/distance-and-kernel-thinking.md
```

## Model idea

`LinearSVM` is a binary margin-based classifier.

It learns a linear decision boundary:

```txt
score = X * w + b
```

For one sample:

```txt
score_i = w^T x_i + b
```

The predicted class is determined by the sign of the score:

```txt
y_pred_i = 1 if score_i >= 0 else 0
```

Internally, the model maps public binary labels:

```txt
0, 1
```

to SVM margin labels:

```txt
-1, +1
```

This internal mapping allows the margin condition:

```txt
y_i * score_i >= 1
```

where:

```txt
y_i ∈ {-1, +1}
```

## Training objective

The linear SVM objective combines hinge loss and L2 regularization.

For one sample:

```txt
hinge_loss_i = max(0, 1 - y_i * score_i)
```

where:

```txt
score_i = w^T x_i + b
y_i ∈ {-1, +1}
```

The training objective is:

```txt
loss = mean_i max(0, 1 - y_i * (w^T x_i + b)) + lambda * ||w||^2
```

where:

```txt
lambda = l2 regularization strength
```

The goal is to find a separating hyperplane with a large margin while penalizing margin violations.

## Public API to math mapping

## `LinearSVMOptions`

```cpp
struct LinearSVMOptions {
    double learning_rate;
    std::size_t max_epochs;
    double l2_lambda;
};
```

### Mathematical role

`LinearSVMOptions` controls the optimization process and L2 regularization.

Important fields:

```txt
learning_rate
max_epochs
l2_lambda
```

Math meaning:

```txt
learning_rate:
  optimization step size

max_epochs:
  number of full training passes

l2_lambda:
  strength of L2 weight penalty
```

The regularization term is:

```txt
l2_lambda * ||w||^2
```

Higher `l2_lambda` encourages smaller weights and a wider, more regularized margin.

## `validate_linear_svm_options`

```cpp
void validate_linear_svm_options(
    const LinearSVMOptions& options,
    const std::string& context
);
```

### Mathematical role

This function ensures that the optimization configuration is mathematically valid.

Requirements include:

```txt
learning_rate > 0
max_epochs >= 1
l2_lambda >= 0
```

Invalid values would make optimization meaningless or unstable.

### Infrastructure role

It also provides consistent error messages and context-specific validation.

## Constructor

```cpp
LinearSVM();
explicit LinearSVM(LinearSVMOptions options);
```

### Mathematical role

The constructor does not perform learning.

It only stores the training configuration:

```txt
learning_rate
max_epochs
l2_lambda
```

Math implemented:

```txt
none directly
```

Infrastructure role:

```txt
configure optimizer behavior
configure regularization strength
validate options
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

by minimizing the regularized hinge-loss objective.

Public targets are first mapped:

```txt
0 -> -1
1 -> +1
```

For each sample:

```txt
score_i = w^T x_i + b
margin_i = y_i * score_i
```

The hinge loss is active when:

```txt
margin_i < 1
```

and inactive when:

```txt
margin_i >= 1
```

If the margin condition is satisfied, only regularization affects the weight update.

If the margin condition is violated, the sample contributes a hinge-loss gradient.

### Hinge-loss gradient intuition

For one training sample:

```txt
loss_i = max(0, 1 - y_i * (w^T x_i + b))
```

If:

```txt
y_i * score_i >= 1
```

then:

```txt
hinge gradient = 0
```

If:

```txt
y_i * score_i < 1
```

then:

```txt
d hinge / dw = -y_i * x_i
d hinge / db = -y_i
```

With L2 regularization:

```txt
d regularization / dw = 2 * lambda * w
```

The bias is usually not L2-regularized.

### What `fit` does mathematically

`fit` implements:

```txt
1. validate binary labels
2. map labels from {0, 1} to {-1, +1}
3. initialize weights and bias
4. compute linear decision scores
5. evaluate margin condition y * score
6. apply hinge-loss updates for margin violations
7. apply L2 regularization to weights
8. repeat for max_epochs
9. store training loss history
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
option validation
input validation
shape checking
empty dataset rejection
feature-count tracking
deterministic training flow
fitted-state tracking
loss-history storage
```

These are necessary for correctness but are not the core margin equation.

## `decision_function`

```cpp
Vector decision_function(const Matrix& X) const;
```

### Mathematical role

`decision_function` computes the raw linear SVM score.

Math implemented:

```txt
score = Xw + b
```

For one sample:

```txt
score_i = dot(x_i, w) + b
```

### Interpretation

```txt
score_i > 0:
  model predicts class 1

score_i < 0:
  model predicts class 0

score_i = 0:
  sample lies on the decision boundary

large positive score:
  strong class-1 margin

large negative score:
  strong class-0 margin

score near 0:
  sample is near the decision boundary
```

The score is a margin-like value, not a probability.

### Infrastructure role

`decision_function` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output vector allocation
```

## `predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` converts decision scores into binary class labels.

Math implemented:

```txt
scores = decision_function(X)
y_pred_i = 1 if score_i >= 0 else 0
```

The decision boundary is:

```txt
w^T x + b = 0
```

### Infrastructure role

`predict` also handles:

```txt
calling decision_function
thresholding scores at zero
output vector allocation
```

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

## `options`

```cpp
const LinearSVMOptions& options() const;
```

### Mathematical role

Exposes the training configuration:

```txt
learning_rate
max_epochs
l2_lambda
```

These values determine the optimization behavior and regularization strength.

### Infrastructure role

Also provides access to the configuration used during training.

## `weights`

```cpp
const Vector& weights() const;
```

### Mathematical role

Returns the learned weight vector:

```txt
w
```

The weight vector is normal to the separating hyperplane.

Decision boundary:

```txt
w^T x + b = 0
```

Geometric interpretation:

```txt
w:
  direction perpendicular to the decision boundary

||w||:
  related to margin width
```

In classical SVM theory, margin width is related to:

```txt
1 / ||w||
```

or often:

```txt
2 / ||w||
```

depending on the margin convention.

### Infrastructure role

Also checks fitted state before exposing learned parameters.

## `bias`

```cpp
double bias() const;
```

### Mathematical role

Returns the learned intercept:

```txt
b
```

The bias shifts the decision boundary:

```txt
w^T x + b = 0
```

### Infrastructure role

Also checks fitted state.

## `training_loss_history`

```cpp
const std::vector<double>& training_loss_history() const;
```

### Mathematical role

Returns the objective value across training epochs.

The recorded loss should represent:

```txt
mean hinge loss + l2_lambda * ||w||^2
```

or the equivalent objective used in the implementation.

This allows inspection of whether optimization is stable.

### Infrastructure role

Stores diagnostic information for plotting and practical workflow exports.

## Important internal math concepts

## Public label encoding vs SVM label encoding

The public model API uses common binary labels:

```txt
0
1
```

The SVM margin formulation uses:

```txt
-1
+1
```

Mapping:

```txt
0 -> -1
1 -> +1
```

This mapping is necessary because the margin condition uses multiplication:

```txt
y_i * score_i
```

If the sample is correctly classified with sufficient margin:

```txt
y_i * score_i >= 1
```

## Linear decision score

The raw decision score is:

```txt
score = w^T x + b
```

This score determines both:

```txt
classification sign
margin strength
```

The predicted label is based on the sign:

```txt
score >= 0 -> class 1
score < 0  -> class 0
```

## Margin

The signed margin condition is:

```txt
y_i * score_i
```

where:

```txt
y_i ∈ {-1, +1}
```

Interpretation:

```txt
y_i * score_i > 0:
  sample is correctly classified

y_i * score_i < 0:
  sample is misclassified

y_i * score_i >= 1:
  sample is correctly classified with sufficient margin
```

## Hinge loss

Hinge loss is:

```txt
max(0, 1 - y_i * score_i)
```

Behavior:

```txt
margin >= 1:
  loss = 0

margin < 1:
  loss > 0
```

This means the model only receives hinge-loss pressure from samples that are misclassified or too close to the margin.

## L2 regularization

L2 regularization penalizes large weights:

```txt
lambda * ||w||^2
```

Effect:

```txt
encourages smaller weights
encourages wider margin
reduces overfitting risk
```

The objective balances:

```txt
margin violations
weight size
```

## Decision boundary

The learned boundary is:

```txt
w^T x + b = 0
```

Points on one side are predicted as class `1`.

Points on the other side are predicted as class `0`.

Because this is a linear SVM, the boundary is linear in the original feature space.

## Linear SVM vs Logistic Regression

Both models compute a linear score:

```txt
score = w^T x + b
```

But they interpret and train it differently.

Logistic regression:

```txt
uses sigmoid probabilities
optimizes binary cross-entropy
outputs probabilities
```

Linear SVM:

```txt
uses margin scores
optimizes hinge loss
does not output probabilities
```

## Linear SVM vs Kernel SVM

This project implements a primal linear SVM.

It does not implement:

```txt
dual optimization
support vector coefficients
SMO
kernelized decision functions
nonlinear SVM boundaries
```

Kernel SVM is intentionally deferred.

## Method classification

| Method / Struct | Model math | Margin math | Optimization math | Probability math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `LinearSVMOptions` | Partial | Partial | Yes | No | Yes |
| `validate_linear_svm_options` | Partial | Partial | Partial | No | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Yes | Yes | Yes | No | Yes |
| `decision_function` | Yes | Yes | No | No | Yes |
| `predict` | Yes | Yes | No | No | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `options` | Partial | Partial | Partial | No | Yes |
| `weights` | Yes | Yes | No | No | Yes |
| `bias` | Yes | Yes | No | No | Yes |
| `training_loss_history` | No | Diagnostic | Diagnostic | No | Yes |

## Output files and math meaning

## `metrics.csv`

Relevant rows:

```txt
model = LinearSVM
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

These metrics are computed from hard predicted labels.

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
  actual binary label in {0, 1}

y_pred:
  sign-thresholded SVM prediction in {0, 1}

correct:
  1 if y_pred == y_true else 0
```

## `decision_scores.csv`

Relevant column:

```txt
decision_score
```

Math meaning:

```txt
decision_score = Xw + b
```

The sign determines the predicted class:

```txt
decision_score >= 0 -> class 1
decision_score < 0  -> class 0
```

Magnitude indicates distance-like confidence relative to the learned hyperplane, but it is not a calibrated probability.

## `loss_history.csv`

Relevant columns:

```txt
iteration
loss
```

Math meaning:

```txt
iteration:
  training epoch index

loss:
  hinge-loss objective plus L2 regularization
```

This is used to inspect optimization behavior.

## Practical interpretation

`LinearSVM` is useful when the binary classification task can be separated reasonably well by a linear margin.

It is most useful when:

```txt
features are numeric and standardized
margin behavior matters
decision scores are useful
probability estimates are not required
```

For imbalanced datasets, the default sign threshold can still produce poor recall or poor precision.

In the Phase 11 NASA KC1 software-defect workflow, `LinearSVM` provides a margin-based comparison point against `LogisticRegression`.

Accuracy alone can be misleading, so precision, recall, and F1 should be inspected.

## Summary

`LinearSVM` maps to the following mathematical pipeline:

```txt
data matrix X
binary target vector y in {0, 1}
        ↓
map labels to {-1, +1}
        ↓
linear decision scores Xw + b
        ↓
margin values y * score
        ↓
hinge loss + L2 regularization
        ↓
training updates for w and b
        ↓
learned linear margin classifier
        ↓
sign-thresholded predictions and classification metrics
```

The core math lives in:

```txt
fit
decision_function
predict
weights
bias
training_loss_history
```

The supporting infrastructure lives in:

```txt
options
validate_linear_svm_options
is_fitted
input validation
output export workflows
```
