# LinearRegression Math Map

## Purpose

This document maps the public `LinearRegression` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/linear_models/linear_regression.hpp
src/linear_models/linear_regression.cpp
include/ml/linear_models/regularization.hpp
include/ml/common/math_ops.hpp
include/ml/common/regression_metrics.hpp
```

Related practical usage doc:

```txt
docs/practical/models/linear-regression-usage.md
```

Related theory doc:

```txt
docs/theory/linear-models.md
```

## Model equation

`LinearRegression` models a continuous target as a linear function of the input features:

```txt
y_pred = X * w + b
```

For one sample:

```txt
y_pred_i = w_1 x_i1 + w_2 x_i2 + ... + w_d x_id + b
```

where:

```txt
X = feature matrix
w = weight vector
b = scalar bias
y_pred = predicted target vector
```

The training objective is based on mean squared error:

```txt
MSE = (1 / n) * sum_i (y_pred_i - y_i)^2
```

With Ridge regularization enabled, the objective becomes:

```txt
loss = MSE + lambda * ||w||^2
```

With Lasso regularization enabled, the objective conceptually adds:

```txt
lambda * ||w||_1
```

depending on the implemented regularization branch.

## Public API to math mapping

## `LinearRegressionOptions`

### Mathematical role

`LinearRegressionOptions` controls the optimization process and optional regularization.

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

This struct is mostly optimization configuration, not the model equation itself.

## `LinearRegressionTrainingHistory`

### Mathematical role

`LinearRegressionTrainingHistory` records the behavior of the optimization process.

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

This is diagnostic infrastructure connected to the loss function.

It does not change the model mathematics.

## Constructor

```cpp
LinearRegression();
explicit LinearRegression(LinearRegressionOptions options);
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

by minimizing the training objective.

For unregularized linear regression, the objective is:

```txt
L(w, b) = (1 / n) * ||Xw + b - y||^2
```

where:

```txt
n = number of samples
```

The residual vector is:

```txt
r = y_pred - y = Xw + b - y
```

The gradients are:

```txt
dL/dw = (2 / n) * X^T r
dL/db = (2 / n) * sum_i r_i
```

Gradient descent updates:

```txt
w := w - learning_rate * dL/dw
b := b - learning_rate * dL/db
```

With Ridge regularization:

```txt
L(w, b) = MSE + lambda * ||w||^2
```

The weight gradient receives an additional term:

```txt
2 * lambda * w
```

The bias is usually not regularized.

### What `fit` does mathematically

`fit` implements:

```txt
1. initialize weights and bias
2. compute predictions
3. compute residuals
4. compute MSE or regularized loss
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
loss-history storage
convergence flag updates
fitted-state tracking
```

These are necessary for correctness but are not part of the model equation.

## `predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` applies the learned linear function to new samples.

Math implemented:

```txt
y_pred = X * w + b
```

For each row:

```txt
y_pred_i = dot(x_i, w) + b
```

### What `predict` does mathematically

`predict` implements the forward pass of linear regression:

```txt
matrix-vector multiplication
bias addition
```

### What `predict` does as infrastructure

`predict` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output vector allocation
```

## `score_mse`

```cpp
double score_mse(const Matrix& X, const Vector& y);
```

### Mathematical role

`score_mse` evaluates prediction error using mean squared error.

Math implemented:

```txt
predictions = predict(X)
MSE = (1 / n) * sum_i (predictions_i - y_i)^2
```

### What `score_mse` does mathematically

It combines:

```txt
linear forward pass
residual computation
mean squared error
```

### What `score_mse` does as infrastructure

It also handles:

```txt
input validation
matching prediction/target size validation
delegation to reusable metric utilities
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

For standardized features, larger absolute values suggest stronger linear influence.

Math meaning:

```txt
positive weight:
  increasing the feature increases the prediction

negative weight:
  increasing the feature decreases the prediction
```

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

The bias shifts all predictions by a constant amount.

For one sample:

```txt
y_pred_i = dot(x_i, w) + b
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
const LinearRegressionTrainingHistory& training_history() const;
```

### Mathematical role

Returns optimization diagnostics.

The most mathematical field is:

```txt
losses
```

which tracks the training objective across iterations.

For unregularized training, each value is usually:

```txt
MSE
```

For regularized training, each value may be:

```txt
MSE + regularization penalty
```

depending on the implementation.

### Infrastructure role

Also exposes:

```txt
iterations_run
converged
```

which are optimization-process metadata.

## `options`

```cpp
const LinearRegressionOptions& options() const;
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

## Linear prediction

Implemented through reusable math utilities or direct Eigen operations.

Math:

```txt
Xw + b
```

This is the core forward computation used in both training and prediction.

## Residuals

Residuals are:

```txt
r = y_pred - y
```

They measure signed prediction error.

Residuals are used for:

```txt
MSE computation
gradient computation
residual plots in notebooks
```

## Mean squared error

MSE is:

```txt
MSE = mean(r^2)
```

It penalizes large errors more strongly because the residual is squared.

Used in:

```txt
training objective
score_mse
regression workflow metrics
```

## Gradient descent

Gradient descent is the optimization method used to find the learned parameters.

Update rule:

```txt
parameter := parameter - learning_rate * gradient
```

For linear regression:

```txt
w := w - alpha * dL/dw
b := b - alpha * dL/db
```

where:

```txt
alpha = learning_rate
```

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

## Lasso regularization

Lasso adds an L1 penalty:

```txt
lambda * ||w||_1
```

Effect:

```txt
encourages sparse weights
can push some weights toward zero
```

If implemented through a simple gradient/subgradient path, it should be interpreted as educational rather than a full coordinate-descent Lasso solver.

## Method classification

| Method / Struct | Model math | Optimization math | Metrics math | Infrastructure |
|---|---:|---:|---:|---:|
| `LinearRegressionOptions` | Partial | Yes | No | Yes |
| `LinearRegressionTrainingHistory` | No | Diagnostic | No | Yes |
| Constructor | No | No | No | Yes |
| `fit` | Yes | Yes | Yes | Yes |
| `predict` | Yes | No | No | Yes |
| `score_mse` | Yes | No | Yes | Yes |
| `weights` | Yes | No | No | Yes |
| `bias` | Yes | No | No | Yes |
| `is_fitted` | No | No | No | Yes |
| `training_history` | No | Diagnostic | No | Yes |
| `options` | Partial | Partial | No | Yes |

## Output files and math meaning

## `metrics.csv`

Relevant rows:

```txt
model = LinearRegression
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
  fraction of target variance explained relative to a mean baseline
```

## `predictions.csv`

Relevant columns:

```txt
y_true
y_pred
error
```

Math meaning:

```txt
y_true:
  actual target value

y_pred:
  Xw + b

error:
  y_pred - y_true
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
  training objective value at that step
```

This is used to inspect optimization behavior.

## Practical interpretation

`LinearRegression` is mathematically simple and useful as a baseline.

If it performs well, the dataset likely contains a strong approximately linear relationship.

If it performs poorly, possible causes include:

```txt
weak feature-target relationship
nonlinear structure
missing interaction features
noisy target
insufficient preprocessing
poor optimization settings
```

In the Phase 11 stock OHLCV workflow, the model tends to predict values close to zero and can produce negative R2, which suggests that next-day return prediction from the current simple engineered features is difficult.

## Summary

`LinearRegression` maps cleanly to the following mathematical pipeline:

```txt
data matrix X
target vector y
        ↓
linear prediction Xw + b
        ↓
residuals y_pred - y
        ↓
MSE or regularized MSE
        ↓
gradient descent updates for w and b
        ↓
learned linear model
        ↓
predictions and regression metrics
```

The core math lives in:

```txt
fit
predict
score_mse
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
