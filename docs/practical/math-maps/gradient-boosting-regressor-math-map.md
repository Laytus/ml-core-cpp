# GradientBoostingRegressor Math Map

## Purpose

This document maps the public `GradientBoostingRegressor` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/trees/gradient_boosting.hpp
src/trees/gradient_boosting.cpp
include/ml/trees/regression_tree.hpp
src/trees/regression_tree.cpp
include/ml/common/math_ops.hpp
include/ml/common/regression_metrics.hpp
```

Related practical usage doc:

```txt
docs/practical/models/gradient-boosting-regressor-usage.md
```

Related theory docs:

```txt
docs/theory/trees-and-ensembles.md
docs/theory/tree-ensembles.md
```

## Model idea

`GradientBoostingRegressor` is an additive ensemble regression model.

Instead of fitting one large model directly, it builds a sequence of small regression trees.

Each new tree tries to correct the errors left by the current ensemble.

The model starts with a constant prediction:

```txt
F_0(x) = mean(y)
```

Then each boosting stage adds a new regression tree:

```txt
F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
```

where:

```txt
F_m(x) = ensemble prediction after m trees
h_m(x) = regression tree fitted at stage m
learning_rate = shrinkage factor
```

For squared-error regression, each tree is fitted to residuals:

```txt
residual_i = y_i - F_{m-1}(x_i)
```

The final prediction is:

```txt
F_M(x) = F_0(x) + learning_rate * sum_m h_m(x)
```

where:

```txt
M = number of estimators
```

## Training objective

For regression, the main objective is mean squared error:

```txt
MSE = (1 / n) * sum_i (y_i - F(x_i))^2
```

Gradient boosting can be interpreted as stage-wise optimization of this loss.

For squared error, the negative gradient is the residual:

```txt
negative_gradient_i = y_i - F_{m-1}(x_i)
```

So fitting a tree to residuals is equivalent to fitting a tree to the negative gradient of the squared-error loss.

## Public API to math mapping

## `GradientBoostingRegressorOptions`

```cpp
struct GradientBoostingRegressorOptions {
    std::size_t n_estimators;
    double learning_rate;
    std::size_t max_depth;
    std::size_t min_samples_split;
    std::size_t min_samples_leaf;
    unsigned int random_seed;
};
```

### Mathematical role

`GradientBoostingRegressorOptions` controls the additive ensemble construction process.

Important fields:

```txt
n_estimators
learning_rate
max_depth
min_samples_split
min_samples_leaf
random_seed
```

Math meaning:

```txt
n_estimators:
  number of boosting stages / number of regression trees

learning_rate:
  shrinkage multiplier applied to each tree contribution

max_depth:
  maximum depth of each weak regression tree

min_samples_split:
  minimum samples required to split a tree node

min_samples_leaf:
  minimum samples required in each leaf

random_seed:
  deterministic training behavior where randomness is used
```

These options define the capacity and update behavior of the boosted ensemble.

## `validate_gradient_boosting_regressor_options`

```cpp
void validate_gradient_boosting_regressor_options(
    const GradientBoostingRegressorOptions& options,
    const std::string& context
);
```

### Mathematical role

Ensures the boosting configuration is meaningful.

Examples of mathematical requirements:

```txt
n_estimators >= 1
learning_rate > 0
max_depth >= 1
min_samples_split >= 2
min_samples_leaf >= 1
```

Invalid values would make the ensemble impossible or meaningless.

For example:

```txt
n_estimators = 0:
  no additive corrections can be learned

learning_rate <= 0:
  tree contributions do not move the model in a useful minimizing direction
```

### Infrastructure role

It also provides:

```txt
consistent error messages
context-specific validation
early failure for invalid configuration
```

## Constructor

```cpp
GradientBoostingRegressor();
explicit GradientBoostingRegressor(
    GradientBoostingRegressorOptions options
);
```

### Mathematical role

The constructor does not train trees.

It only stores the ensemble configuration:

```txt
number of estimators
learning rate
weak-tree depth
weak-tree stopping rules
random seed
```

Math implemented:

```txt
none directly
```

Infrastructure role:

```txt
configure boosting behavior
validate options
```

## `fit`

```cpp
void fit(const Matrix& X, const Vector& y);
```

### Mathematical role

`fit` is the core training method.

It builds an additive ensemble of regression trees.

The initial model is:

```txt
F_0(x) = mean(y_train)
```

Then, for each boosting stage `m`:

```txt
current_prediction_i = F_{m-1}(x_i)
residual_i = y_i - current_prediction_i
```

A shallow `DecisionTreeRegressor` is fitted to:

```txt
X -> residuals
```

Then the ensemble is updated:

```txt
F_m(x_i) = F_{m-1}(x_i) + learning_rate * h_m(x_i)
```

where:

```txt
h_m = tree fitted to residuals at stage m
```

The training loss is usually tracked as:

```txt
MSE(y, F_m(X))
```

### What `fit` does mathematically

`fit` implements:

```txt
1. initialize constant prediction with mean(y)
2. initialize current predictions to that constant
3. for each boosting stage:
   - compute residuals y - current_predictions
   - fit a shallow DecisionTreeRegressor to residuals
   - predict residual corrections with that tree
   - update current predictions using learning_rate
   - compute and store training loss
4. store all fitted trees
5. store the initial prediction
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
option validation
input validation
shape checking
empty dataset rejection
feature-count tracking
tree storage
loss-history storage
fitted-state tracking
```

These are necessary for correctness but are not the additive-model equation itself.

## `predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` applies the learned additive model to new samples.

It starts with the learned initial prediction:

```txt
prediction = initial_prediction
```

Then it adds each tree contribution:

```txt
prediction += learning_rate * tree_m.predict(X)
```

Full model:

```txt
F_M(X) = initial_prediction + learning_rate * sum_m h_m(X)
```

Expected output:

```txt
predictions.size() == X.rows()
```

### What `predict` does mathematically

`predict` implements:

```txt
constant baseline prediction
tree-by-tree additive correction
learning-rate scaling of each correction
```

### What `predict` does as infrastructure

`predict` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output vector allocation
looping over fitted trees
```

## `is_fitted`

```cpp
bool is_fitted() const;
```

### Mathematical role

None directly.

This is state-management infrastructure.

It answers whether the ensemble has:

```txt
initial prediction
fitted residual trees
```

available for prediction.

## `options`

```cpp
const GradientBoostingRegressorOptions& options() const;
```

### Mathematical role

Returns the boosting configuration.

This configuration determines:

```txt
number of additive stages
shrinkage strength
weak learner complexity
```

### Infrastructure role

Useful for diagnostics and reproducibility.

## `initial_prediction`

```cpp
double initial_prediction() const;
```

### Mathematical role

Returns the constant starting prediction:

```txt
F_0(x) = mean(y_train)
```

This is the baseline value before any trees are added.

For squared-error regression, the mean target is the optimal constant prediction.

### Infrastructure role

Also exposes the fitted model state for inspection.

## `num_trees`

```cpp
std::size_t num_trees() const;
```

### Mathematical role

Returns the number of fitted residual trees:

```txt
M
```

This controls the number of additive terms in:

```txt
F_M(x) = F_0(x) + learning_rate * sum_m h_m(x)
```

### Infrastructure role

Useful for checking that the expected number of estimators was trained.

## `training_loss_history`

```cpp
const std::vector<double>& training_loss_history() const;
```

### Mathematical role

Returns the loss value after boosting stages.

For squared-error regression, this is typically:

```txt
MSE(y_train, F_m(X_train))
```

for each stage `m`.

This shows how the additive model improves on the training set as more trees are added.

### Infrastructure role

Stores diagnostic information for plotting and practical workflow exports.

## Important internal math concepts

## Additive model

Gradient boosting builds a function as a sum of simple functions:

```txt
F_M(x) = F_0(x) + learning_rate * h_1(x) + ... + learning_rate * h_M(x)
```

Each `h_m` is a regression tree.

This is why boosting can model nonlinear structure even when each individual tree is shallow.

## Initial prediction

For squared-error regression, the best constant prediction is the mean of the target:

```txt
F_0 = mean(y)
```

This minimizes:

```txt
sum_i (y_i - c)^2
```

over all constants `c`.

## Residuals

At stage `m`, residuals are:

```txt
r_i = y_i - F_{m-1}(x_i)
```

They represent the part of the target that the current ensemble has not explained.

The next tree is trained to predict these residuals.

## Negative gradient interpretation

For squared-error loss:

```txt
L = (1 / 2) * (y - F(x))^2
```

the derivative with respect to the prediction is:

```txt
dL/dF = F(x) - y
```

The negative gradient is:

```txt
-(dL/dF) = y - F(x)
```

which is exactly the residual.

So residual fitting is gradient boosting for squared-error loss.

## Weak learners

Each individual regression tree is usually intentionally small.

Common weak learner constraints:

```txt
small max_depth
minimum samples per split
minimum samples per leaf
```

A weak learner does not need to solve the full problem alone.

It only needs to provide useful correction signals.

## Learning rate / shrinkage

The learning rate scales each tree contribution:

```txt
F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
```

Interpretation:

```txt
small learning_rate:
  conservative updates
  often needs more estimators
  may generalize better

large learning_rate:
  aggressive updates
  may fit faster
  may overfit or become unstable
```

## Number of estimators

The number of estimators controls how many additive corrections are learned.

Interpretation:

```txt
too few estimators:
  underfitting

too many estimators:
  possible overfitting
  higher compute cost
```

## Training loss vs test performance

Boosting often reduces training loss as more trees are added.

However, decreasing training loss does not guarantee improved test performance.

This is why Phase 11 exports:

```txt
metrics.csv
loss_history.csv
hyperparameter_sweep.csv
```

to compare training behavior and test metrics.

## Relationship to DecisionTreeRegressor

`GradientBoostingRegressor` should reuse `DecisionTreeRegressor` as its weak learner.

The tree handles:

```txt
split search
recursive partitioning
leaf mean prediction
```

Gradient boosting handles:

```txt
residual computation
sequential tree fitting
additive updates
learning-rate shrinkage
training-loss tracking
```

The ensemble should not duplicate regression-tree split logic.

## Method classification

| Method / Struct | Regression tree math | Boosting math | Optimization/loss math | Metrics math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `GradientBoostingRegressorOptions` | Partial | Yes | Yes | No | Yes |
| `validate_gradient_boosting_regressor_options` | Partial | Partial | Partial | No | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Yes | Yes | Yes | Yes | Yes |
| `predict` | Yes | Yes | No | No | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `options` | Partial | Partial | Partial | No | Yes |
| `initial_prediction` | No | Yes | Yes | No | Yes |
| `num_trees` | No | Yes | No | No | Yes |
| `training_loss_history` | No | Diagnostic | Diagnostic | Partial | Yes |

## Output files and math meaning

## `metrics.csv`

Relevant rows:

```txt
model = GradientBoostingRegressor
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

These metrics are computed on the test split in the practical workflow.

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
  actual continuous target value

y_pred:
  F_M(x), the boosted ensemble prediction

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
  boosting stage index

loss:
  training MSE after that stage
```

This is used to inspect whether the ensemble is fitting the training data progressively.

## `hyperparameter_sweep.csv`

Relevant rows:

```txt
model = GradientBoostingRegressor
param_name = n_estimators
param_name = learning_rate
metric = mse
metric = rmse
metric = mae
metric = r2
```

Math meaning:

```txt
n_estimators:
  number of additive residual trees

learning_rate:
  shrinkage multiplier for each tree contribution

metric value:
  test performance produced by that configuration
```

## Practical interpretation

`GradientBoostingRegressor` is a nonlinear regression model built from sequential corrections.

It is most useful when:

```txt
the target has nonlinear structure
shallow trees can capture useful residual patterns
the learning rate and number of estimators are tuned together
```

If boosting improves over a single regression tree, it suggests that sequential residual correction is useful.

If it performs poorly, possible causes include:

```txt
target is extremely noisy
too few or too many estimators
learning rate too high or too low
weak trees too shallow or too deep
features do not contain predictive signal
train/test split is not appropriate for the data
```

For the Phase 11 stock OHLCV workflow, next-day return prediction is difficult. A negative test R2 does not automatically mean the implementation is wrong; it means the model underperformed a mean-target baseline on that test split.

## Summary

`GradientBoostingRegressor` maps to the following mathematical pipeline:

```txt
training matrix X
continuous target y
        ↓
initialize F_0(x) = mean(y)
        ↓
for each boosting stage:
    compute residuals y - F_{m-1}(X)
    fit DecisionTreeRegressor to residuals
    update predictions:
      F_m(X) = F_{m-1}(X) + learning_rate * h_m(X)
    record training loss
        ↓
store all residual trees
        ↓
predict by adding all tree corrections
        ↓
evaluate with regression metrics
```

The core boosting math lives in:

```txt
fit
predict
initial_prediction
num_trees
training_loss_history
residual computation
learning-rate-scaled additive updates
```

The base tree math lives in:

```txt
DecisionTreeRegressor
regression split search
leaf mean prediction
```

The supporting infrastructure lives in:

```txt
options
validate_gradient_boosting_regressor_options
is_fitted
input validation
feature-count tracking
tree storage
output export workflows
```
