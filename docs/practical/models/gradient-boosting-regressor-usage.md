# GradientBoostingRegressor Usage

## What the model does

`GradientBoostingRegressor` is an ensemble regression model.

It builds an additive model by training many small regression trees sequentially. Each new tree tries to correct the errors left by the previous ensemble.

Conceptually:

```txt
initial_prediction = mean(y_train)

for each boosting stage:
  residuals = y_train - current_predictions
  fit a small DecisionTreeRegressor to residuals
  update predictions:
    current_predictions += learning_rate * tree_predictions
```

The final prediction is:

```txt
prediction = initial_prediction + learning_rate * sum(tree_predictions)
```

In practical terms, `GradientBoostingRegressor` is useful when you want:

```txt
a nonlinear regression model
a tree-based ensemble for tabular data
better performance than a single regression tree
a model that learns additive corrections
training loss history across boosting stages
```

## Supported task type

`GradientBoostingRegressor` supports:

```txt
regression
```

It predicts continuous numeric values.

It does not support:

```txt
binary classification
multiclass classification
probability prediction
clustering
dimensionality reduction
```

For classification tree ensembles in this project, use:

```txt
RandomForestClassifier
DecisionTreeClassifier
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
y contains continuous numeric target values
```

Typical shapes:

```txt
X: n_samples x n_features
y: n_samples
```

Example:

```cpp
ml::Matrix X(4, 2);
X << 1.0, 2.0,
     2.0, 3.0,
     3.0, 4.0,
     4.0, 5.0;

ml::Vector y(4);
y << 10.0, 15.0, 21.0, 28.0;
```

For real dataset workflows, use the CSV loader to create `X` and `y` from a processed CSV file.

## Target format

The target must be a continuous numeric value.

Examples:

```txt
price
return
score
measurement
```

In the Phase 11 stock OHLCV workflow, the target is:

```txt
target_next_return
```

## Preprocessing usually needed

Tree-based models do not require feature standardization in the same way as gradient-based linear models or distance-based models.

However, preprocessing is still important.

Recommended preprocessing:

```txt
reject missing values or preprocess them before loading
avoid non-numeric feature columns
ensure target values are finite
remove obviously invalid rows
use consistent train/test split discipline
```

Feature standardization is not strictly required for gradient boosting trees, but the Phase 11 workflow may standardize features for consistent model comparisons.

For time-series-like data such as stock OHLCV rows, prefer chronological splitting for realistic evaluation.

## How to instantiate the model

Basic model:

```cpp
#include "ml/trees/gradient_boosting.hpp"

ml::GradientBoostingRegressorOptions options;
options.n_estimators = 100;
options.learning_rate = 0.1;
options.max_depth = 2;
options.min_samples_split = 2;
options.min_samples_leaf = 1;
options.random_seed = 42;

ml::GradientBoostingRegressor model(options);
```

Default constructor is also available:

```cpp
ml::GradientBoostingRegressor model;
```

## Main options

`GradientBoostingRegressorOptions` contains:

```txt
n_estimators
learning_rate
max_depth
min_samples_split
min_samples_leaf
random_seed
```

### `n_estimators`

Number of boosting stages.

Interpretation:

```txt
more estimators:
  more additive corrections
  higher model capacity
  higher compute cost
  can overfit if too large
```

### `learning_rate`

Shrinkage factor applied to each tree contribution.

Interpretation:

```txt
smaller learning_rate:
  more conservative updates
  often needs more estimators
  may generalize better

larger learning_rate:
  stronger updates
  may learn faster
  can overfit or become unstable
```

### `max_depth`

Maximum depth of each weak regression tree.

Interpretation:

```txt
small max_depth:
  weak learners
  smoother additive corrections
  less risk of overfitting

larger max_depth:
  more complex weak learners
  can capture interactions
  higher overfitting risk
```

### `min_samples_split`

Minimum number of samples required to split a node.

Interpretation:

```txt
larger value:
  fewer splits
  simpler trees
  more regularization
```

### `min_samples_leaf`

Minimum number of samples required in a leaf.

Interpretation:

```txt
larger value:
  smoother leaf predictions
  less sensitivity to outliers
  more regularization
```

## How to call `fit`

```cpp
model.fit(X_train, y_train);
```

Expected behavior:

```txt
- validates options
- validates non-empty matrix/vector inputs
- validates matching number of rows
- validates finite target values
- initializes prediction with the mean training target
- trains n_estimators regression trees sequentially
- each tree fits current residuals
- stores training loss history
- marks the model as fitted
```

The target vector must contain continuous numeric values.

## How to call `predict`

```cpp
ml::Vector predictions = model.predict(X_test);
```

Expected output:

```txt
predictions.size() == X_test.rows()
```

Each prediction is a continuous numeric value.

Calling `predict` before `fit` should be rejected.

## How to inspect model state

After fitting:

```cpp
bool fitted = model.is_fitted();
const auto& options = model.options();
double initial_prediction = model.initial_prediction();
std::size_t tree_count = model.num_trees();
```

Useful interpretation:

```txt
initial_prediction:
  mean target prediction before any boosting corrections

num_trees:
  number of fitted boosting stages
```

## How to inspect training loss history

After fitting:

```cpp
const std::vector<double>& losses = model.training_loss_history();
```

The loss history stores the training MSE after boosting stages.

It is useful for checking:

```txt
whether training loss decreases
whether additional trees keep improving training loss
whether training may be overfitting if test metrics worsen while training loss decreases
```

## How to evaluate predictions

Use regression metrics:

```cpp
#include "ml/common/math_ops.hpp"
#include "ml/common/regression_metrics.hpp"

double mse = ml::mean_squared_error(predictions, y_test);
double rmse = ml::root_mean_squared_error(predictions, y_test);
double mae = ml::mean_absolute_error(predictions, y_test);
double r2 = ml::r2_score(predictions, y_test);
```

Interpretation:

```txt
lower MSE/RMSE/MAE is better
higher R2 is better
negative R2 means worse than a mean baseline
```

## How to read outputs

In Phase 11 practical workflows, `GradientBoostingRegressor` writes results through the regression comparison and hyperparameter sweep workflows.

Main output files:

```txt
outputs/practical-exercises/regression/metrics.csv
outputs/practical-exercises/regression/predictions.csv
outputs/practical-exercises/regression/loss_history.csv
outputs/practical-exercises/regression/hyperparameter_sweep.csv
```

### `metrics.csv`

Contains rows such as:

```txt
run_id,workflow,dataset,model,split,metric,value
gradient_boosting_regressor_baseline,regression,stock_ohlcv_engineered,GradientBoostingRegressor,test,mse,...
gradient_boosting_regressor_baseline,regression,stock_ohlcv_engineered,GradientBoostingRegressor,test,rmse,...
gradient_boosting_regressor_baseline,regression,stock_ohlcv_engineered,GradientBoostingRegressor,test,mae,...
gradient_boosting_regressor_baseline,regression,stock_ohlcv_engineered,GradientBoostingRegressor,test,r2,...
```

### `predictions.csv`

Contains:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,error
```

where:

```txt
error = y_pred - y_true
```

### `loss_history.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,iteration,loss
```

For `GradientBoostingRegressor`, each iteration corresponds to a boosting stage.

### `hyperparameter_sweep.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

For gradient boosting sweeps, parameters include:

```txt
n_estimators
learning_rate
```

Example:

```txt
gb_n5_lr0.03,regression,stock_ohlcv_engineered,GradientBoostingRegressor,test,n_estimators,5,mse,...
gb_n5_lr0.03,regression,stock_ohlcv_engineered,GradientBoostingRegressor,test,learning_rate,0.03,mse,...
```

## Practical workflow example

`GradientBoostingRegressor` is used in the Phase 11 regression workflow:

```txt
include/ml/workflows/regression_comparison.hpp
src/workflows/regression_comparison.cpp
```

It is also used in the hyperparameter sweep workflow:

```txt
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

The corresponding notebooks are:

```txt
notebooks/practical-workflows/01_regression_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The notebooks visualize:

```txt
metric comparison
predicted vs true values
residual distributions
loss history
parameter-vs-metric sweep behavior
```

## Hyperparameter behavior

The most important hyperparameters are:

```txt
n_estimators
learning_rate
max_depth
min_samples_leaf
```

Common patterns:

```txt
more estimators + small learning rate:
  often more stable
  may need more compute

few estimators + high learning rate:
  faster but less conservative
  can overfit or miss smoother behavior

shallow trees:
  weak learners
  better for additive boosting behavior

deeper trees:
  can model interactions
  higher overfitting risk
```

In the Phase 11 stock OHLCV sweep, the best tested configuration was:

```txt
run_id: gb_n5_lr0.03
n_estimators: 5
learning_rate: 0.03
```

with:

```txt
mse  = 0.000530764443451
rmse = 0.0230383255349
mae  = 0.0170641325285
r2   = -0.0114576760095
```

This suggests that, for the noisy next-day stock-return target, conservative boosting worked better than more aggressive settings in the tested sweep.

## Common mistakes to avoid

### 1. Assuming more estimators always improves test performance

More estimators usually reduce training loss, but they do not always improve test metrics.

Check test MSE/RMSE/MAE/R2.

### 2. Ignoring the learning-rate / estimator trade-off

`learning_rate` and `n_estimators` should be interpreted together.

A small learning rate often requires more estimators.

A large learning rate may need fewer estimators but can generalize worse.

### 3. Using trees that are too deep

Gradient boosting usually works well with shallow trees.

Deep trees can make each boosting stage too strong and increase overfitting risk.

### 4. Interpreting negative R2 as an implementation failure

Negative R2 means the model performed worse than a mean baseline on the test split.

For noisy targets, such as next-day stock returns, this can happen even when the implementation is correct.

### 5. Forgetting to compare against baselines

A regression workflow should compare against simple baselines such as:

```txt
mean target prediction
zero return prediction for financial returns
```

This makes performance easier to interpret.

### 6. Treating training loss as enough

Training loss can decrease while test performance worsens.

Always inspect test metrics.

### 7. Calling `predict` before `fit`

The model must be trained before prediction.

Calling `predict` before `fit` should be rejected.

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating this model are:

```txt
outputs/practical-exercises/regression/metrics.csv
outputs/practical-exercises/regression/predictions.csv
outputs/practical-exercises/regression/loss_history.csv
outputs/practical-exercises/regression/hyperparameter_sweep.csv
```

The main visualization notebooks are:

```txt
notebooks/practical-workflows/01_regression_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The main workflow implementations are:

```txt
include/ml/workflows/regression_comparison.hpp
src/workflows/regression_comparison.cpp
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

The sweep interpretation doc is:

```txt
docs/practical/sweeps/regression-gradient-boosting-sweep.md
```

## When to use this model

Use `GradientBoostingRegressor` when:

```txt
the task is regression
the feature-target relationship may be nonlinear
you want a stronger model than a single regression tree
you want a tree-based ensemble for tabular data
you want to study additive residual correction
you want training loss history across boosting stages
```

Avoid relying on it alone when:

```txt
the target is extremely noisy
you have not compared against simple baselines
you need calibrated probabilities
the task is classification
the dataset requires time-series-specific validation and you are using random splits
training loss improves but test metrics do not
```
