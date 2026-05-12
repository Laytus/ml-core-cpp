# LinearRegression Usage

## What the model does

`LinearRegression` fits a linear relationship between numeric input features and a continuous numeric target.

It models predictions as:

```txt
y_pred = X * weights + bias
```

In practical terms, it is used when the target is a real-valued quantity, such as a price, return, measurement, or score.

In this project, `LinearRegression` is also the base model used to expose Ridge-style regularized linear regression behavior through the regularization options.

## Supported task type

`LinearRegression` supports:

```txt
regression
```

It does not support classification, probabilities, clustering, or dimensionality reduction.

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
ml::Matrix X(3, 2);
X << 1.0, 2.0,
     2.0, 3.0,
     3.0, 4.0;

ml::Vector y(3);
y << 3.0, 5.0, 7.0;
```

For real dataset workflows, use the CSV loader to create `X` and `y` from a processed CSV file.

## Preprocessing usually needed

Linear regression is sensitive to feature scale when trained with gradient descent.

Recommended preprocessing:

```txt
standardize numeric features
reject missing values or preprocess them before loading
avoid non-numeric columns
use train-fitted preprocessing statistics only
```

For real workflows in this project, feature standardization is handled in the workflow layer before training.

For stock OHLCV regression, the processed dataset currently uses engineered numeric features such as:

```txt
return_1d
return_5d
volatility_5d
range_pct
volume_change_1d
```

and the target:

```txt
target_next_return
```

## How to instantiate the model

Basic unregularized model:

```cpp
#include "ml/linear_models/linear_regression.hpp"

ml::LinearRegressionOptions options;
options.learning_rate = 0.01;
options.max_iterations = 1000;
options.tolerance = 1e-8;
options.store_loss_history = true;

ml::LinearRegression model(options);
```

Default constructor is also available:

```cpp
ml::LinearRegression model;
```

## Ridge-style regularized configuration

Ridge behavior is configured through `RegularizationConfig`:

```cpp
#include "ml/linear_models/linear_regression.hpp"
#include "ml/linear_models/regularization.hpp"

ml::LinearRegressionOptions options;
options.learning_rate = 0.01;
options.max_iterations = 1000;
options.tolerance = 1e-8;
options.regularization = ml::RegularizationConfig::ridge(0.01);
options.store_loss_history = true;

ml::LinearRegression ridge_model(options);
```

Use Ridge when you want to penalize large weights and reduce sensitivity to noisy or correlated features.

## How to call `fit`

```cpp
model.fit(X_train, y_train);
```

Expected behavior:

```txt
- validates non-empty matrix/vector inputs
- validates matching number of rows
- initializes weights and bias
- trains using gradient descent
- stores loss history if enabled
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

## How to inspect learned parameters

After fitting:

```cpp
const ml::Vector& weights = model.weights();
double bias = model.bias();
```

These represent the learned linear relationship.

Example interpretation:

```txt
positive weight:
  increasing the feature tends to increase the prediction

negative weight:
  increasing the feature tends to decrease the prediction

larger absolute weight:
  stronger linear influence, assuming features are similarly scaled
```

## How to inspect training history

If `store_loss_history` is enabled:

```cpp
const auto& history = model.training_history();

const std::vector<double>& losses = history.losses;
std::size_t iterations = history.iterations_run;
bool converged = history.converged;
```

This is useful for checking whether training is stable and whether the loss decreased over time.

## How to read outputs

In Phase 11 practical workflows, `LinearRegression` writes results through the regression comparison workflow.

Main output files:

```txt
outputs/practical-exercises/regression/metrics.csv
outputs/practical-exercises/regression/predictions.csv
outputs/practical-exercises/regression/loss_history.csv
```

### `metrics.csv`

Contains rows such as:

```txt
run_id,workflow,dataset,model,split,metric,value
linear_regression_baseline,regression,stock_ohlcv_engineered,LinearRegression,test,mse,...
linear_regression_baseline,regression,stock_ohlcv_engineered,LinearRegression,test,rmse,...
linear_regression_baseline,regression,stock_ohlcv_engineered,LinearRegression,test,mae,...
linear_regression_baseline,regression,stock_ohlcv_engineered,LinearRegression,test,r2,...
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

This is used by notebooks to plot training loss.

## Practical workflow example

`LinearRegression` is used in the Phase 11 regression workflow:

```txt
src/workflows/regression_comparison.cpp
include/ml/workflows/regression_comparison.hpp
```

The corresponding notebook is:

```txt
notebooks/practical-workflows/01_regression_outputs.ipynb
```

The notebook visualizes:

```txt
metrics comparison
predicted vs true values
residual distributions
loss history
```

## Common mistakes to avoid

### 1. Using unscaled features with gradient descent

Linear regression trained with gradient descent can converge poorly if features have very different scales.

Prefer standardization.

### 2. Interpreting weights without considering scaling

Weights are easier to compare when features are standardized. Without scaling, large or small weights may reflect feature units rather than true importance.

### 3. Calling `predict` before `fit`

The model must be trained before prediction.

### 4. Expecting good results on noisy targets automatically

Linear regression can work well when the feature-target relationship is approximately linear. It may perform poorly on noisy or nonlinear tasks.

In the stock OHLCV workflow, next-day return prediction is difficult, and negative R2 values are possible.

### 5. Treating Ridge as a different model family

In this project, Ridge behavior is exposed as regularized `LinearRegression`, not as a completely separate core model implementation.

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating this model are:

```txt
outputs/practical-exercises/regression/metrics.csv
outputs/practical-exercises/regression/predictions.csv
outputs/practical-exercises/regression/loss_history.csv
```

The main visualization notebook is:

```txt
notebooks/practical-workflows/01_regression_outputs.ipynb
```

The main workflow implementation is:

```txt
include/ml/workflows/regression_comparison.hpp
src/workflows/regression_comparison.cpp
```

## When to use this model

Use `LinearRegression` when:

```txt
the target is continuous
you want a simple interpretable baseline
you want to test whether a linear relationship exists
you want a fast regression model
you want to compare unregularized vs Ridge behavior
```

Avoid relying on it alone when:

```txt
the relationship is strongly nonlinear
the target is extremely noisy
features contain strong interactions not represented directly
the task requires classification probabilities or class labels
```
