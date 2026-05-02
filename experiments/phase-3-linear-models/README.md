# Phase 3 — Linear Models Experiments

This folder contains manual sanity checks and experiments for Phase 3 of ML Core.

Phase 3 introduces linear models properly, starting with multivariate linear regression and Ridge regularization.

The goal is to validate that linear regression works as a serious dataset-level model using:

```txt
Eigen matrix operations
vectorized prediction
vectorized MSE
batch gradient descent
Ridge regularization
Phase 2 evaluation infrastructure
```

---

## Scope

This experiment group validates:

```txt
RegularizationConfig
LinearRegressionOptions
LinearRegressionTrainingHistory
LinearRegression
unregularized linear regression
Ridge regression
loss history
convergence behavior
model prediction shape
MSE scoring
```

The initial sanity checks focus on correctness.

Later Phase 3 experiments should compare:

```txt
unregularized vs Ridge
scaled vs unscaled data
different learning rates
residual behavior
baseline vs trained model
```

---

## Main files

```txt
experiments/phase-3-linear-models/
├── README.md
├── phase3_linear_models_sanity.hpp
└── phase3_linear_models_sanity.cpp
```

Shared manual test helpers are located in:

```txt
experiments/common/
├── manual_test_utils.hpp
└── manual_test_utils.cpp
```

---

## Related source modules

Phase 3 uses the following linear model modules:

```txt
include/ml/linear_models/regularization.hpp
include/ml/linear_models/linear_regression.hpp
```

with implementation in:

```txt
src/linear_models/linear_regression.cpp
```

It also depends on Phase 1 and Phase 2 infrastructure:

```txt
include/ml/common/types.hpp
include/ml/common/math_ops.hpp
include/ml/common/shape_validation.hpp
include/ml/common/preprocessing_pipeline.hpp
include/ml/common/baselines.hpp
include/ml/common/evaluation_harness.hpp
include/ml/common/experiment_summary.hpp
```

---

## Running the checks

Build the project:

```bash
cmake --build build
```

Run the manual validation executable:

```bash
./build/ml_core_tests
```

The Phase 3 section should print checks for:

```txt
[Phase 3.1] Regularization config tests
[Phase 3.2] Linear regression implementation tests
```

---

## Current sanity checks

The first Phase 3 sanity checks validate that:

```txt
RegularizationConfig none is disabled
RegularizationConfig ridge is enabled
RegularizationConfig rejects negative Ridge lambda

LinearRegression rejects empty X/y
LinearRegression rejects mismatched X/y
LinearRegression rejects predict before fit
LinearRegression fits simple multivariate data
LinearRegression predict returns expected shape
LinearRegression score_mse is small after fit
LinearRegression stores loss history
LinearRegression loss decreases
LinearRegression rejects invalid learning rate
LinearRegression rejects Lasso fitting for now
LinearRegression supports Ridge lambda > 0
```

These checks confirm the basic model implementation before adding larger experiments.

---

## Experiment plan

The next experiments should use small controlled datasets to compare:

### 1. Unregularized vs Ridge

Purpose:

```txt
show how Ridge changes learned weights
show how Ridge can reduce weight magnitude
compare MSE and R² against baseline
```

Expected output:

```txt
CSV/TXT summaries for unregularized and Ridge models
notes on weight magnitude and metric differences
```

---

### 2. Scaled vs unscaled data

Purpose:

```txt
show why feature scaling matters for gradient descent
show why Ridge is sensitive to feature scale
compare convergence behavior
```

Expected output:

```txt
loss histories
evaluation summaries
short notes explaining convergence differences
```

---

### 3. Learning rate / convergence behavior

Purpose:

```txt
compare small, reasonable, and too-large learning rates
inspect loss history
identify convergence or instability
```

Expected output:

```txt
loss trend summaries
iterations_run
converged flag
notes on learning-rate sensitivity
```

---

### 4. Residual behavior

Purpose:

```txt
inspect prediction errors after training
connect residuals to model fit quality
identify systematic underprediction or overprediction
```

Expected output:

```txt
small residual table
summary statistics
interpretation notes
```

---

## Output files

Phase 3 experiment outputs should go under:

```txt
outputs/phase-3-linear-models/
```

Recommended output types:

```txt
small CSV summaries
small TXT summaries
small residual tables
loss history CSV files
experiment notes
```

Recommended rule:

```txt
commit:
    small representative summaries
    important experiment notes

do not commit:
    large generated files
    disposable temporary outputs
    large datasets
```

---

## Design rules

Phase 3 experiments must respect the project rules:

```txt
use multivariate data
use vectorized prediction
use the Phase 2 evaluation flow
compare against a baseline
avoid leakage
avoid scalar-only toy implementations
do not turn main.cpp into a scratchpad
```

Reusable code belongs in:

```txt
include/
src/
```

Experiment orchestration belongs in:

```txt
experiments/
```

Generated results belong in:

```txt
outputs/
```

---

## Phase exit meaning

Phase 3 is complete when ML Core has:

```txt
multivariate LinearRegression
vectorized prediction
vectorized MSE scoring
Ridge support
controlled experiments comparing Ridge/unregularized models
scaled vs unscaled comparison
learning-rate convergence comparison
residual interpretation notes
```

After Phase 3, the project should have a serious linear regression implementation that can be evaluated with the Phase 2 infrastructure.