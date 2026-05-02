# Phase 2 — Evaluation Methodology Experiments

This folder contains manual sanity checks and small validation experiments for Phase 2 of ML Core.

Phase 2 establishes the reusable data and evaluation infrastructure needed before implementing serious ML models.

The goal is not to train models yet. The goal is to make sure future models can plug into a consistent workflow for:

```txt
dataset representation
data splitting
cross-validation
leakage-safe preprocessing
baseline evaluation
metric computation
evaluation reports
experiment summary export
```

---

## Scope

This experiment group validates:

```txt
SupervisedDataset
train_test_split
train_validation_test_split
k_fold_split
StandardScaler
MinMaxScaler
MeanRegressor
regression metrics
evaluation harness
experiment summary export
```

These checks are intentionally small and deterministic.

They are designed to answer:

```txt
Can a later model use this infrastructure without redesigning evaluation?
```

---

## Main files

```txt
experiments/phase-2-evaluation-methodology/
├── README.md
├── phase2_evaluation_sanity.hpp
└── phase2_evaluation_sanity.cpp
```

Shared manual test helpers are located in:

```txt
experiments/common/
├── manual_test_utils.hpp
└── manual_test_utils.cpp
```

---

## Related source modules

Phase 2 uses the following reusable modules:

```txt
include/ml/common/dataset.hpp
include/ml/common/data_split.hpp
include/ml/common/cross_validation.hpp
include/ml/common/preprocessing_pipeline.hpp
include/ml/common/regression_metrics.hpp
include/ml/common/baselines.hpp
include/ml/common/evaluation.hpp
include/ml/common/evaluation_harness.hpp
include/ml/common/experiment_summary.hpp
```

with implementations in:

```txt
src/common/
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

The Phase 2 section should print checks for:

```txt
[Phase 2.1] Supervised dataset abstraction tests
[Phase 2.2] Train/test split tests
[Phase 2.2] Train/validation/test split tests
[Phase 2.3] K-fold cross-validation split tests
[Phase 2.4] Preprocessing pipeline tests
[Phase 2.5] Baseline evaluation flow tests
[Phase 2.6] Reusable evaluation harness tests
[Phase 2.7] Metrics and experiment summary export tests
```

---

## Output files

Some tests write small output files under:

```txt
outputs/phase-2-evaluation-methodology/
```

Example generated files:

```txt
test_regression_summary.csv
test_regression_summary_content.csv
test_regression_summary.txt
test_regression_summary_content.txt
```

These are generated test artifacts.

They are useful for validating export behavior, but they are not necessarily meant to be committed.

Recommended rule:

```txt
commit:
    source code
    documentation
    intentional small representative summaries if useful

do not commit:
    disposable generated test outputs
    large generated files
    temporary experiment artifacts
```

---

## Evaluation discipline validated by this phase

The Phase 2 experiments validate the following workflow:

```txt
1. Represent supervised data as X/y.
2. Split data before fitting preprocessing.
3. Fit preprocessing on training data only.
4. Transform validation/test using training statistics.
5. Compare trained predictions against a baseline.
6. Use consistent regression metrics.
7. Export small experiment summaries.
```

This discipline is required before moving into model implementation phases.

---

## Phase exit meaning

When these checks pass, ML Core has the infrastructure needed for Phase 3:

```txt
linear regression can use the same dataset abstraction
linear regression can use the same split utilities
linear regression can use leakage-safe preprocessing
linear regression can compare against MeanRegressor
linear regression can export evaluation summaries
```

Phase 2 is complete when the source code, theory notes, and manual tests all reflect this workflow.