# ML Core

**Classical machine learning implemented in C++17 with Eigen, from reusable model components to real-dataset experimentation workflows.**

ML Core is a C++ implementation project covering the main foundations of classical machine learning, including supervised and unsupervised models, optimization, preprocessing, evaluation, and practical model comparison.

The project is **complete within its defined scope**. Python and Jupyter are used only for validation and visualization; the core ML implementations remain in C++.

---

## Highlights

- Classical ML models implemented from the algorithmic level in **C++17**
- Matrix and vector operations powered by **Eigen**
- Reusable preprocessing, dataset splitting, cross-validation, and evaluation utilities
- Linear models, trees, ensembles, distance-based methods, probabilistic models, PCA, and KMeans
- Reusable optimization components covering batch GD, SGD, mini-batch GD, and momentum
- Practical workflows on real regression, binary classification, multiclass classification, and unsupervised datasets
- Structured CSV outputs with Python/Pandas verification and Jupyter visualization
- Dedicated theory, model-usage, and method-to-math documentation
- Minimal neural-network bridge with manual forward propagation and backpropagation

---

## Implemented models

| Family | Implementations |
| --- | --- |
| Regression | `LinearRegression`, Ridge regularization, `DecisionTreeRegressor`, `GradientBoostingRegressor` |
| Binary classification | `LogisticRegression`, `LinearSVM`, `GaussianNaiveBayes`, `DecisionTreeClassifier`, `RandomForestClassifier`, `TinyMLPBinaryClassifier`, `Perceptron` |
| Multiclass classification | `SoftmaxRegression`, `KNNClassifier`, `GaussianNaiveBayes`, `DecisionTreeClassifier`, `RandomForestClassifier` |
| Unsupervised learning | `PCA`, `KMeans` |

### Supporting components

The reusable support layer includes:

- train/test and train/validation/test splitting
- k-fold cross-validation
- preprocessing pipelines
- standardization and normalization utilities
- regression, binary classification, and multiclass metrics
- distance metrics
- linear, polynomial, and RBF kernel utilities
- bootstrap sampling
- CSV dataset loading
- structured experiment output utilities

---

## Practical workflows

The project includes end-to-end workflows for:

- regression
- binary classification
- multiclass classification
- unsupervised learning
- hyperparameter sweeps

Representative datasets used during the final practical validation include:

- `stock_ohlcv_engineered`
- `nasa_kc1_software_defects`
- `wine`

The workflow layer covers:

```text
dataset loading
      ↓
preprocessing
      ↓
train / validation / test workflow
      ↓
model fitting
      ↓
metrics and model comparison
      ↓
structured CSV outputs
      ↓
Python/Pandas verification
      ↓
Jupyter visualization and interpretation
```

Core model logic remains in C++; Python and Jupyter are used only as analysis and verification tools.

---

## Architecture

The repository separates reusable implementation code from experiments, documentation, and analysis tooling.

```text
include/ml/     Public C++ interfaces
src/            Reusable C++ implementations
experiments/    Model sanity checks and behavior studies
data/           Input datasets and metadata
outputs/        Generated experiment artifacts
docs/theory/    Mathematical and conceptual documentation
docs/practical/ Model usage, math maps, and workflow documentation
docs/general/   Project architecture, inventories, and wrap-up material
notebooks/      Jupyter analysis and visualization
scripts/        Python verification and summary helpers
app/            Executable entrypoints
```

The main implementation modules include:

```text
common/
linear_models/
optimization/
trees/
distance/
unsupervised/
probabilistic/
dl_bridge/
```

---

## Build and run

### Requirements

- C++17-compatible compiler
- CMake
- Eigen

### Configure and build

```bash
cmake -S . -B build
cmake --build build
```

### Run

```bash
./build/ml_core_app
./build/ml_core_tests
```

`ml_core_app` provides the minimal project entrypoint.

`ml_core_tests` currently acts as the structured validation runner used for sanity checks, phase-level validation, and practical workflow checks.

> A dedicated automated unit-test/CTest layer is planned as a repository-quality improvement.

---

## Python analysis environment

Python is not required for the core C++ implementation.

It is used for verification and visualization of exported practical results.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Useful verification scripts include:

```bash
python3 scripts/verify_practical_outputs.py
python3 scripts/summarize_hyperparameter_sweeps.py
```

Jupyter notebooks for practical analysis live under:

```text
notebooks/practical-workflows/
```

---

## Validation approach

ML Core was developed with several complementary validation layers:

- deterministic sanity checks for individual implementations
- model-behavior experiments
- metric and optimization comparisons
- real-dataset end-to-end workflows
- exported result verification with Python/Pandas
- Jupyter-based visualization and interpretation

The repository intentionally separates reusable model code from experiment and analysis code.

---

## Documentation

### General project documentation

- [`docs/general/ml-core.md`](docs/general/ml-core.md) — project identity and scope
- [`docs/general/action-plan.md`](docs/general/action-plan.md) — completed phase-by-phase execution plan
- [`docs/general/model-inventory.md`](docs/general/model-inventory.md) — final model inventory
- [`docs/general/experiment-inventory.md`](docs/general/experiment-inventory.md) — experiment and output inventory
- [`docs/general/ml-core-wrap-up.md`](docs/general/ml-core-wrap-up.md) — final project closure and scope summary

### Theory

Mathematical and conceptual notes live under:

```text
docs/theory/
```

They cover the theory behind the main implemented model families and optimization methods.

### Practical documentation

Practical documentation lives under:

```text
docs/practical/
```

It includes:

- model usage guides
- method-to-math mappings
- hyperparameter sweep summaries
- practical workflow interpretation

---

## Scope and limitations

ML Core is intentionally **not** a production ML framework or a replacement for libraries such as scikit-learn.

Some areas were deliberately scoped or deferred.

### Intentionally limited

- `LinearSVM` uses a primal linear formulation rather than a full kernel SVM solver
- kernel functions are available as utilities, but SMO/dual kernel-SVM training is not implemented
- `TinyMLPBinaryClassifier` is a minimal educational bridge, not a general neural-network framework
- missing-value handling in trees uses explicit rejection rather than learned routing
- advanced pruning and full out-of-bag scoring were deferred

### Out of scope

- production MLOps and model serving
- distributed training
- GPU acceleration
- automatic differentiation
- AutoML
- full deep-learning architectures
- large-scale benchmark infrastructure

These boundaries are intentional: the repository is focused on classical ML implementation, experimentation, and the conceptual bridge to deep learning.

---

## Project status

**Status: Complete**

ML Core is frozen as a classical machine-learning foundation project.

Future changes should be limited primarily to:

- bug fixes
- test improvements
- documentation corrections
- repository-quality improvements

Further model-family expansion belongs in separate projects.
