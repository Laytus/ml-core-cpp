# ML Core

A serious C++ Machine Learning foundation project designed to build the full classical ML core needed **before Deep Learning**.

This project is now in its final wrap-up stage. The main implementation, experiment, practical workflow, and documentation layers are complete.

---

## Purpose

ML Core builds a strong Machine Learning foundation through a combination of:

- theory
- C++ implementation
- experiments
- structured documentation
- practical workflows with real datasets
- Python/Jupyter analysis of exported outputs

The goal is to study Machine Learning seriously enough that the transition to Deep Learning happens on top of a real foundation rather than on top of fragmented intuition.

This project is:

- conceptually solid
- implementation-oriented
- mathematically grounded
- experimentally validated
- connected to real practical workflows
- optimized for fast progress without oversimplifying the material

---

## Project Positioning

This repository should be treated as:

- a serious ML foundation project before Deep Learning
- a structured and optimized study roadmap
- a C++ implementation project supported by practical helper tools
- a bridge between introductory ML intuition and full DL preparation
- a completed foundation layer for the next Deep Learning project

It should **not** be treated as:

- a toy ML repo
- a rushed overview of random ML topics
- a full production ML framework
- a replacement for mature ML libraries
- an endless classical ML expansion project

---

## Current Status

The core project is complete.

Completed layers include:

- mathematical and statistical foundations
- data pipeline and evaluation methodology
- linear regression and regularized linear behavior
- binary and multiclass linear classification
- optimization methods for trainable models
- tree models and tree ensembles
- distance-based learning and linear SVM
- unsupervised learning with PCA and KMeans
- probabilistic ML with Gaussian Naive Bayes
- a minimal bridge to Deep Learning through Perceptron and Tiny MLP
- practical workflows with real datasets
- model usage documentation
- method-to-math mapping documentation
- final inventory and wrap-up documentation

The remaining work is final cleanup and project closure.

---

## Scope

ML Core covers the serious core of classical Machine Learning that should be understood before starting Deep Learning seriously.

This includes:

- mathematical and statistical foundations for ML
- data pipeline and evaluation methodology
- multivariate linear models
- logistic regression and softmax regression
- optimization for ML
- regularization and generalization
- trees and ensemble intuition
- distance-based learning and margin intuition
- unsupervised learning essentials
- probabilistic ML essentials
- bridge to neural networks and backpropagation
- practical real-dataset workflows

This project is not trying to cover every ML topic.

It is trying to cover the **real core** that matters before DL.

---

## Implemented Model Families

### Regression

- `LinearRegression`
- Ridge behavior through `LinearRegression` regularization
- `DecisionTreeRegressor`
- `GradientBoostingRegressor`

### Binary classification

- `LogisticRegression`
- `LinearSVM`
- `GaussianNaiveBayes`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `TinyMLPBinaryClassifier`
- `Perceptron` as an educational bridge model

### Multiclass classification

- `SoftmaxRegression`
- `KNNClassifier`
- `GaussianNaiveBayes`
- `DecisionTreeClassifier`
- `RandomForestClassifier`

### Unsupervised learning

- `PCA`
- `KMeans`

### Supporting utilities

- regression metrics
- binary classification metrics
- multiclass metrics
- data splitting utilities
- cross-validation utilities
- preprocessing utilities and fitted preprocessing pipelines
- distance metrics
- kernel similarity utilities
- bootstrap sampling utilities
- CSV dataset loading utilities
- structured practical output writers

---

## Practical Workflow Layer

The final project includes practical workflows with real datasets.

The practical workflow layer demonstrates:

- numeric CSV dataset loading
- conversion into `Matrix` / `Vector`
- preprocessing discipline
- model comparison
- hyperparameter sweeps
- structured CSV exports
- Python/Pandas verification
- Jupyter visualization
- practical interpretation

Main workflow families:

- regression
- binary classification
- multiclass classification
- unsupervised learning
- hyperparameter sweeps

Main output root:

```txt
outputs/practical-exercises/
```

Main notebook folder:

```txt
notebooks/practical-workflows/
```

Main practical docs:

```txt
docs/practical/
```

---

## Tooling

ML Core is implemented in **C++17** and uses:

- **CMake**
- **Eigen** for matrix operations
- CSV/data utilities
- structured output writers
- Python/Pandas verification scripts
- Jupyter notebooks for visualization
- optional external reference comparisons where useful

### Why Eigen is used

Eigen is used because:

- matrix operations become central very quickly in serious ML
- the project goal is ML, not writing a full matrix library
- using Eigen avoids wasting time on infrastructure that is not the main learning target

The core ML logic remains in C++.

Python and Jupyter are support tools for validating and visualizing exported results.

---

## Current Build Targets

The repository uses two executables:

### `ml_core_app`

A clean and minimal project entrypoint.

### `ml_core_tests`

A structured manual validation runner for:

- sanity checks
- per-phase demos
- practical workflow checks
- manual testing during development

This keeps the main app clean while still allowing fast iteration.

---

## Repository Structure

Main repository areas:

```txt
docs/
include/
src/
data/
experiments/
outputs/
notebooks/
scripts/
app/
```

### Responsibilities

- `docs/` → project docs, theory notes, action plans, practical docs, final inventories
- `include/` → reusable public headers
- `src/` → reusable implementations
- `data/` → input datasets
- `experiments/` → phase-specific experiment workflows
- `outputs/` → generated artifacts and output folder structure
- `notebooks/` → Python/Jupyter analysis of exported outputs
- `scripts/` → verification and summary helper scripts
- `app/` → executable entrypoints

For detailed structural rules, see:

- `docs/general/repo-structure.md`

---

## Build

### Configure and build

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

### Run

```bash
./build/ml_core_app
./build/ml_core_tests
```

For project-specific build conventions, see:

- `docs/general/build-notes.md`

---

## Python Environment for Practical Analysis

The C++ project can be built and tested without Python.

Python is used for practical output verification and notebooks.

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Useful scripts:

```bash
python3 scripts/verify_practical_outputs.py
python3 scripts/summarize_hyperparameter_sweeps.py
```

---

## Documentation Map

### General docs

- `docs/general/action-plan.md`
- `docs/general/ml-core.md`
- `docs/general/model-inventory.md`
- `docs/general/experiment-inventory.md`
- `docs/general/ml-core-wrap-up.md`
- `docs/general/dl-roadmap-entry.md`
- `docs/general/repo-structure.md`
- `docs/general/build-notes.md`

### Theory docs

Theory docs live under:

```txt
docs/theory/
```

They connect the C++ implementations to the mathematical foundations.

### Practical docs

Practical docs live under:

```txt
docs/practical/
```

Important practical docs include:

- `docs/practical/practical-workflows-summary.md`
- `docs/practical/models/`
- `docs/practical/math-maps/`
- `docs/practical/sweeps/`

---

## Working Method

For each implementation step, the project followed the same concise workflow:

1. add theory to the corresponding doc
2. write the step’s concise action plan
3. define header file(s)
4. define validations
5. define implementation action plan without code
6. define the test plan
7. implement the code
8. run sanity checks
9. document results

This workflow kept:

- theory aligned with implementation
- scope explicit
- code structure clean
- progress trackable
- final documentation reliable

---

## Verification Checklist

Useful final checks:

```bash
cmake --build build
./build/ml_core_tests
python3 scripts/verify_practical_outputs.py
python3 scripts/summarize_hyperparameter_sweeps.py
```

The Python scripts require practical CSV outputs to exist under:

```txt
outputs/practical-exercises/
```

Those CSV files are generated artifacts and may be ignored by Git.

---

## Final Outcome

By the end of ML Core, the repository provides:

- a serious understanding of classical ML foundations
- strong intuition for optimization and generalization
- practical experience with vectorized ML implementations
- a clear understanding of evaluation methodology
- reusable implementations for major classical ML model families
- a useful foundation in trees, ensembles, unsupervised learning, and probabilistic ML
- a direct conceptual bridge to neural networks and Deep Learning
- a real-dataset practical workflow layer
- structured usage and math-map docs for the main models
- final inventories and wrap-up documentation

Most importantly, it leaves the next Deep Learning project as a natural continuation rather than a leap into partially understood ideas.

---

## Next Step

The next step is not to keep expanding ML Core indefinitely.

The next step is to start the Deep Learning project defined in:

```txt
docs/general/dl-roadmap-entry.md
```

---

## Final Note

This project built the actual base layer that was missing before Deep Learning.

It is not a warm-up anymore.

It is the completed ML foundation project.
