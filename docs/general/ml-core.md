# ML Core

## Purpose

ML Core is a serious Machine Learning study project designed to build the full practical and theoretical foundation needed **before moving into Deep Learning**.

This project is not a toy implementation pass.

It is intended to cover the core of classical Machine Learning in a way that is:

- conceptually solid
- implementation-oriented
- mathematically grounded
- experimentally validated
- connected to real practical workflows
- optimized for fast progress without oversimplifying the material

The goal is to study Machine Learning seriously enough that the transition to Deep Learning happens on top of a real foundation rather than on top of fragmented intuition.

---

## Current Status

This project is now in its final wrap-up stage.

The main ML Core implementation and practical validation phases are complete, including:

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

The remaining work is documentation cleanup, final README alignment, and final project closure.

---

## Main Objective

Build a strong ML core through a combination of:

- theory
- focused implementations
- experiments
- structured documentation
- progressive model comparison
- practical real-dataset workflows

The project should produce:

1. a serious understanding of the main ML ideas
2. a clean C++ codebase for the most important implementations
3. a clear bridge from classical ML to Deep Learning
4. a structured study reference that can be reused later
5. a practical experimentation layer that proves the models can run end-to-end on real datasets

---

## What “ML Core” means here

In this project, “ML Core” means the set of concepts that should be understood well before starting Deep Learning seriously.

This includes:

- mathematical and statistical foundations for ML
- proper evaluation methodology
- linear models in multivariate form
- optimization for trainable models
- regularization and generalization
- binary and multiclass classification
- trees and ensemble intuition
- distance-based learning and margin intuition
- unsupervised learning essentials
- probabilistic ML essentials
- a direct bridge to neural networks and backpropagation
- practical workflows for using models on real datasets

This project is not meant to cover literally every topic in Machine Learning.

It is meant to cover the **serious core** that gives enough depth to move into DL correctly.

---

## Project Philosophy

### 1. Serious, but optimized

This project should move as fast as possible, but without artificially simplifying the concepts.

Speed matters.

But correctness of scope matters more.

So the project should be optimized through:

- good sequencing
- selective implementation depth
- strong documentation
- practical validation
- helper tools where appropriate

and **not** through reducing the concepts to toy-only versions.

### 2. Theory and implementation must stay connected

Every major concept should be understood:

- mathematically
- conceptually
- programmatically
- experimentally

This means the project should avoid two extremes:

- theory with no code intuition
- code with no theoretical grounding

### 3. C++ remains the implementation language

The project is a C++ project.

The core ML models, metrics, preprocessing utilities, and workflow runners are implemented in C++.

Python and Jupyter are used only as support tools for:

- validating exported CSV outputs
- summarizing experiment results
- visualizing practical workflow outputs
- inspecting model behavior

The core ML logic remains in C++.

### 4. Build ML knowledge, not unnecessary low-level tooling

The goal is Machine Learning.

So supporting tools are allowed when they help keep the project focused on ML rather than on unrelated engineering overhead.

This is why the project uses:

- Eigen for matrix operations
- CSV / data utilities
- structured output writers
- Python/Pandas verification scripts
- Jupyter notebooks for visualization
- optional external reference comparisons when useful

### 5. Explicit implementation depth

Not every topic should be implemented at the same depth.

To keep the project serious and fast, each phase or sub-phase is classified as one of the following:

- **Level A — full implementation**
- **Level B — partial implementation + experiments**
- **Level C — theory + small demo only**

This is a core design rule of the project.

It allows:

- deeper implementation where it matters most
- theory coverage where implementation would add too much scope
- faster progress without fake completeness

---

## Tooling Policy

### Matrix engine

This project uses **Eigen** as the matrix library.

Reason:

- matrix and vector operations are central to ML
- writing a full personal matrix library would add too much scope
- the project goal is ML Core, not numerical library engineering

A small personal matrix module may be explored later as a side exercise if useful, but it is not part of the main ML Core path.

### Other allowed helpers

The project also uses or may use:

- CSV/data utilities
- structured output helpers
- Python/Pandas verification scripts
- Jupyter notebooks
- plotting/export helpers
- optional external reference comparisons

These tools are allowed because they support the study of ML instead of replacing it.

---

## What this project fixed

This project exists because a first implementation-based ML repo was useful, but too limited.

That earlier work was good for:

- first intuition
- C++ practice
- simple model mechanics

But it was not enough for:

- serious ML readiness
- strong theoretical grounding
- vectorized model implementation
- evaluation discipline
- practical real-dataset workflows
- a confident transition into DL

ML Core fixed this by becoming:

- broader
- deeper
- more structured
- more practical
- more honest about what “real preparation” requires

---

## Scope Covered

The project covers:

- math and statistical foundations for ML
- data pipeline and evaluation methodology
- multivariate linear models
- multivariate logistic regression
- softmax regression
- optimization for ML
- trees and ensemble foundations
- advanced tree ensemble behavior
- distance and kernel intuition
- linear SVM
- unsupervised learning essentials
- probabilistic ML essentials
- bridge to Deep Learning
- real-dataset practical workflows
- model usage documentation
- math-map documentation
- final model and experiment inventories

This is the intended serious core.

Topics outside this scope may be mentioned, but they are not the main target unless later added explicitly.

---

## Implemented Model Families

The final implemented model inventory includes:

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

The main workflow families are:

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

## Final Documentation Set

The project now includes final documentation layers:

### General docs

- `docs/general/action-plan.md`
- `docs/general/ml-core.md`
- `docs/general/model-inventory.md`
- `docs/general/experiment-inventory.md`
- `docs/general/ml-core-wrap-up.md`
- `docs/general/dl-roadmap-entry.md`

### Practical docs

- `docs/practical/practical-workflows-summary.md`
- `docs/practical/models/`
- `docs/practical/math-maps/`
- `docs/practical/sweeps/`

### Theory docs

Theory docs live under:

```txt
docs/theory/
```

They connect the C++ implementations to the mathematical foundations.

---

## Non-Goals

To keep the project focused, these are not primary goals:

- covering every ML subfield exhaustively
- building production MLOps systems
- distributed training systems
- GPU systems programming
- full numerical computing infrastructure from scratch
- full research-level treatment of every model family
- full AutoML
- full kernel SVM / SMO implementation
- full deep-learning framework
- production-grade model serving

Also, this project should not become:

- a general-purpose ML framework
- a replacement for mature ML libraries
- a pure math notebook without implementation depth
- an endless classical ML expansion project

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

This workflow was mandatory because it kept:

- theory aligned with implementation
- scope explicit
- code structure clean
- progress trackable
- final documentation reliable

---

## Final Outcome

By the end of ML Core, the project provides:

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

---

## Final Positioning

ML Core should be understood as:

- a serious ML foundation project before DL
- a structured and optimized ML study roadmap
- a C++ implementation project supported by practical helper tools
- a bridge between introductory ML intuition and full DL preparation
- a completed foundation layer for the next Deep Learning project

The next step is not to keep expanding ML Core indefinitely.

The next step is to start the Deep Learning project defined in:

```txt
docs/general/dl-roadmap-entry.md
```