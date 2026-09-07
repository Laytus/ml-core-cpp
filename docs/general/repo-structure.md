# Repository Structure – ML Core

## Purpose

This document defines the repository organization rules for **ML Core**.

Its goal is to keep the project:

- consistent
- easy to navigate
- modular
- resistant to structural drift
- clear about the separation between reusable implementation code, experiments, generated outputs, documentation, datasets, and analysis tooling

This document should be treated as the structural reference for where code, docs, experiments, outputs, datasets, notebooks, scripts, and executable entrypoints belong.

---

## Top-Level Repository Structure

The final project is organized around the following main areas:

```text
app/
data/
docs/
experiments/
include/
notebooks/
outputs/
scripts/
src/
```

Supporting top-level files include project build, environment, and repository configuration such as:

```text
CMakeLists.txt
requirements.txt
README.md
.gitignore
```

The main structural principle is:

```text
include/ + src/   reusable C++ implementation
experiments/      sanity checks and behavior studies
data/             input datasets and metadata
outputs/          generated experiment artifacts
docs/             theory, practical documentation, and project-level documentation
notebooks/        analysis and visualization
scripts/          Python verification and summary helpers
app/              executable entrypoints
```

Each area has a distinct responsibility and should not be mixed with the others.

---

## `docs/`

The `docs/` folder stores all handwritten project documentation.

It is divided into three main areas:

```text
docs/general/
docs/theory/
docs/practical/
```

### `docs/general/`

This folder contains project-level documentation.

Examples include:

- project identity and scope
- action plans
- repository structure and build notes
- model and experiment inventories
- wrap-up summaries
- transition documentation

Representative files:

```text
docs/general/ml-core.md
docs/general/action-plan.md
docs/general/repo-structure.md
docs/general/build-notes.md
docs/general/model-inventory.md
docs/general/experiment-inventory.md
docs/general/ml-core-wrap-up.md
docs/general/dl-roadmap-entry.md
```

### `docs/theory/`

This folder contains the mathematical and conceptual notes behind the implemented ML topics.

Representative topics include:

- mathematical foundations
- statistical learning foundations
- evaluation methodology
- linear models
- logistic and softmax regression
- optimization
- trees and ensembles
- distance and kernel thinking
- unsupervised learning
- probabilistic ML
- the Deep Learning bridge

These documents explain the concepts behind the implementation rather than acting as API or workflow documentation.

### `docs/practical/`

This folder contains documentation for using and interpreting the implemented models and practical workflows.

Its main responsibilities include:

```text
docs/practical/models/
docs/practical/math-maps/
docs/practical/sweeps/
docs/practical/practical-workflows-summary.md
```

#### `docs/practical/models/`

Model usage guides.

These documents should explain:

- supported task type
- expected inputs
- preprocessing expectations
- model construction
- `fit`
- prediction-related methods
- outputs
- common mistakes
- related practical workflow artifacts

#### `docs/practical/math-maps/`

Method-to-math mappings for the implemented models.

These documents connect public and important internal methods to the mathematical process they implement.

#### `docs/practical/sweeps/`

Interpretation and summary material for selected hyperparameter sweeps.

### Documentation rule

Use:

```text
docs/general/   project identity, architecture, planning, inventories, wrap-up
docs/theory/    mathematical and conceptual foundations
docs/practical/ model usage, math mappings, workflow interpretation
```

Do not mix these roles.

Generated artifacts do **not** belong in `docs/`.

---

## `include/`

This folder contains the public headers for reusable project code.

Its structure mirrors the major implementation modules:

```text
include/ml/common/
include/ml/linear_models/
include/ml/optimization/
include/ml/trees/
include/ml/distance/
include/ml/unsupervised/
include/ml/probabilistic/
include/ml/dl_bridge/
```

### Purpose

Headers in `include/` should define:

- public structs
- classes
- function declarations
- reusable interfaces
- options/configuration types
- small inline or template utilities where appropriate

### Rule

Only reusable project code belongs here.

Do **not** put:

- experiment runners
- notebook logic
- one-off analysis code
- phase notes
- large validation-only code
- generated outputs

---

## `src/`

This folder contains the reusable C++ implementations corresponding to the public interfaces in `include/`.

Its organization should mirror the main modules in `include/ml/`.

Representative structure:

```text
src/common/
src/linear_models/
src/optimization/
src/trees/
src/distance/
src/unsupervised/
src/probabilistic/
src/dl_bridge/
```

### Purpose

`src/` should contain:

- model implementations
- reusable utilities
- reusable algorithms
- preprocessing and evaluation infrastructure
- optimization logic
- data-loading helpers that belong to the C++ core

### Rule

If code is intended to survive beyond a single experiment and is conceptually part of ML Core, it should normally live in `include/` + `src/`.

Do not let reusable logic accumulate in:

```text
app/
experiments/
scripts/
notebooks/
```

---

## `app/`

This folder contains executable entrypoints.

The project uses:

```text
app/main.cpp
app/test_runner.cpp
```

### `main.cpp`

This should remain minimal and stable.

Its role is:

- clean project entrypoint
- simple executable wiring
- intentionally small demonstration surface

It should **not** become:

- a scratchpad
- a phase archive
- a test suite
- a location for reusable model logic

### `test_runner.cpp`

This is the structured validation runner used during project development.

Its role is to support:

- sanity checks
- phase-level validation
- practical workflow checks
- manual validation during development

It is not intended to replace a dedicated automated unit-test suite.

### Rule

`app/` is for executable wiring and manual validation entrypoints.

Reusable logic belongs in `include/` + `src/`.

Larger experiment workflows belong in `experiments/`.

---

## `experiments/`

This folder contains experiment-specific workflows and validation code.

Examples include:

- phase-specific sanity workflows
- behavior studies
- optimizer comparisons
- model comparisons
- practical workflow runners
- experiment-specific export logic

### Purpose

Use `experiments/` when a workflow is too specific to belong to the reusable ML core but too substantial to live directly in an app entrypoint.

### Rule

Use the following boundary:

```text
reusable across models/phases   -> include/ + src/
small executable wiring         -> app/
experiment-specific workflow    -> experiments/
generated result                -> outputs/
```

Do not move experiment-specific behavior into the reusable core unless it becomes a genuine reusable abstraction.

---

## `outputs/`

This folder stores generated artifacts produced by experiments, evaluations, and practical workflows.

Examples include:

- CSV results
- JSON summaries
- text summaries
- metric tables
- prediction outputs
- probability outputs
- decision scores
- loss histories
- hyperparameter sweep results
- dimensionality-reduction projections
- clustering assignments
- generated plots or plot-ready data

### Organization

Outputs should be grouped by phase or workflow family rather than mixed in one flat directory.

Representative structures include:

```text
outputs/phase-3-linear-models/
outputs/phase-5-optimization/
outputs/phase-8-unsupervised/
outputs/practical-exercises/
```

Within a workflow folder, use clear artifact names and subfolders where useful.

For example:

```text
outputs/<workflow>/
  csv/
  json/
  notes/
  plots/
```

or descriptive flat files when the workflow is already narrow:

```text
metrics_summary.csv
predictions.csv
probabilities.csv
sweep_results.csv
experiment_summary.txt
```

### Output storage policy

Commit only artifacts that provide useful evidence or documentation value.

Good candidates for version control include:

- small representative CSV files
- compact JSON summaries
- concise text summaries
- strategically useful plots
- artifacts referenced directly by documentation

Do not commit:

- huge generated files
- disposable temporary outputs
- large intermediate artifacts
- regenerated noise with no documentation value

### Rule

Generated outputs belong in `outputs/`, not in `docs/` or `data/`.

---

## `data/`

This folder stores input datasets and dataset metadata.

The practical workflow layer uses the following convention:

```text
data/raw/
data/processed/
data/metadata/
```

### `data/raw/`

Stores source datasets in the form in which they are brought into the project, subject to repository-size and licensing constraints.

### `data/processed/`

Stores cleaned or transformed datasets when keeping those files in the repository is useful and appropriate.

### `data/metadata/`

Stores dataset-level metadata such as:

- dataset name
- source
- license or usage note when known
- target column
- feature columns
- task type
- preprocessing notes

### Rule

`data/` is for **inputs**.

Generated predictions, metrics, transformed experiment outputs, and analysis artifacts belong in `outputs/`.

If a dataset is too large or inappropriate to commit, keep only the strategically necessary subset or metadata and document how to obtain it.

---

## `notebooks/`

This folder contains Jupyter notebooks used for analysis and visualization.

The practical analysis layer lives primarily under:

```text
notebooks/practical-workflows/
```

### Purpose

Notebooks may be used for:

- plotting model outputs
- comparing metrics
- visualizing predictions
- inspecting probability distributions
- visualizing hyperparameter sweeps
- PCA projections
- KMeans cluster visualizations
- interpreting exported C++ workflow results

### Rule

Notebooks are analysis tools only.

They must not become an alternative implementation path for the core ML models.

The architectural boundary is:

```text
C++       model logic and reusable ML implementation
Python    verification and summary helpers
Jupyter   visualization and interpretation
```

---

## `scripts/`

This folder contains Python helper scripts used to verify and summarize outputs generated by the C++ workflows.

Representative scripts include:

```text
scripts/verify_practical_outputs.py
scripts/summarize_hyperparameter_sweeps.py
```

### Purpose

Scripts may be used for:

- checking generated CSV outputs
- validating output schemas
- summarizing experiment results
- preparing concise analysis summaries
- supporting notebook workflows

### Rule

Scripts should support the C++ project rather than duplicate its model logic.

The core ML implementation remains in C++.

---

## Module Ownership Rules

Each reusable implementation should belong to one clear module.

### `common/`

Use `common/` only for utilities that are truly shared across multiple model families or phases.

Examples include:

- dataset abstractions
- data splitting
- cross-validation
- preprocessing pipelines
- metrics
- evaluation support
- CSV dataset loading
- shared validation helpers
- experiment export utilities where genuinely reusable

Do **not** use `common/` as a miscellaneous dumping ground.

### Model-specific modules

If code belongs clearly to one model family or concept, keep it there.

Examples:

```text
linear regression, logistic regression, softmax, LinearSVM
    -> linear_models/

Decision Trees, Random Forest, Gradient Boosting, bootstrap helpers
    -> trees/

distance metrics, k-NN, kernel utilities
    -> distance/

PCA, KMeans
    -> unsupervised/

Gaussian Naive Bayes
    -> probabilistic/

Perceptron, Tiny MLP, activations
    -> dl_bridge/

gradient descent, SGD, mini-batch GD, momentum
    -> optimization/
```

This keeps ownership explicit and prevents structural drift.

---

## Naming Conventions

### Files

Prefer lowercase snake_case filenames.

Examples:

```text
linear_regression.hpp
training_history.cpp
evaluation_harness.hpp
random_forest.cpp
gaussian_naive_bayes_usage.md
```

### Classes and structs

Use PascalCase.

Examples:

```cpp
LinearRegression
RandomForestClassifier
TrainingHistory
TinyMLPBinaryClassifier
```

### Functions

Use snake_case.

Examples:

```cpp
train_test_split
compute_accuracy
predict_proba
```

### Namespaces

Keep the project namespace consistent.

Representative patterns:

```cpp
namespace ml {
    ...
}
```

and, where appropriate:

```cpp
namespace ml::common {
    ...
}
```

---

## Reusable Code vs Experiment Code

This is one of the most important structural boundaries in the repository.

### Reusable code

Put code in `include/` + `src/` when it is:

- conceptually part of the ML library
- expected to be reused
- part of a model or shared utility
- independent of one narrow experiment

### Experiment code

Put code in `experiments/` when it is:

- used for model comparison
- tied to one behavior study
- specific to one phase
- used for hyperparameter sweeps
- used to export experiment-specific artifacts

### Analysis code

Use:

```text
scripts/     automated verification and summaries
notebooks/   visualization and interactive interpretation
```

### Generated artifacts

Use:

```text
outputs/
```

### Final rule

```text
Reusable C++ logic        -> include/ + src/
Executable wiring         -> app/
Experiment workflows      -> experiments/
Generated artifacts       -> outputs/
Input data                -> data/
Theory/project docs       -> docs/
Python verification       -> scripts/
Visualization/analysis    -> notebooks/
```

---

## Documentation Rules

Every serious project component should leave documentation at the appropriate level.

Use:

- `docs/theory/` for mathematical and conceptual material
- `docs/practical/models/` for model usage
- `docs/practical/math-maps/` for method-to-math mappings
- `docs/practical/sweeps/` for selected hyperparameter interpretation
- `docs/general/` for architecture, inventories, scope, and project-level material

Do not place generated experiment outputs inside documentation directories.

Do not rely on chat history or temporary notes as the only explanation of important project behavior.

---

## Dataset and Output Separation

The project must preserve a strict distinction between inputs and generated results.

```text
data/      input datasets and metadata
outputs/   experiment-generated artifacts
```

Examples:

```text
data/raw/wine.csv
data/metadata/wine.md

outputs/practical-exercises/multiclass_metrics.csv
outputs/phase-8-unsupervised/pca_projection.csv
```

Do not store generated model outputs under `data/`.

---

## App / Validation / Experiment Separation

The intended boundary is:

### `app/main.cpp`

Minimal and stable application entrypoint.

### `app/test_runner.cpp`

Structured manual validation runner.

### `experiments/`

Larger phase-specific sanity checks, behavior studies, and practical workflows.

### `include/` + `src/`

Reusable implementation code.

This boundary should remain stable even as repository-quality improvements add a dedicated automated testing layer.

---

## Anti-Drift Rules

To keep the repository coherent:

- do not use `main.cpp` as a scratchpad
- do not let `test_runner.cpp` become a permanent archive of unrelated historical checks
- do not duplicate reusable logic inside experiment runners
- do not place model implementations in notebooks or Python scripts
- do not place theory in generated-output folders
- do not place generated results in `data/`
- do not use `common/` as a miscellaneous dumping ground
- do not keep large generated artifacts without a clear reason
- do not create new top-level folders when an existing responsibility already fits the content

---

## Repository-Quality Improvements

The final project structure is stable.

Future repository-quality work may add infrastructure such as:

```text
.github/workflows/
tests/
LICENSE
.gitattributes
```

These additions should improve validation, CI, repository metadata, and presentation without changing the core ownership rules defined above.

In particular, a future dedicated automated test layer should remain separate from:

```text
app/test_runner.cpp
experiments/
```

so that:

```text
tests/        automated unit/integration tests
app/          executable wiring
experiments/  behavior studies and practical workflows
```

remain distinct responsibilities.

---

## Final Principle

The repository should remain organized so that:

```text
docs/         explain the project, theory, and practical usage
include/      expose reusable C++ interfaces
src/          implement the reusable ML core
app/          provide executable entrypoints and manual validation wiring
experiments/  contain phase-specific validation and behavior studies
data/         store input datasets and metadata
outputs/      store generated artifacts
scripts/      verify and summarize exported results
notebooks/    analyze and visualize outputs
```

The structure should make the codebase easier to understand, validate, and extend without turning ML Core into a broader framework than it was designed to be.
