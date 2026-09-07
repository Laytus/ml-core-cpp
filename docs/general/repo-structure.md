# Repository Structure – ML Core

## Purpose

This document defines the repository organization rules for **ML Core**.

Its goal is to keep the project:

- consistent
- easy to navigate
- modular
- resistant to structural drift
- explicit about the separation between reusable implementation code, automated tests, experiments, generated outputs, documentation, datasets, and analysis tooling

This document is the structural reference for where code, tests, docs, experiments, outputs, datasets, notebooks, scripts, CI configuration, and executable entrypoints belong.

---

## Top-Level Repository Structure

The final project is organized around the following main areas:

```text
.github/
app/
data/
docs/
experiments/
include/
notebooks/
outputs/
scripts/
src/
tests/
```

Supporting top-level files include build, environment, and repository configuration such as:

```text
CMakeLists.txt
requirements.txt
README.md
.gitignore
```

The main structural principle is:

```text
include/ + src/   reusable C++ implementation
tests/            automated deterministic tests
experiments/      sanity checks and behavior studies
data/             input datasets and metadata
outputs/          generated experiment artifacts
docs/             theory, practical documentation, and project-level documentation
notebooks/        analysis and visualization
scripts/          Python verification and summary helpers
app/              executable entrypoints and manual validation wiring
.github/workflows continuous integration
```

Each area has a distinct responsibility and should not be mixed with the others.

---

## `.github/`

The `.github/` folder stores GitHub-specific repository automation and metadata.

The current CI workflow lives under:

```text
.github/workflows/ci.yml
```

### Purpose

GitHub Actions is used to automatically validate the repository on pushes and pull requests.

The CI workflow performs:

```text
checkout
   ↓
install build dependencies
   ↓
CMake configure
   ↓
build
   ↓
CTest
```

### Rule

CI configuration belongs under `.github/workflows/`.

Do not place:

- application logic
- model code
- experiment code
- generated outputs

inside `.github/`.

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

These documents explain:

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
include/ml/workflows/
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

- automated tests
- experiment runners
- notebook logic
- one-off analysis code
- phase notes
- generated outputs

---

## `src/`

This folder contains the reusable C++ implementations corresponding to the public interfaces in `include/`.

Its organization mirrors the main modules in `include/ml/`.

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
src/workflows/
```

### Purpose

`src/` should contain:

- model implementations
- reusable utilities
- reusable algorithms
- preprocessing and evaluation infrastructure
- optimization logic
- data-loading helpers that belong to the C++ core
- reusable practical workflow orchestration

### Rule

If code is intended to survive beyond a single experiment and is conceptually part of ML Core, it should normally live in `include/` + `src/`.

Do not let reusable logic accumulate in:

```text
app/
experiments/
tests/
scripts/
notebooks/
```

---

## `tests/`

This folder contains the automated test suite.

The project uses:

```text
Catch2
CTest
```

The test suite is intended to remain:

- deterministic
- fast
- self-contained
- independent from large external datasets
- suitable for local execution and CI

Representative structure:

```text
tests/
  common/
  linear_models/
  trees/
  distance/
  unsupervised/
  probabilistic/
  dl_bridge/
  CMakeLists.txt
```

### Purpose

Automated tests should verify:

- known mathematical behavior on small synthetic datasets
- input and option validation
- API misuse
- model invariants
- fitted/unfitted behavior
- deterministic behavior where applicable

### Rule

Automated tests should not become behavior-study scripts or large practical workflows.

Use:

```text
tests/        correctness and regression checks
experiments/  behavior studies and practical workflows
```

The automated suite should be runnable through:

```bash
ctest --test-dir build --output-on-failure
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

This is the structured manual validation runner used during project development.

Its compiled executable is:

```text
ml_core_validation
```

Its role is to support:

- sanity checks
- phase-level validation
- behavior studies
- practical workflow checks
- manual validation during development

It is intentionally separate from the automated Catch2/CTest suite.

### Rule

`app/` is for executable wiring and manual validation entrypoints.

Reusable logic belongs in `include/` + `src/`.

Automated tests belong in `tests/`.

Larger experiment workflows belong in `experiments/`.

---

## `experiments/`

This folder contains experiment-specific workflows and validation code.

Examples include:

- phase-specific sanity workflows
- behavior studies
- optimizer comparisons
- model comparisons
- hyperparameter sweeps
- practical workflow runners
- experiment-specific export logic

### Purpose

Use `experiments/` when a workflow is too specific to belong to the reusable ML core but too substantial to live directly in an app entrypoint.

### Rule

Use the following boundary:

```text
reusable across models/phases   -> include/ + src/
automated correctness test      -> tests/
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

### `linear_models/`

Use for reusable linear and margin-based supervised models such as:

```text
LinearRegression
LogisticRegression
SoftmaxRegression
LinearSVM
```

### `optimization/`

Use for reusable optimization infrastructure such as:

```text
batch gradient descent
SGD
mini-batch gradient descent
momentum
training history
```

### `trees/`

Use for tree and ensemble components such as:

```text
DecisionTreeClassifier
DecisionTreeRegressor
RandomForestClassifier
GradientBoostingRegressor
bootstrap helpers
split-scoring infrastructure
```

### `distance/`

Use for:

```text
distance metrics
KNNClassifier
kernel similarity utilities
```

### `unsupervised/`

Use for:

```text
PCA
KMeans
```

### `probabilistic/`

Use for:

```text
GaussianNaiveBayes
```

### `dl_bridge/`

Use for the deliberately scoped Deep Learning bridge:

```text
Perceptron
TinyMLPBinaryClassifier
activation helpers
```

### `workflows/`

Use for reusable orchestration supporting practical end-to-end model workflows.

Representative responsibilities include:

- regression comparison workflows
- binary classification comparison workflows
- multiclass classification comparison workflows
- unsupervised workflows
- hyperparameter sweeps
- structured output writing

### Rule

The `workflows/` module should contain reusable workflow orchestration, not one-off experiment logic.

If a workflow is narrow, phase-specific, or exploratory, keep it in `experiments/`.

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

## Reusable Code vs Tests vs Experiment Code

This is one of the most important structural boundaries in the repository.

### Reusable code

Put code in `include/` + `src/` when it is:

- conceptually part of the ML core
- expected to be reused
- part of a model or shared utility
- independent of one narrow experiment

### Automated test code

Put code in `tests/` when it verifies:

- deterministic correctness
- API contracts
- input validation
- invariants
- regression behavior

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
Automated correctness     -> tests/
Executable wiring         -> app/
Experiment workflows      -> experiments/
Generated artifacts       -> outputs/
Input data                -> data/
Theory/project docs       -> docs/
Python verification       -> scripts/
Visualization/analysis    -> notebooks/
CI configuration          -> .github/workflows/
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

## Automated Tests vs Manual Validation vs Experiments

The project now has three distinct validation layers.

### `tests/`

Automated Catch2 tests discovered and executed by CTest.

Use for:

- deterministic correctness checks
- validation behavior
- known synthetic-data results
- regression protection

### `app/test_runner.cpp` → `ml_core_validation`

Manual validation runner.

Use for:

- selected sanity workflows
- phase-level manual validation
- convenient development-time checks

### `experiments/`

Broader behavior studies and practical workflows.

Use for:

- model comparisons
- optimizer studies
- hyperparameter sweeps
- real-dataset workflows
- generated output artifacts

### Rule

These layers should remain separate.

A large practical workflow should not be moved into CTest merely to increase test count.

An automated correctness check should not depend on generated experiment outputs or large datasets.

---

## Build and Test Integration

The project uses CMake as the build system.

With automated tests enabled:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The automated tests are integrated through:

```text
Catch2
CTest
tests/CMakeLists.txt
```

The manual validation runner remains separately executable as:

```bash
./build/ml_core_validation
```

---

## Continuous Integration

The repository uses GitHub Actions to build and test the project automatically.

Workflow:

```text
.github/workflows/ci.yml
```

The current CI validates the project on Ubuntu using:

```text
CMake
Ninja
Eigen
Catch2
CTest
```

### Rule

CI should run the fast automated test suite.

Manual validation workflows and heavier experiment workflows should remain outside the default CI path unless there is a clear future reason to include them.

---

## Anti-Drift Rules

To keep the repository coherent:

- do not use `main.cpp` as a scratchpad
- do not let `test_runner.cpp` become a permanent archive of unrelated historical checks
- do not duplicate reusable logic inside tests or experiment runners
- do not turn `tests/` into a behavior-study framework
- do not place model implementations in notebooks or Python scripts
- do not place theory in generated-output folders
- do not place generated results in `data/`
- do not use `common/` as a miscellaneous dumping ground
- do not keep large generated artifacts without a clear reason
- do not create new top-level folders when an existing responsibility already fits the content
- do not add heavy experiment workflows to CI simply to increase apparent coverage

---

## Repository-Quality Infrastructure

The repository-quality layer now includes:

```text
tests/
.github/workflows/ci.yml
```

Additional repository metadata and presentation files may include:

```text
LICENSE
.gitattributes
```

These files should improve repository clarity, portability, language classification, and reuse without changing the core ownership rules defined above.

---

## Final Principle

The repository should remain organized so that:

```text
.github/     contains CI and GitHub-specific automation
docs/        explain the project, theory, and practical usage
include/     expose reusable C++ interfaces
src/         implement the reusable ML core
tests/       provide automated deterministic correctness checks
app/         provide executable entrypoints and manual validation wiring
experiments/ contain phase-specific validation and behavior studies
data/        store input datasets and metadata
outputs/     store generated artifacts
scripts/     verify and summarize exported results
notebooks/   analyze and visualize outputs
```

The structure should make the codebase easier to understand, validate, and extend without turning ML Core into a broader framework than it was designed to be.
