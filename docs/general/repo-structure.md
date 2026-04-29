# Repository Structure – ML Core

## Purpose

This document defines the repository organization rules for **ML Core**.

Its goal is to keep the project:
- consistent
- scalable
- easy to navigate
- resistant to drift as the number of phases grows

This document should be treated as the structural reference for where code, docs, experiments, and outputs belong.

---

## Top-Level Repository Structure

The project is organized into the following main areas:

```text
docs/
include/
src/
data/
experiments/
outputs/
app/
```

Each area has a different responsibility and should not be mixed with the others.

---

## Folder Responsibilities

## `docs/`

The `docs/` folder stores all project writing.

It is divided into:

```text
docs/general/
docs/theory/
```

### `docs/general/`
This folder contains:
- project identity docs
- action plans
- repo structure and workflow docs
- wrap-up summaries
- transition docs

Examples:
- `ml-core.md`
- `action-plan.md`
- `repo-structure.md`
- `build-notes.md`

### `docs/theory/`
This folder contains theory notes for ML topics.

These notes are the conceptual reference for implementation phases.

Examples:
- `math-foundations.md`
- `optimization.md`
- `linear-models.md`
- `unsupervised-learning.md`

### Rule
- `docs/general/` = project management and project identity
- `docs/theory/` = ML concepts and technical theory

Do not mix these two roles.

---

## `include/`

This folder contains **public headers** for reusable project code.

Its structure mirrors the code organization of the project:

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
- small inline/template utilities where appropriate

### Rule
Only reusable project code belongs here.

Do **not** put:
- experiments
- temporary scratch code
- large test-only logic
- phase notes
- one-off debug utilities unless they clearly belong to reusable infrastructure

---

## `src/`

This folder contains **implementations** of the reusable code declared in `include/`.

Its structure mirrors `include/`.

Examples:
- `src/common/`
- `src/linear_models/`
- `src/optimization/`
- `src/trees/`

### Purpose
`src/` should contain:
- implementation of model behavior
- reusable utilities
- reusable algorithms
- reusable helpers tied to the project library

### Rule
If code is meant to survive beyond a single test or experiment, it should usually live in `src/` and have a matching header in `include/`.

Do not let reusable logic accumulate in `app/` or `experiments/`.

---

## `app/`

This folder contains executable entrypoints.

For now, the project uses:

```text
app/main.cpp
app/test_runner.cpp
```

### `main.cpp`
This should remain minimal.

Its role is:
- clean app entrypoint
- stable demonstration entrypoint
- intentionally small executable

It should **not** become a scratchpad or test archive.

### `test_runner.cpp`
This is the structured manual validation runner.

Its role is:
- run sanity checks
- run per-phase demos
- run manual validation code
- support early development without polluting reusable code

### Rule
`app/` is for executable wiring and manual validation entrypoints.

It is **not** the place for:
- reusable model implementations
- theory
- giant historical commented blocks
- duplicated logic that really belongs in `src/`

---

## `experiments/`

This folder contains experiment-specific workflows and experiment support artifacts.

Examples:
- experiment scripts
- experiment-specific code
- small comparison runners
- experiment metadata
- saved experiment descriptions

### Purpose
Use `experiments/` when an experiment becomes too large or too specific to belong in `app/test_runner.cpp`.

### Rule
If something is:
- reusable across phases → put it in `include/` + `src/`
- small manual validation → `app/test_runner.cpp`
- larger experiment workflow → `experiments/`

This is an important separation rule.

---

## `outputs/`

This folder stores generated artifacts.

### Purpose

`outputs/` exists to store generated artifacts produced by experiments, evaluations, and reusable phase-level analyses.

This folder should contain machine-generated or experiment-generated results, not handwritten documentation.

Examples:
- CSV results
- JSON summaries
- text summaries
- notes derived from experiments
- plots or plot-ready outputs

### Recommended structure
Use phase-specific folders such as:

```text
outputs/phase-1-math-sanity/
outputs/phase-3-linear-models/
outputs/phase-5-optimization/
```

Inside a phase folder, a useful convention is:

```text
csv/
json/
notes/
plots/
```

### Output storage policy

Outputs should be stored by phase first, and then by artifact type.

Recommended structure:

```text
outputs/
  phase-1-math-sanity/
    csv/
    json/
    notes/
    plots/
  phase-3-linear-models/
    csv/
    json/
    notes/
    plots/
```

Use this pattern so that:
- outputs remain easy to locate by phase
- generated artifacts from different phases do not get mixed together
- experiment summaries stay readable even when the number of saved files grows

### What belongs in each output subfolder

#### `csv/`
Use for tabular experiment results, metric tables, and exportable numeric summaries.

Examples:
- optimizer comparisons
- per-epoch metrics
- regularization comparisons

#### `json/`
Use for structured experiment summaries and machine-readable metadata.

Examples:
- configuration summaries
- metric snapshots
- reproducible experiment summaries

#### `notes/`
Use for short generated or semi-structured experiment conclusions that are still derived from a run.

Examples:
- experiment summaries
- interpretation notes tied to outputs
- concise result readouts saved after a run

#### `plots/`
Use for generated plots, images, or plot-ready artifacts.

Examples:
- loss curves
- metric comparison charts
- PCA scatter plots

### Rule
Do not mix handwritten docs with generated outputs.

Generated outputs belong in `outputs/`, not in `docs/`.

---

## `data/`

This folder stores project datasets and small input files used by experiments and implementations.

Examples:
- small CSV datasets
- cleaned study datasets
- synthetic datasets saved for reuse
- configuration-like input files if needed

### Rule
Keep `data/` for input data, not generated results.

If a dataset becomes very large, store only the strategically necessary subset in-repo and document the rest externally.

Do not place outputs here.

---

## Module Ownership Rules

Each implementation belongs to one main module area.

## `common/`
Use `common/` only for utilities that are truly shared across multiple phases or model families.

Examples:
- dataset abstractions
- preprocessing pipeline helpers
- evaluation harness support
- general metrics utilities
- common math helpers not naturally covered by Eigen

Do **not** dump unrelated code into `common/`.

A utility should go into `common/` only if it is clearly cross-cutting.

## Model-specific folders
If code belongs clearly to one concept, keep it in that concept’s folder.

Examples:
- linear regression → `linear_models/`
- decision tree → `trees/`
- k-means → `unsupervised/`

This keeps ownership clear and reduces future confusion.

---

## Naming Conventions

## Files
Prefer lowercase snake_case filenames.

Examples:
- `linear_regression.hpp`
- `training_history.cpp`
- `evaluation_harness.hpp`

## Classes / structs
Use PascalCase.

Examples:
- `LinearRegression`
- `TrainingHistory`
- `DecisionTreeNode`

## Functions
Use snake_case.

Examples:
- `train_test_split`
- `fit_batch`
- `compute_accuracy`

## Namespaces
Keep the project namespace consistent.

Recommended pattern:

```cpp
namespace ml {
    ...
}
```

And nested namespaces when appropriate:

```cpp
namespace ml::common {
    ...
}
```

or the equivalent compatible style depending on the C++ version you use.

---

## Reusable Code vs Experiment Code

This is one of the most important rules in the project.

## Reusable code
Put code in `include/` + `src/` when it is:
- conceptually part of the ML library
- expected to be reused
- part of the model or utility architecture
- not tied to one temporary experiment

## Experiment code
Put code in `app/test_runner.cpp` or `experiments/` when it is:
- only for validation
- only for demonstration
- tied to one narrow experiment
- not intended to become project infrastructure

### Rule
Do not let temporary experiment code become permanent architecture by accident.

---

## Documentation Rules

Every serious phase should leave behind:

- at least one theory document in `docs/theory/`
- code modules in `include/` + `src/` if implementation depth requires it
- experiment outputs in `outputs/`
- optional experiment support files in `experiments/`

This prevents progress from existing only in chat history or temporary code.

---

## Output Rules

When outputs are saved, they should be:
- grouped by phase
- named clearly
- easy to delete or regenerate if needed

### In-repo vs external artifact policy

The default policy is:

#### Commit to the repository
- small summary JSON/CSV
- important experiment notes
- small representative output artifacts

#### Do not commit to the repository
- huge generated files
- disposable temporary outputs
- very large datasets unless strategically important

This rule exists to keep the repository useful and readable without turning it into a storage dump for heavy generated artifacts.

When in doubt, commit the small summary that explains the result, not every large intermediate file that was produced while getting there.

Recommended naming style:

```text
outputs/phase-3-linear-models/
outputs/phase-5-optimization/
outputs/phase-8-unsupervised/
```

Inside those folders, use descriptive names like:
- `metrics_summary.json`
- `ridge_vs_unregularized.csv`
- `optimizer_comparison.csv`
- `experiment_notes.md`

---

## App / Test / Experiment Separation

The expected boundary is:

### `app/main.cpp`
Minimal and stable.

### `app/test_runner.cpp`
Structured manual validation.

### `experiments/`
Bigger, phase-specific experimental workflows.

### `include/` + `src/`
Reusable implementation code.

This separation should remain stable throughout the project.

---

## Anti-Drift Rules

To avoid repeating the previous project’s problems:

- do not keep giant commented historical test blocks
- do not use `main.cpp` as a phase archive
- do not duplicate reusable logic in runners
- do not place theory in code comments when it belongs in docs
- do not place generated outputs in handwritten docs folders
- do not use `common/` as a dumping ground

---

## Final Principle

The repository should stay organized so that:
- docs explain the project
- `include/` and `src/` implement the reusable ML core
- `app/` runs entrypoints and manual checks
- `experiments/` supports broader experiments
- `outputs/` stores generated artifacts
- `data/` stores inputs

This structure should make later phases easier, not heavier.
