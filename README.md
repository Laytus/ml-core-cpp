# ML Core

A serious C++ project to build the full Machine Learning core needed **before Deep Learning**.

---

## Purpose

ML Core is designed to build a strong ML foundation through a combination of:
- theory
- implementation
- experiments
- structured documentation

The goal is to study Machine Learning seriously enough that the transition to Deep Learning happens on top of a real base rather than on top of fragmented intuition.

This project is:
- conceptually solid
- implementation-oriented
- mathematically grounded
- optimized for fast progress without oversimplifying the material

---

## Project Positioning

This repository should be treated as:
- a serious ML foundation project before Deep Learning
- a structured and optimized study roadmap
- a C++ implementation project supported by practical helper tools
- a bridge between introductory ML intuition and full DL preparation

It should **not** be treated as:
- a toy ML repo
- a rushed overview of random ML topics
- a full production ML framework
- a replacement for mature ML libraries

---

## Scope

ML Core is intended to cover the serious core of Machine Learning that should be understood before starting Deep Learning seriously.

This includes:
- mathematical and statistical foundations for ML
- data pipeline and evaluation methodology
- multivariate linear models
- multivariate logistic regression
- optimization for ML
- trees and ensemble intuition
- distance and kernel intuition
- unsupervised learning essentials
- probabilistic ML essentials
- bridge to neural networks and backpropagation

So this project is not trying to cover literally every ML topic.

It is trying to cover the **real core** that matters before DL.

---

## Philosophy

### Serious, but optimized

The project should move as fast as possible, but without artificially simplifying the concepts.

Speed matters.

But correctness of scope matters more.

### Theory and implementation stay connected

Every major concept should be understood:
- mathematically
- conceptually
- programmatically

This project avoids both extremes:
- theory with no code intuition
- code with no theoretical grounding

### Build ML knowledge, not unnecessary infrastructure

The goal is Machine Learning.

So helper tools are allowed when they reduce unrelated engineering overhead and keep the focus on ML.

---

## Tooling

ML Core is implemented in **C++17** and uses:

- **CMake**
- **Eigen** for matrix operations
- CSV/data utilities where useful
- plotting/export helpers where useful
- optional external reference comparisons

### Why Eigen is used

Eigen is used because:
- matrix operations become central very quickly in serious ML
- the project goal is ML, not writing a full matrix library
- using Eigen avoids wasting time on infrastructure that is not the main learning target

---

## Current Build Targets

The repository currently uses two executables:

### `ml_core_app`
A clean and minimal project entrypoint.

### `ml_core_tests`
A structured manual validation runner for:
- sanity checks
- per-phase demos
- manual testing during development

This keeps the main app clean while still allowing fast iteration.

---

## Repository Structure

Main repository areas:

```text
docs/
include/
src/
data/
experiments/
outputs/
app/
```

### Responsibilities

- `docs/` → project docs, theory notes, action plans
- `include/` → reusable public headers
- `src/` → reusable implementations
- `data/` → input datasets
- `experiments/` → phase-specific experiment workflows
- `outputs/` → generated artifacts
- `app/` → executable entrypoints

For the detailed structural rules, see:
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

## Working Method

For each step, the project follows the same concise workflow:

1. add theory to the corresponding doc
2. write the step’s concise action plan
3. define header file(s)
4. define validations
5. define implementation action plan without code
6. define the test plan

This workflow keeps:
- theory aligned with implementation
- scope explicit
- code structure clean
- progress trackable

---

## Execution Plan

The full roadmap is tracked in:

- `docs/general/action-plan.md`

The high-level identity of the project is defined in:

- `docs/general/ml-core.md`

---

## Expected Outcome

By the end of ML Core, the repository should provide:

- a serious understanding of classical ML foundations
- strong intuition for optimization and generalization
- practical experience with vectorized ML implementations
- a clear understanding of evaluation methodology
- a useful foundation in trees, unsupervised learning, and probabilistic ML
- a direct conceptual bridge to neural networks and Deep Learning

Most importantly, it should leave you in a position where starting DL is a natural next step rather than a leap into partially understood ideas.

---

## Final Note

This project is meant to build the actual base layer that was still missing before Deep Learning.

It is not a warm-up anymore.

It is the real ML foundation project.
