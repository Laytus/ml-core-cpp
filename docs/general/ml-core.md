# ML Core

## Purpose

ML Core is a serious Machine Learning study project designed to build the full practical and theoretical foundation needed **before moving into Deep Learning**.

This project is not a toy implementation pass.

It is intended to cover the core of classical Machine Learning in a way that is:
- conceptually solid
- implementation-oriented
- mathematically grounded
- optimized for fast progress without oversimplifying the material

The goal is to study Machine Learning seriously enough that the transition to Deep Learning happens on top of a real foundation rather than on top of fragmented intuition.

---

## Main Objective

Build a strong ML core through a combination of:
- theory
- focused implementations
- experiments
- structured documentation
- progressive model comparison

The project should produce:

1. a serious understanding of the main ML ideas
2. a clean C++ codebase for the most important implementations
3. a clear bridge from classical ML to Deep Learning
4. a structured study reference that can be reused later

---

## What “ML Core” means here

In this project, “ML Core” means the set of concepts that should be understood well before starting Deep Learning seriously.

This includes:

- mathematical and statistical foundations for ML
- proper evaluation methodology
- linear models in multivariate form
- optimization for trainable models
- regularization and generalization
- trees and ensemble intuition
- unsupervised learning essentials
- probabilistic ML essentials
- a direct bridge to neural networks and backpropagation

So this project is not meant to cover literally every topic in Machine Learning.

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
- helper tools where appropriate

and **not** through reducing the concepts to toy-only versions.

### 2. Theory and implementation must stay connected

Every major concept should be understood:
- mathematically
- conceptually
- programmatically

This means the project should avoid two extremes:
- theory with no code intuition
- code with no theoretical grounding

### 3. C++ remains the implementation language

The project is still a C++ project.

But unlike the previous repo, this one is allowed to use helper tools when they reduce unnecessary infrastructure work and keep the focus on actual ML.

### 4. Build ML knowledge, not unnecessary low-level tooling

The goal is Machine Learning.

So supporting tools are allowed when they help keep the project focused on ML rather than on unrelated engineering overhead.

This is why the project will use:
- Eigen for matrix operations
- CSV / data utilities
- plotting / export helpers
- optional external reference comparisons

### 5. Explicit implementation depth

Not every topic should be implemented at the same depth.

To keep the project serious and fast, each phase or sub-phase should be classified as one of the following:

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

This project will use **Eigen** as the matrix library.

Reason:
- matrix and vector operations will become central very quickly
- writing a full personal matrix library would add too much scope
- the project goal is ML Core, not numerical library engineering

A small personal matrix module may be explored later as a side exercise if useful, but it is not part of the main ML Core path.

### Other allowed helpers

The project may also use:
- CSV/data utilities
- plotting/export helpers
- optional external reference comparisons

These tools are allowed because they support the study of ML instead of replacing it.

---

## What this project is trying to fix

This project exists because a first implementation-based ML repo was useful, but too limited.

That earlier work was good for:
- first intuition
- C++ practice
- simple model mechanics

But it was not enough for:
- serious ML readiness
- strong theoretical grounding
- a confident transition into DL

So ML Core is the next step:
- broader
- deeper
- more structured
- more honest about what “real preparation” requires

---

## Scope

The project is expected to cover:

- math and statistical foundations for ML
- data pipeline and evaluation methodology
- multivariate linear models
- multivariate logistic regression
- optimization for ML
- trees and ensemble foundations
- distance and kernel intuition
- unsupervised learning essentials
- probabilistic ML essentials
- bridge to Deep Learning

This is the intended serious core.

Topics outside this scope may be mentioned, but they are not the main target unless later added explicitly.

---

## Non-Goals

To keep the project focused, these are not primary goals:

- covering every ML subfield exhaustively
- building production MLOps systems
- distributed training systems
- GPU systems programming
- full numerical computing infrastructure from scratch
- full research-level treatment of every model family

Also, this project should not become:
- a general-purpose ML framework
- a replacement for mature ML libraries
- a pure math notebook without implementation depth

---

## Working Method

For each step, the project will follow the same concise workflow:

1. add theory to the corresponding doc
2. write the step’s concise action plan
3. define header file(s)
4. define validations
5. define implementation action plan without code
6. define the test plan

This workflow is mandatory because it keeps:
- theory aligned with implementation
- scope explicit
- code structure clean
- progress trackable

---

## Expected Outcome

By the end of ML Core, the project should provide:

- a serious understanding of classical ML foundations
- strong intuition for optimization and generalization
- practical experience with vectorized ML implementations
- a clear understanding of evaluation methodology
- a useful foundation in trees, unsupervised learning, and probabilistic ML
- a direct conceptual bridge to neural networks and Deep Learning

---

## Final Positioning

ML Core should be understood as:

- a serious ML foundation project before DL
- a structured and optimized ML study roadmap
- a C++ implementation project supported by practical helper tools
- a bridge between introductory ML intuition and full DL preparation