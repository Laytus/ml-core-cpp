# Action Plan – ML Core

## General Objective

Build a serious and complete Machine Learning core before Deep Learning, with strong theoretical grounding, efficient implementation choices, and a codebase that avoids disposable work.

This action plan is designed to:
- move fast without oversimplifying the concepts
- avoid useless implementations that would have to be rewritten later
- prioritize reusable components and correct sequencing
- cover the real ML core needed before DL
- remain detailed enough to guide execution phase by phase without relying on memory alone

---

## How to Use This Action Plan

This document is not only a roadmap.

It is the execution reference for the entire project.

For every phase, it should answer:
- why the phase exists
- what must already be ready before starting it
- what exactly must be produced
- what should be implemented vs kept theoretical
- what the exit condition is before moving on

Each phase should therefore be treated as a contract.

A phase is only complete when its exit criteria are satisfied.

---

## Global Project Rules

### 1. No disposable scalar-first implementations

Do **not** build toy scalar versions if the real project will need vectorized or matrix-based versions later.

Default target:
- vectorized
- multivariate
- dataset-level
- reusable

### 2. Use helper tools when infrastructure is not the learning target

Use:
- **Eigen** for matrix operations
- CSV/data utilities
- plotting/export helpers
- optional external reference comparisons

Do **not** spend project time rebuilding infrastructure that does not directly improve ML understanding.

### 3. Implementation depth must be explicit

Each phase or major block must be classified as:
- **Level A — full implementation**
- **Level B — partial implementation + experiments**
- **Level C — theory + small demo only**

### 4. Build in dependency order

A later serious implementation should not depend on an earlier oversimplified version.

So the order must always follow:
- foundations first
- reusable utilities second
- real models third
- bridge concepts after the core is strong

### 5. Keep the same workflow for every step

For each step:
1. add theory to the corresponding doc
2. write the step’s concise action plan
3. define header file(s)
4. define validations
5. define implementation action plan without code
6. define the test plan

### 6. Every phase must leave reusable artifacts

A phase should never end with only temporary notebook-style understanding.

Each phase must leave behind some combination of:
- stable code modules
- stable docs
- experiment outputs
- reusable helpers
- explicit conclusions

### 7. Every serious implementation must be evaluated in context

When a model or method is implemented seriously, it must be tested in a real mini-workflow:
- data preparation
- train/validation/test logic
- metrics
- interpretation of results

---

## Phase Overview

The project will cover all phases from 0 to 11.

Each phase below includes:
- purpose
- implementation depth
- estimated hours
- prerequisites
- detailed tasks
- expected files / deliverables
- concrete outputs
- exit criteria
- optimization notes

---

## Phase 0 – Reset and Infrastructure

**Goal:** Prepare the repository and tooling for a serious ML project.

**Level:** A

**Estimated effort:** 6–10 hours

### Why this phase exists
This phase exists to prevent structural refactors later.

The previous project suffered from the natural tendency to grow around early choices.

This time, the project structure should be correct before model work starts.

### Prerequisites
- none

### Detailed tasks
- [x] Define the top-level repo structure for a serious ML project
- [x] Define the folder structure for:
  - docs/general
  - docs/theory
  - include
  - src
  - data
  - experiments
  - outputs
- [x] Integrate Eigen cleanly into the build system
- [x] Update CMake structure for matrix-based code and reusable modules
- [x] Define file naming rules and module ownership conventions
- [x] Define how demos and experiments will be separated from reusable implementation code
- [x] Define how outputs will be stored:
  - CSV
  - JSON
  - notes/results
- [x] Decide which comparison artifacts will be kept in-repo vs external
- [x] Write the high-level project identity docs

### Expected files / deliverables
- `docs/general/ml-core.md`
- `docs/general/action-plan.md`
- updated `CMakeLists.txt`
- created/confirmed top-level folders:
  - `docs/general/`
  - `docs/theory/`
  - `include/`
  - `src/`
  - `data/`
  - `experiments/`
  - `outputs/`
- optional helper files for build/dependency notes:
  - `docs/general/build-notes.md`
  - `docs/general/repo-structure.md`

### Concrete outputs
- working repository structure
- working CMake + Eigen integration
- stable directory layout
- project identity docs ready

### Exit criteria
- [x] Repo structure is stable enough that later phases should not need a redesign
- [x] Eigen builds cleanly
- [x] There is a clear distinction between library code, experiments, docs, and outputs
- [x] Project naming and structure are consistent

### Optimization note
Do this once and do it correctly. Any ambiguity here will multiply later.

---

## Phase 1 – Math and Statistical Foundations

**Goal:** Build the real mathematical and statistical base required before implementing data pipelines and ML models.

**Level:** A

**Estimated effort:** 12–18 hours

**Status:** Complete

### Why this phase exists
This phase is the minimum serious base required before implementing real ML models.

Without it, later work becomes procedural instead of grounded.

Phase 1 establishes the shared mathematical conventions, validation layer, Eigen usage patterns, reusable math operations, descriptive statistics, and feature scaling utilities that later phases will depend on.

### Prerequisites
- Phase 0 complete

### Detailed tasks
- [x] Write theory notes for vector and matrix fundamentals
- [x] Write theory notes for matrix multiplication and geometric interpretation
- [x] Write theory notes for derivatives, gradients, and partial derivatives
- [x] Write theory notes for chain rule and why it matters for trainable models
- [x] Write theory notes for expectation, variance, covariance, and distributions intuition
- [x] Write theory notes for train / validation / test roles
- [x] Write theory notes for:
  - underfitting
  - overfitting
  - bias
  - variance
- [x] Implement only minimal math helpers that will be reused later and are not already handled naturally by Eigen
- [x] Create one concise validation/demo layer showing:
  - vectorized expressions
  - gradient interpretation
  - covariance / variance intuition if useful

### Expected files / deliverables
- theory docs:
  - `docs/theory/math-foundations.md`
  - `docs/theory/calculus-foundations.md`
  - `docs/theory/statistical-learning-foundations.md`
- optional minimal reusable helpers:
  - `include/ml/common/math_utils.hpp`
  - `src/common/math_utils.cpp`
- optional small validation/demo file:
  - `experiments/phase-1-math-sanity/`
  - `outputs/phase-1-math-sanity/`

### Concrete outputs
- theory docs for math foundations
- theory docs for statistical learning foundations
- minimal reusable math helpers
- one or more small sanity demos for key ideas

### Exit criteria
- [x] You can explain gradients, vectorized model form, and bias/variance clearly without ambiguity
- [x] No important later phase depends on undefined math/stat terms
- [x] Only necessary utilities were implemented; no unnecessary matrix reimplementation exists

### Optimization note
Keep code light here. The real value of this phase is conceptual clarity plus a few reusable helpers.

---

## Phase 2 – Data Pipeline and Evaluation Methodology

**Goal:** Build the real evaluation workflow used by serious ML work.

**Level:** A

**Estimated effort:** 10–16 hours

### Why this phase exists
If evaluation structure is added too late, all early models become inconsistent and require rework.

This phase defines the environment inside which all later models should be studied.

### Prerequisites
- Phase 0 complete
- Phase 1 complete

### Detailed tasks
- [x] Define dataset abstractions for supervised learning
- [x] Implement train / validation / test split utilities
- [x] Decide on deterministic vs shuffled split strategy and expose both where useful
- [x] Implement k-fold cross-validation utilities
- [x] Define preprocessing pipeline rules so transforms are fitted only on training data
- [x] Add explicit theory and examples for data leakage
- [x] Define baseline evaluation flow:
  - baseline predictor
  - trained predictor
  - metric comparison
- [x] Build a reusable evaluation harness structure that later models can plug into
- [x] Define how model metrics and experiment summaries are exported
- [x] Add theory notes on proper evaluation discipline

### Expected files / deliverables
- theory doc:
  - `docs/theory/evaluation-methodology.md`
- reusable data/evaluation headers and sources:
  - `include/ml/common/dataset.hpp`
  - `include/ml/common/data_split.hpp`
  - `include/ml/common/cross_validation.hpp`
  - `include/ml/common/preprocessing_pipeline.hpp`
  - `include/ml/common/evaluation_harness.hpp`
  - `src/common/data_split.cpp`
  - `src/common/cross_validation.cpp`
  - `src/common/preprocessing_pipeline.cpp`
  - `src/common/evaluation_harness.cpp`
- optional experiment folders:
  - `experiments/phase-2-evaluation-methodology/`
  - `outputs/phase-2-evaluation-methodology/`

### Concrete outputs
- reusable dataset split utilities
- reusable cross-validation utilities
- preprocessing discipline rules
- evaluation harness skeleton
- leakage documentation

### Exit criteria
- [x] A later model can be inserted into a reusable train/validate/test workflow without redesign
- [x] Cross-validation exists as a reusable utility
- [x] Leakage prevention rules are explicit in both code structure and docs
- [x] Baseline-vs-model evaluation is clearly defined

### Optimization note
This phase should be finished before any “serious” model implementation begins.

---

## Phase 3 – Linear Models, Properly

**Goal:** Implement serious multivariate linear models instead of toy scalar versions.

**Level:** A

**Estimated effort:** 12–18 hours

### Why this phase exists
Linear models are the cleanest place to learn vectorized supervised learning properly.

This is where the project shifts from introductory mechanics to real model structure.

### Prerequisites
- Phases 0–2 complete

### Detailed tasks
- [x] Write theory for multivariate linear regression
- [x] Write theory for linear models as matrix operations
- [x] Write theory for MSE in vectorized form
- [x] Write theory for normal equation concept and why it matters even if not the main training route
- [x] Write theory for regularization:
  - Ridge in detail
  - Lasso conceptually
- [x] Define reusable linear-model interface decisions
- [x] Implement multivariate linear regression with Eigen
- [x] Implement vectorized prediction
- [x] Implement vectorized MSE
- [x] Implement Ridge regularization support
- [x] Add experiments comparing:
  - unregularized vs Ridge
  - scaled vs unscaled data
  - different learning rates / convergence behavior
- [x] Add interpretation notes for residual behavior

### Expected files / deliverables
- theory doc:
  - `docs/theory/linear-models.md`
- model headers and sources:
  - `include/ml/linear_models/linear_regression.hpp`
  - `src/linear_models/linear_regression.cpp`
- optional supporting headers:
  - `include/ml/linear_models/regularization.hpp`
  - `src/linear_models/regularization.cpp`
- experiment folders:
  - `experiments/phase-3-linear-models/`
  - `outputs/phase-3-linear-models/`

### Concrete outputs
- multivariate linear regression implementation
- vectorized loss and prediction flow
- regularized linear model support
- experiments and summaries

### Exit criteria
- [x] The model is multivariate and vectorized
- [x] Regularization is integrated cleanly, not bolted on later
- [x] The model fits naturally into the evaluation pipeline
- [x] At least one experiment compares regularized and unregularized behavior

### Optimization note
Do not build a second temporary linear-regression path. Build the serious one directly.

---

## Phase 4 – Linear Classification Models

**Goal:** Implement serious classification foundations in vectorized form, covering both binary logistic regression and multiclass softmax regression.

**Level:** A

**Estimated effort:** 18–28 hours

### Why this phase exists
This is the first real classification foundation phase and one of the key bridges to DL.

It introduces logits, probabilities, cross-entropy, thresholds, decision boundaries, class-boundary thinking, and multiclass probability distributions in a serious form.

This phase is divided into two parts:
- **Phase 4A – Binary Logistic Regression**
- **Phase 4B – Softmax Regression for Multiclass Classification**

Binary logistic regression introduces the core classification pattern:

```txt
linear score → probability → cross-entropy → class prediction
```

Softmax regression extends that same pattern to multiple classes:

```txt
class logits → softmax probabilities → categorical cross-entropy → argmax prediction
```

### Prerequisites
- Phases 0–3 complete

### Detailed tasks

#### Phase 4A – Binary Logistic Regression
- [x] Write theory for multivariate logistic regression
- [x] Write theory for logits and sigmoid interpretation
- [x] Write theory for binary cross-entropy in vectorized form
- [x] Write theory for thresholding and decision boundaries
- [x] Write theory for regularization in classification
- [x] Define reusable logistic-model interface decisions
- [x] Implement vectorized logistic regression with Eigen
- [x] Implement logits, probabilities, and class predictions cleanly
- [x] Implement vectorized BCE
- [x] Integrate regularization support
- [x] Integrate with evaluation harness and metrics
- [x] Add experiments for:
  - threshold variation
  - regularization variation
  - class-boundary interpretation
  - evaluation across multiple metrics

#### Phase 4B – Softmax Regression for Multiclass Classification
- [x] Add softmax/multiclass theory bridge
- [x] Define reusable softmax-model interface decisions
- [x] Implement softmax regression for multiclass classification
- [x] Implement vectorized softmax with numerical stability
- [x] Implement categorical cross-entropy
- [x] Implement multiclass prediction with argmax
- [x] Add multiclass classification metrics and evaluation support
- [x] Add experiments for multiclass classification behavior

### Expected files / deliverables
- theory doc:
  - `docs/theory/logistic-regression.md`
- binary classification model headers and sources:
  - `include/ml/linear_models/logistic_regression.hpp`
  - `src/linear_models/logistic_regression.cpp`
- multiclass classification model headers and sources:
  - `include/ml/linear_models/softmax_regression.hpp`
  - `src/linear_models/softmax_regression.cpp`
- binary classification helpers:
  - `include/ml/common/classification_utils.hpp`
  - `src/common/classification_utils.cpp`
  - `include/ml/common/classification_metrics.hpp`
  - `src/common/classification_metrics.cpp`
  - `include/ml/common/classification_evaluation.hpp`
  - `src/common/classification_evaluation.cpp`
- multiclass classification helpers:
  - `include/ml/common/multiclass_metrics.hpp`
  - `src/common/multiclass_metrics.cpp`
- experiment folders:
  - `experiments/phase-4-logistic-regression/`
  - `outputs/phase-4-logistic-regression/`

### Concrete outputs
- multivariate binary logistic regression implementation
- probability and binary class-prediction pipeline
- BCE implementation
- binary classification metrics and evaluation harness integration
- threshold/evaluation experiments
- softmax regression implementation
- stable row-wise softmax
- categorical cross-entropy implementation
- multiclass argmax prediction
- multiclass classification metrics and evaluation support
- multiclass classification experiments

### Exit criteria
- [x] Binary logistic regression is multivariate and vectorized
- [x] Threshold analysis is explicit rather than hardcoded and forgotten
- [x] The binary model integrates naturally with the evaluation framework
- [x] The docs clearly connect logits, BCE, and decision boundaries
- [x] Softmax regression works as a vectorized multiclass classifier
- [x] Multiclass cross-entropy and prediction behavior are documented and tested
- [x] Multiclass evaluation metrics are reusable and tested
- [x] At least one experiment demonstrates multiclass classification behavior

### Optimization note
This phase should be built as the real classification foundation, not as a minimal extension of the previous project.

Binary logistic regression and softmax regression should share concepts and utilities where appropriate, but their model APIs should remain explicit and easy to reason about.

---

## Phase 5 – Optimization for ML

**Goal:** Build the real optimization intuition needed before Deep Learning.

**Level:** A

**Estimated effort:** 14–22 hours

### Why this phase exists
Optimization is one of the most important conceptual bridges to DL.

It should therefore exist as a reusable and serious part of the project rather than scattered model-specific logic.

### Prerequisites
- Phases 0–4 complete

### Detailed tasks
- [x] Write theory for batch GD, SGD, and mini-batch GD
- [x] Write theory for momentum and why it helps
- [x] Write theory for conditioning, scaling, and optimization geometry
- [x] Write theory for initialization sensitivity and convergence behavior
- [x] Write theory bridge for adaptive optimizers conceptually
- [x] Design reusable optimizer interfaces that can support multiple trainable models
- [x] Refactor any model-specific training logic into reusable optimizer-aware structures
- [x] Implement batch GD
- [x] Implement SGD
- [x] Implement mini-batch GD
- [x] Implement momentum
- [x] Implement reusable logging/history structures for optimization runs
- [x] Add experiments comparing:
  - batch vs SGD vs mini-batch
  - with and without momentum
  - different learning rates
  - scaled vs unscaled optimization behavior
- [x] Summarize optimizer behavior in docs/outputs

### Expected files / deliverables
- theory doc:
  - `docs/theory/optimization.md`
- optimizer headers and sources:
  - `include/ml/optimization/optimizer.hpp`
  - `include/ml/optimization/gradient_descent.hpp`
  - `include/ml/optimization/sgd.hpp`
  - `include/ml/optimization/minibatch_gd.hpp`
  - `include/ml/optimization/momentum.hpp`
  - `include/ml/optimization/training_history.hpp`
  - `src/optimization/gradient_descent.cpp`
  - `src/optimization/sgd.cpp`
  - `src/optimization/minibatch_gd.cpp`
  - `src/optimization/momentum.cpp`
  - `src/optimization/training_history.cpp`
- experiment folders:
  - `experiments/phase-5-optimization/`
  - `outputs/phase-5-optimization/`

### Concrete outputs
- reusable optimization framework
- optimizer history/logging utilities
- comparison experiments
- optimization theory docs

### Exit criteria
- [ ] At least two trainable models can share the optimization framework cleanly
- [ ] Batch / SGD / mini-batch are implemented and comparable
- [ ] Momentum is integrated without ad-hoc duplication
- [ ] Convergence analysis is based on structured outputs, not manual inspection only

### Optimization note
This phase must reduce duplication, not add another layer of it.

---

## Phase 6 – Trees and Ensembles

**Goal:** Build a real non-linear tabular ML foundation.

**Level:** B

**Estimated effort:** 16–24 hours

### Why this phase exists
Trees are the main conceptual entry point to non-linear tabular ML.

A real simple tree builder is worth implementing. Full ensemble systems are better treated more selectively.

### Prerequisites
- Phases 0–5 complete

### Detailed tasks
- [ ] Write theory for split quality, Gini, entropy, and weighted child impurity
- [ ] Write theory for recursive tree construction and stopping rules
- [ ] Write theory for pruning intuition
- [ ] Write theory for ensembles:
  - random forest intuition
  - gradient boosting intuition
- [ ] Replace the previous isolated utilities mindset with a real tree-building workflow
- [ ] Implement weighted split scoring
- [ ] Implement candidate-threshold evaluation
- [ ] Implement best-threshold selection for the simplified setting
- [ ] Implement recursive tree growth
- [ ] Implement stopping rules
- [ ] Implement tree-based prediction traversal
- [ ] Add experiments comparing tree behavior under different depth / stopping settings
- [ ] Keep random forest and boosting at theory + light experimental comparison level unless later needed more deeply

### Expected files / deliverables
- theory doc:
  - `docs/theory/trees-and-ensembles.md`
- decision-tree headers and sources:
  - `include/ml/trees/decision_tree.hpp`
  - `include/ml/trees/tree_node.hpp`
  - `include/ml/trees/split_scoring.hpp`
  - `include/ml/trees/tree_builder.hpp`
  - `src/trees/decision_tree.cpp`
  - `src/trees/split_scoring.cpp`
  - `src/trees/tree_builder.cpp`
- optional ensemble comparison note/output files:
  - `experiments/phase-6-trees-ensembles/`
  - `outputs/phase-6-trees-ensembles/`

### Concrete outputs
- real simple Decision Tree implementation
- split scoring utilities
- recursive builder
- tree prediction flow
- ensemble theory notes and comparison summaries

### Exit criteria
- [ ] The project contains a genuine simple Decision Tree, not only isolated split utilities
- [ ] Split scoring is quantitative and reusable
- [ ] Tree growth and stopping rules are explicit
- [ ] Ensemble topics are at least conceptually integrated even if not fully implemented

### Optimization note
Implement one tree properly. Do not fully implement every ensemble method unless it becomes strategically necessary.

---

## Phase 7 – Distance and Kernel Thinking

**Goal:** Complete the classical ML intuition around geometry, neighborhoods, and margins.

**Level:** B

**Estimated effort:** 8–14 hours

### Why this phase exists
This phase fills the geometric part of ML intuition that is often underdeveloped before DL.

### Prerequisites
- Phases 0–6 complete

### Detailed tasks
- [ ] Write theory for multivariate k-NN and distance metrics
- [ ] Write theory for curse of dimensionality
- [ ] Write theory for margins and SVM intuition
- [ ] Write theory for kernels and feature-space lifting intuition
- [ ] Upgrade k-NN from the earlier simplified form into a more serious multivariate experimental setting where useful
- [ ] Add experiments comparing different metrics or neighborhood behavior where useful
- [ ] Keep SVM and kernel work at theory + small demo level unless deeper implementation becomes clearly justified

### Expected files / deliverables
- theory doc:
  - `docs/theory/distance-and-kernel-thinking.md`
- optional upgraded k-NN files:
  - `include/ml/distance/knn_classifier.hpp`
  - `include/ml/distance/distance_metrics.hpp`
  - `src/distance/knn_classifier.cpp`
  - `src/distance/distance_metrics.cpp`
- optional small SVM/kernel demo docs/notes:
  - `experiments/phase-7-distance-kernel/`
  - `outputs/phase-7-distance-kernel/`

### Concrete outputs
- stronger distance-based learning theory docs
- upgraded k-NN experiments
- SVM/kernel conceptual bridge docs

### Exit criteria
- [ ] Distance-based learning is covered beyond the earlier toy intuition
- [ ] Curse of dimensionality is explicitly understood and documented
- [ ] Margin and kernel ideas are clear enough to recognize their importance without needing a full solver implementation

### Optimization note
This phase should deepen geometric intuition, not become a heavy optimization-project detour.

---

## Phase 8 – Unsupervised Learning

**Goal:** Cover the unsupervised ML core needed before DL.

**Level:** A

**Estimated effort:** 14–22 hours

### Why this phase exists
Unsupervised learning fills major conceptual gaps around structure discovery, variance, and latent representation.

It is too important to leave as a side note.

### Prerequisites
- Phases 0–7 complete

### Detailed tasks
- [ ] Write theory for clustering fundamentals
- [ ] Write theory for k-means objective and centroid behavior
- [ ] Write theory for PCA, covariance, principal directions, and explained variance
- [ ] Implement k-means
- [ ] Implement cluster assignment and centroid update loops cleanly
- [ ] Implement PCA using Eigen-supported matrix operations
- [ ] Add dimensionality-reduction experiments
- [ ] Add reconstruction / explained variance experiments where useful
- [ ] Compare unsupervised representations qualitatively through saved outputs/plots

### Expected files / deliverables
- theory doc:
  - `docs/theory/unsupervised-learning.md`
- model headers and sources:
  - `include/ml/unsupervised/kmeans.hpp`
  - `include/ml/unsupervised/pca.hpp`
  - `src/unsupervised/kmeans.cpp`
  - `src/unsupervised/pca.cpp`
- experiment folders:
  - `experiments/phase-8-unsupervised/`
  - `outputs/phase-8-unsupervised/`

### Concrete outputs
- k-means implementation
- PCA implementation
- clustering and dimensionality-reduction experiments
- unsupervised learning theory docs

### Exit criteria
- [ ] k-means runs end-to-end in a reusable way
- [ ] PCA exists as a real implementation, not only a theory note
- [ ] There are experiments showing at least variance or dimensionality-reduction behavior
- [ ] The connection between covariance structure and PCA is clearly documented

### Optimization note
This phase is worth implementing seriously because it builds representation intuition that matters before DL.

---

## Phase 9 – Probabilistic ML Essentials

**Goal:** Fill the probability gap that often blocks true ML understanding.

**Level:** B

**Estimated effort:** 10–16 hours

### Why this phase exists
Many ML ideas become much clearer once likelihood, uncertainty, and probabilistic interpretation are made explicit.

### Prerequisites
- Phases 0–8 complete

### Detailed tasks
- [ ] Write theory for maximum likelihood and MAP intuition
- [ ] Write theory for probabilistic outputs and uncertainty concepts
- [ ] Write theory for Naive Bayes and simple Gaussian-model thinking
- [ ] Decide whether to implement Naive Bayes or another lightweight probabilistic classifier as the main concrete anchor
- [ ] Implement one lightweight probabilistic model if worthwhile
- [ ] Add small experiments showing how probabilistic assumptions affect behavior

### Expected files / deliverables
- theory doc:
  - `docs/theory/probabilistic-ml.md`
- optional lightweight probabilistic model files:
  - `include/ml/probabilistic/naive_bayes.hpp`
  - `src/probabilistic/naive_bayes.cpp`
- experiment folders:
  - `experiments/phase-9-probabilistic-ml/`
  - `outputs/phase-9-probabilistic-ml/`

### Concrete outputs
- probabilistic ML theory docs
- one lightweight probabilistic implementation or small demo anchor
- probability-centered experiment notes

### Exit criteria
- [ ] Likelihood-based thinking is clearly understood and documented
- [ ] At least one concrete probabilistic model or demo exists
- [ ] The project now has a probability-centered interpretation layer, not only deterministic prediction thinking

### Optimization note
Prioritize conceptual power over breadth of implementations.

---

## Phase 10 – Bridge to Deep Learning

**Goal:** Make the transition to DL conceptually natural.

**Level:** A

**Estimated effort:** 14–22 hours

### Why this phase exists
This phase is the real payoff of ML Core.

It should make DL feel like a natural extension of concepts already understood rather than a jump into a new universe.

### Prerequisites
- Phases 0–9 complete

### Detailed tasks
- [ ] Write theory for perceptron and its limitations
- [ ] Write theory for multilayer perceptron intuition
- [ ] Write theory for forward propagation in vectorized form
- [ ] Write theory for backpropagation as structured chain rule
- [ ] Write theory for activation functions in the neural-network context
- [ ] Write theory for neural networks as layered differentiable computation
- [ ] Design a tiny neural-network bridge module
- [ ] Implement a minimal perceptron or tiny MLP demo
- [ ] Implement vectorized forward pass
- [ ] Implement the simplest backprop-supporting logic needed for the bridge
- [ ] Explicitly connect previous optimization phases to neural-network training logic
- [ ] Summarize what stays the same from ML to DL and what changes

### Expected files / deliverables
- theory doc:
  - `docs/theory/dl-bridge.md`
- bridge model files:
  - `include/ml/dl_bridge/perceptron.hpp`
  - `include/ml/dl_bridge/mlp.hpp`
  - `src/dl_bridge/perceptron.cpp`
  - `src/dl_bridge/mlp.cpp`
- optional helper files:
  - `include/ml/dl_bridge/activations.hpp`
  - `src/dl_bridge/activations.cpp`
- experiment folders:
  - `experiments/phase-10-dl-bridge/`
  - `outputs/phase-10-dl-bridge/`

### Concrete outputs
- DL-bridge theory docs
- tiny neural-network bridge implementation or demo
- explicit conceptual mapping from classical ML to DL

### Exit criteria
- [ ] You can explain backprop as chain rule over layered computation clearly
- [ ] There is a concrete minimal neural-network bridge artifact in code or demo form
- [ ] The conceptual transition from logistic regression / optimization to neural nets is explicit and documented

### Optimization note
This phase should stop exactly at the point where the DL project can begin cleanly.

---

## Phase 11 – Wrap-Up and Transition

**Goal:** Freeze ML Core cleanly and prepare the DL roadmap.

**Level:** A

**Estimated effort:** 5–8 hours

### Why this phase exists
Without a wrap-up, hard-earned structure gets lost and the DL phase starts without a stable handoff.

### Prerequisites
- Phases 0–10 complete

### Detailed tasks
- [ ] Update README and general positioning docs
- [ ] Summarize what was fully implemented vs partially implemented vs theory-only
- [ ] Summarize the strongest concepts gained from the project
- [ ] Identify the remaining weak points before DL
- [ ] Define the entry plan for the DL project
- [ ] Freeze the repo scope clearly so ML Core does not keep expanding indefinitely

### Expected files / deliverables
- updated general docs:
  - `README.md`
  - `docs/general/ml-core.md`
  - `docs/general/action-plan.md`
- wrap-up/transition docs:
  - `docs/general/ml-core-wrap-up.md`
  - `docs/general/dl-roadmap-entry.md`

### Concrete outputs
- final repo positioning
- implementation-depth summary
- remaining-gap summary
- DL roadmap entry document

### Exit criteria
- [ ] The project has a clear final identity
- [ ] The implemented/theory-only boundary is documented honestly
- [ ] The next DL project can begin with no ambiguity about what ML Core already covered

### Optimization note
This is not cosmetic. It is the handoff layer to the next major project.

---

## Recommended Execution Order

Follow the phases in this order:

1. Phase 0 – Reset and Infrastructure
2. Phase 1 – Math and Statistical Foundations
3. Phase 2 – Data Pipeline and Evaluation Methodology
4. Phase 3 – Linear Models, Properly
5. Phase 4 – Linear Classification Models
6. Phase 5 – Optimization for ML
7. Phase 6 – Trees and Ensembles
8. Phase 7 – Distance and Kernel Thinking
9. Phase 8 – Unsupervised Learning
10. Phase 9 – Probabilistic ML Essentials
11. Phase 10 – Bridge to Deep Learning
12. Phase 11 – Wrap-Up and Transition

This order is chosen to avoid useless implementation and dependency problems.
<!-- 
| Phase | Description | Time | Status |
|-------|-------------|------|--------|
| 0 | Reset and Infrastructure | 6–10 h | ☐ | 
| 0 | Reset and Infrastructure | 6–10 h | ☑ |  -->

---

## Phase Time Summary

Approximate hours per phase:

- Phase 0: 6–10 h
- Phase 1: 12–18 h
- Phase 2: 10–16 h
- Phase 3: 12–18 h
- Phase 4: 12–18 h
- Phase 5: 14–22 h
- Phase 6: 16–24 h
- Phase 7: 8–14 h
- Phase 8: 14–22 h
- Phase 9: 10–16 h
- Phase 10: 14–22 h
- Phase 11: 5–8 h

### Approximate total

- lower bound: **133 h**
- upper bound: **208 h**

This is a serious project.

The objective is not to make it artificially short.

The objective is to make it as efficient as possible **without losing the depth required before DL**.

---

## Final Positioning of This Action Plan

This action plan should be treated as:
- the real ML foundation project before Deep Learning
- a serious and optimized ML roadmap
- a dependency-aware implementation plan that avoids throwaway work
- the execution reference for the whole project

It should not be treated as:
- a toy ML repo
- a purely theoretical reading list
- a rushed overview of random ML topics

This is the project meant to build the actual base layer that was still missing.

---

## Quick Execution Table

| Phase | Level | Estimated Hours | Main Code Outputs | Main Theory Outputs | Exit Condition |
|---|---:|---:|---|---|---|
| 0. Reset and Infrastructure | A | 6–10 | Repo structure, CMake, Eigen integration | Project identity docs | Stable structure and clean build |
| 1. Math and Statistical Foundations | A | 12–18 | Minimal reusable helpers only | Math/stat foundations docs | No undefined core math/stat concepts remain |
| 2. Data Pipeline and Evaluation Methodology | A | 10–16 | Splits, CV, evaluation harness | Evaluation/leakage docs | Real models can be trained/evaluated consistently |
| 3. Linear Models, Properly | A | 12–18 | Multivariate linear regression, Ridge | Linear-model theory doc | Real vectorized linear model works in pipeline |
| 4. Linear Classification Models | A | 18–28 | Logistic regression, softmax regression, BCE, categorical CE, classification metrics | Logistic/softmax theory doc | Real vectorized binary and multiclass classifiers work in pipeline |
| 5. Optimization for ML | A | 14–22 | Batch/SGD/mini-batch/momentum framework | Optimization theory doc | Shared optimizer logic works across models |
| 6. Trees and Ensembles | B | 16–24 | Simple recursive Decision Tree | Trees/ensembles theory doc | Real tree builder works and ensembles are conceptually covered |
| 7. Distance and Kernel Thinking | B | 8–14 | Upgraded k-NN experiments | Distance/kernel theory doc | Distance, margins, and kernels are conceptually integrated |
| 8. Unsupervised Learning | A | 14–22 | k-means, PCA | Unsupervised-learning theory doc | Real unsupervised implementations and experiments exist |
| 9. Probabilistic ML Essentials | B | 10–16 | One lightweight probabilistic model or demo | Probabilistic-ML theory doc | Likelihood-based thinking is clearly integrated |
| 10. Bridge to Deep Learning | A | 14–22 | Tiny perceptron/MLP bridge | DL-bridge theory doc | Backprop and DL transition are conceptually natural |
| 11. Wrap-Up and Transition | A | 5–8 | Final summaries and DL roadmap entry | Positioning/wrap-up docs | ML Core closes cleanly and DL can start |