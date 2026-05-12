# Experiment Inventory – ML Core

## Purpose

This document is the final experiment inventory for `ml-core-cpp`.

It summarizes the experiment folders, output folders, concepts demonstrated, and the role each experiment plays in the project.

Experiments in this repo are not meant to be production benchmarks. They are sanity tests, behavior studies, model comparisons, and practical workflow checks that validate the implementation and connect code to theory.

---

## Experiment types

The project uses four main experiment types:

| Type | Meaning |
|---|---|
| Sanity test | Verifies that an implementation runs and respects expected behavior |
| Behavior study | Shows how a model or method changes under different parameters |
| Comparison workflow | Compares models or configurations under shared metrics |
| Practical workflow | Uses real datasets and exports results for Python/Jupyter analysis |

---

## Main test runner

Central runner:

```txt
app/test_runner.cpp
```

The test runner integrates the phase sanity checks into a single executable:

```txt
./build/ml_core_tests
```

Phase 11 also uses Python verification scripts:

```txt
python3 scripts/verify_practical_outputs.py
python3 scripts/summarize_hyperparameter_sweeps.py
```

---

## Phase 1 – Math and statistical foundations

Expected experiment folder:

```txt
experiments/phase-1-math-sanity/
```

Expected output folder:

```txt
outputs/phase-1-math-sanity/
```

Main concept demonstrated:

```txt
Eigen vector/matrix conventions
basic math helper behavior
statistical helper behavior
shape validation discipline
```

Experiment role:

```txt
sanity test
```

Why it matters:

```txt
Phase 1 establishes the math and validation conventions used by the rest of the project.
```

---

## Phase 2 – Data pipeline and evaluation methodology

Expected experiment folder:

```txt
experiments/phase-2-evaluation-methodology/
```

Expected output folder:

```txt
outputs/phase-2-evaluation-methodology/
```

Main concept demonstrated:

```txt
dataset abstractions
train/test split
train/validation/test split
cross-validation
preprocessing fitted only on training data
baseline-vs-model evaluation
```

Experiment role:

```txt
sanity test
behavior study
```

Why it matters:

```txt
It verifies the project can evaluate models consistently without data leakage.
```

---

## Phase 3 – Linear models

Experiment folder:

```txt
experiments/phase-3-linear-models/
```

Output folder:

```txt
outputs/phase-3-linear-models/
```

Main concept demonstrated:

```txt
multivariate LinearRegression
vectorized prediction
MSE optimization
Ridge regularization behavior
scaled vs unscaled optimization
learning-rate behavior
```

Experiment role:

```txt
sanity test
behavior study
comparison workflow
```

Representative outputs:

```txt
linear regression metric summaries
loss histories
regularization comparisons
```

Why it matters:

```txt
This phase validates the first serious vectorized supervised model in the repo.
```

---

## Phase 4 – Linear classification models

Experiment folder:

```txt
experiments/phase-4-logistic-regression/
```

Output folder:

```txt
outputs/phase-4-logistic-regression/
```

Main concept demonstrated:

```txt
LogisticRegression
SoftmaxRegression
binary cross-entropy
categorical cross-entropy
probability prediction
threshold behavior
binary and multiclass metrics
```

Experiment role:

```txt
sanity test
behavior study
comparison workflow
```

Representative outputs:

```txt
threshold comparison outputs
classification metric summaries
probability tables
loss histories
multiclass prediction outputs
```

Why it matters:

```txt
This phase validates the core probability-based classification pattern that later connects directly to neural-network output layers.
```

---

## Phase 5 – Optimization for ML

Experiment folder:

```txt
experiments/phase-5-optimization/
```

Output folder:

```txt
outputs/phase-5-optimization/
```

Main concept demonstrated:

```txt
batch gradient descent
SGD
mini-batch gradient descent
momentum
learning-rate sensitivity
scaled vs unscaled optimization
optimizer history logging
```

Experiment role:

```txt
behavior study
comparison workflow
```

Representative outputs:

```txt
optimizer comparison CSVs
training history outputs
learning-rate behavior summaries
momentum comparison summaries
```

Why it matters:

```txt
It confirms that optimization behavior can be studied independently from a single model implementation.
```

---

## Phase 6 – Trees and ensembles foundation

Experiment folder:

```txt
experiments/phase-6-trees-ensembles/
```

Output folder:

```txt
outputs/phase-6-trees-ensembles/
```

Main concept demonstrated:

```txt
split scoring
Gini impurity
entropy
weighted child impurity
candidate-threshold evaluation
recursive tree building
DecisionTreeClassifier behavior
tree depth and stopping controls
```

Experiment role:

```txt
sanity test
behavior study
```

Representative outputs:

```txt
tree behavior summaries
split-scoring checks
depth/stopping comparisons
```

Why it matters:

```txt
This phase validates the base decision tree infrastructure that later ensembles reuse.
```

---

## Phase 6B – Tree ensembles and advanced tree features

Experiment folder:

```txt
experiments/phase-6b-tree-ensembles/
```

Output folder:

```txt
outputs/phase-6b-tree-ensembles/
```

Main concept demonstrated:

```txt
advanced tree options
max_leaf_nodes
max_features
class/sample weighting behavior
missing-value rejection
bootstrap sampling
RandomForestClassifier
GradientBoostingRegressor
single tree vs ensemble behavior
```

Experiment role:

```txt
sanity test
behavior study
comparison workflow
```

Representative outputs:

```txt
single-tree vs random-forest comparisons
gradient-boosting comparisons
bootstrap behavior outputs
advanced tree control summaries
```

Important deferred items:

```txt
learned missing-value routing
full cost-complexity pruning workflow
full out-of-bag scoring workflow
GradientBoostingClassifier
```

Why it matters:

```txt
This phase turns the base tree implementation into a reusable tree-learning family.
```

---

## Phase 7 – Distance and kernel thinking

Experiment folder:

```txt
experiments/phase-7-distance-kernel/
```

Output folder:

```txt
outputs/phase-7-distance-kernel/
```

Main concept demonstrated:

```txt
Euclidean distance
squared Euclidean distance
Manhattan distance
KNNClassifier
different k values
kernel similarity functions
SVM margin intuition
LinearSVM
hinge-loss behavior
decision scores
```

Experiment role:

```txt
sanity test
behavior study
comparison workflow
```

Representative outputs:

```txt
outputs/phase-7-distance-kernel/kernel_similarity_demo.csv
outputs/phase-7-distance-kernel/svm_margin_intuition.txt
outputs/phase-7-distance-kernel/linear_svm_comparison.csv
outputs/phase-7-distance-kernel/linear_svm_comparison.txt
outputs/phase-7-distance-kernel/linear_svm_margin_behavior.csv
```

Deferred item:

```txt
full kernel SVM / SMO / dual optimization
```

Why it matters:

```txt
This phase validates geometric ML intuition: neighborhoods, distances, margins, and kernels.
```

---

## Phase 8 – Unsupervised learning

Experiment folder:

```txt
experiments/phase-8-unsupervised/
```

Output folder:

```txt
outputs/phase-8-unsupervised/
```

Main concept demonstrated:

```txt
KMeans
cluster assignment
centroid updates
inertia
PCA
explained variance
dimensionality reduction
reconstruction intuition
```

Experiment role:

```txt
sanity test
behavior study
```

Representative outputs:

```txt
KMeans inertia comparisons
PCA explained variance outputs
projection outputs
reconstruction summaries
```

Why it matters:

```txt
This phase validates representation learning and structure-discovery intuition before DL.
```

---

## Phase 9 – Probabilistic ML essentials

Experiment folder:

```txt
experiments/phase-9-probabilistic-ml/
```

Output folder:

```txt
outputs/phase-9-probabilistic-ml/
```

Main concept demonstrated:

```txt
GaussianNaiveBayes
class priors
Gaussian likelihoods
posterior probabilities
log probabilities
variance smoothing
probabilistic assumptions
```

Experiment role:

```txt
sanity test
behavior study
```

Representative outputs:

```txt
outputs/phase-9-probabilistic-ml/gaussian_naive_bayes_probability_table.csv
outputs/phase-9-probabilistic-ml/gaussian_naive_bayes_prior_comparison.txt
outputs/phase-9-probabilistic-ml/probabilistic_model_summary.txt
```

Why it matters:

```txt
This phase adds a probability-centered interpretation layer, not only deterministic prediction.
```

---

## Phase 10 – Bridge to Deep Learning

Experiment folder:

```txt
experiments/phase-10-dl-bridge/
```

Output folder:

```txt
outputs/phase-10-dl-bridge/
```

Main concept demonstrated:

```txt
Perceptron
TinyMLPBinaryClassifier
activation functions
forward propagation
manual backpropagation
binary cross-entropy
mini-batch training
classical ML to DL transition
```

Experiment role:

```txt
sanity test
behavior study
bridge demo
```

Representative outputs:

```txt
perceptron behavior outputs
tiny MLP classification demo outputs
loss history outputs
DL bridge summaries
```

Why it matters:

```txt
This phase makes Deep Learning feel like a natural extension of linear models, optimization, and chain rule.
```

---

## Phase 11 – Practical ML workflows with real datasets

Experiment folder:

```txt
experiments/phase-11-practical-workflows/
```

Output folders:

```txt
outputs/practical-exercises/regression/
outputs/practical-exercises/binary-classification/
outputs/practical-exercises/multiclass-classification/
outputs/practical-exercises/unsupervised/
```

Main concept demonstrated:

```txt
real dataset loading
dataset-to-Matrix/Vector conversion
model comparison workflows
structured CSV outputs
hyperparameter sweeps
Python/Pandas verification
Jupyter visualization
practical model documentation
math-map documentation
```

Experiment role:

```txt
practical workflow
comparison workflow
behavior study
final validation layer
```

## Phase 11.1 – Practical workflow runner sanity tests

Main test file:

```txt
experiments/phase-11-practical-workflows/phase11_practical_workflows_sanity.cpp
```

Main concept demonstrated:

```txt
workflow runners load real datasets
OutputWriter writes expected CSV files
outputs are generated in stable schemas
```

Representative checks:

```txt
regression workflow loads stock OHLCV dataset
binary workflow loads NASA KC1 dataset
multiclass workflow loads Wine dataset
unsupervised workflow loads stock OHLCV dataset
OutputWriter writes metrics/predictions/probabilities/decision scores/loss histories
```

## Phase 11.2 – Model comparison workflows

Regression output folder:

```txt
outputs/practical-exercises/regression/
```

Main files:

```txt
metrics.csv
predictions.csv
loss_history.csv
```

Models compared:

```txt
LinearRegression
RidgeRegression
DecisionTreeRegressor
GradientBoostingRegressor
```

Binary output folder:

```txt
outputs/practical-exercises/binary-classification/
```

Main files:

```txt
metrics.csv
predictions.csv
probabilities.csv
decision_scores.csv
loss_history.csv
```

Models compared:

```txt
LogisticRegression
LinearSVM
GaussianNaiveBayes
DecisionTreeClassifier
RandomForestClassifier
TinyMLPBinaryClassifier
```

Multiclass output folder:

```txt
outputs/practical-exercises/multiclass-classification/
```

Main files:

```txt
metrics.csv
predictions.csv
probabilities.csv
loss_history.csv
```

Models compared:

```txt
SoftmaxRegression
KNNClassifier
GaussianNaiveBayes
DecisionTreeClassifier
RandomForestClassifier
```

Unsupervised output folder:

```txt
outputs/practical-exercises/unsupervised/
```

Main files:

```txt
metrics.csv
projections.csv
clustering_assignments.csv
```

Methods compared:

```txt
PCA
KMeans
PCA+KMeans
```

## Phase 11.3 – Hyperparameter sweeps

Main files:

```txt
outputs/practical-exercises/regression/hyperparameter_sweep.csv
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
outputs/practical-exercises/multiclass-classification/hyperparameter_sweep.csv
outputs/practical-exercises/unsupervised/hyperparameter_sweep.csv
```

Sweeps covered:

```txt
GradientBoostingRegressor: n_estimators, learning_rate
LogisticRegression: learning_rate, ridge_lambda
DecisionTreeClassifier: max_depth, min_samples_leaf
RandomForestClassifier: n_estimators, max_features
TinyMLPBinaryClassifier: hidden_units, learning_rate, max_epochs
KNNClassifier: k
PCA: num_components
KMeans: num_clusters
```

Summary script:

```txt
scripts/summarize_hyperparameter_sweeps.py
```

## Phase 11 Python/Jupyter notebooks

Notebook folder:

```txt
notebooks/practical-workflows/
```

Implemented notebooks:

```txt
01_regression_outputs.ipynb
02_binary_classification_outputs.ipynb
03_multiclass_classification_outputs.ipynb
04_unsupervised_outputs.ipynb
05_hyperparameter_sweeps.ipynb
```

Notebook role:

```txt
read exported CSV files
visualize metrics
visualize predictions/errors
visualize probability distributions
visualize PCA/KMeans outputs
visualize hyperparameter sweep behavior
```

## Phase 11 verification scripts

Scripts:

```txt
scripts/verify_practical_outputs.py
scripts/summarize_hyperparameter_sweeps.py
```

`verify_practical_outputs.py` validates that all major practical CSV outputs are readable from Pandas.

`summarize_hyperparameter_sweeps.py` summarizes best runs by metric and helps ground interpretation docs in real outputs.

---

## Practical outputs that should not be committed

The output folder structure can be committed with `.gitkeep`, but generated CSV outputs should usually stay ignored unless there is a specific representative artifact worth preserving.

Recommended ignored generated outputs:

```txt
outputs/practical-exercises/regression/*.csv
outputs/practical-exercises/binary-classification/*.csv
outputs/practical-exercises/multiclass-classification/*.csv
outputs/practical-exercises/unsupervised/*.csv
```

Recommended committed structure:

```txt
outputs/practical-exercises/regression/.gitkeep
outputs/practical-exercises/binary-classification/.gitkeep
outputs/practical-exercises/multiclass-classification/.gitkeep
outputs/practical-exercises/unsupervised/.gitkeep
```

---

## Final experiment inventory conclusion

The experiment layer now validates the project at three levels:

```txt
1. implementation correctness through sanity tests
2. conceptual behavior through controlled experiments
3. practical usability through real-dataset workflows
```

The most important final validation is Phase 11.

It proves that `ml-core-cpp` is not only a set of isolated algorithms. It can load real datasets, train multiple models, export structured results, and support Python/Jupyter analysis.

This is enough to close ML Core and move to the Deep Learning project.
