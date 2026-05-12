# SoftmaxRegression Usage

## What the model does

`SoftmaxRegression` is a multiclass linear classification model.

It generalizes binary logistic regression to more than two classes.

The model computes one linear score, or logit, per class:

```txt
logits = X * weights + bias
```

Then it converts those logits into class probabilities using softmax:

```txt
probability_class_j = exp(logit_j) / sum_k exp(logit_k)
```

The predicted class is the class with the highest probability.

In practical terms, `SoftmaxRegression` is useful when you want:

```txt
a multiclass classifier
probability estimates for each class
an interpretable linear multiclass baseline
a direct bridge from logistic regression to neural-network output layers
```

## Supported task type

`SoftmaxRegression` supports:

```txt
multiclass classification
```

It is not used for regression, binary-only margin classification, clustering, or dimensionality reduction.

For binary classification, use:

```txt
LogisticRegression
LinearSVM
TinyMLPBinaryClassifier
```

For multiclass non-linear tabular baselines, compare with:

```txt
KNNClassifier
DecisionTreeClassifier
RandomForestClassifier
GaussianNaiveBayes
```

## Expected input format

The model expects:

```cpp
Matrix X;
Vector y;
Eigen::Index num_classes;
```

where:

```txt
X.rows() == y.size()
X.cols() == number of numeric features
y contains integer class labels encoded as 0, 1, ..., num_classes - 1
num_classes >= 2
```

Typical shapes:

```txt
X: n_samples x n_features
y: n_samples
weights: n_features x num_classes
bias: num_classes
```

Example:

```cpp
ml::Matrix X(5, 2);
X << 1.0, 2.0,
     1.5, 1.8,
     5.0, 8.0,
     6.0, 9.0,
     9.0, 1.0;

ml::Vector y(5);
y << 0.0, 0.0, 1.0, 1.0, 2.0;

Eigen::Index num_classes = 3;
```

For real dataset workflows, use the CSV loader to create `X` and `y` from a processed CSV file.

## Target encoding

Targets must be encoded as integer-valued labels:

```txt
0
1
2
...
num_classes - 1
```

For the Phase 11 Wine workflow, the original UCI Wine labels are converted from:

```txt
1, 2, 3
```

to:

```txt
0, 1, 2
```

This convention keeps the class labels compatible with matrix column indexing.

## Preprocessing usually needed

Softmax regression is sensitive to feature scale when trained with gradient descent.

Recommended preprocessing:

```txt
standardize numeric features
reject missing values or preprocess them before loading
avoid non-numeric columns
ensure class labels are 0-based integer values
use train-fitted preprocessing statistics only
```

For practical workflows in this project, feature standardization is handled in the workflow layer before training.

## How to instantiate the model

Basic model:

```cpp
#include "ml/linear_models/softmax_regression.hpp"

ml::SoftmaxRegressionOptions options;
options.learning_rate = 0.01;
options.max_iterations = 1000;
options.tolerance = 1e-8;
options.store_loss_history = true;

ml::SoftmaxRegression model(options);
```

Default constructor is also available:

```cpp
ml::SoftmaxRegression model;
```

## Regularized configuration

`SoftmaxRegression` supports regularization through `RegularizationConfig`.

Ridge-style regularization:

```cpp
#include "ml/linear_models/softmax_regression.hpp"
#include "ml/linear_models/regularization.hpp"

ml::SoftmaxRegressionOptions options;
options.learning_rate = 0.01;
options.max_iterations = 1000;
options.tolerance = 1e-8;
options.regularization = ml::RegularizationConfig::ridge(0.01);
options.store_loss_history = true;

ml::SoftmaxRegression model(options);
```

Use regularization when you want to penalize large weights and reduce sensitivity to noisy or correlated features.

## How to call `fit`

```cpp
model.fit(X_train, y_train, num_classes);
```

Expected behavior:

```txt
- validates non-empty matrix/vector inputs
- validates matching number of rows
- validates class labels are in [0, num_classes)
- initializes class weight matrix and bias vector
- trains using gradient descent
- stores loss history if enabled
- marks the model as fitted
```

The target vector must contain 0-based integer class labels.

## How to call `logits`

```cpp
ml::Matrix scores = model.logits(X_test);
```

Expected output shape:

```txt
scores.rows() == X_test.rows()
scores.cols() == num_classes
```

Each row contains one raw score per class.

Interpretation:

```txt
higher logit for a class:
  stronger linear evidence for that class before softmax

logits are not probabilities:
  they can be negative, positive, or any real value
```

Calling `logits` before `fit` should be rejected.

## How to call `predict_proba`

```cpp
ml::Matrix probabilities = model.predict_proba(X_test);
```

Expected output shape:

```txt
probabilities.rows() == X_test.rows()
probabilities.cols() == num_classes
```

Each row contains class probabilities:

```txt
probability_class_0
probability_class_1
...
probability_class_k
```

Each probability row should sum approximately to:

```txt
1.0
```

## How to call `predict_classes`

```cpp
ml::Vector predicted_classes = model.predict_classes(X_test);
```

Expected output:

```txt
predicted_classes.size() == X_test.rows()
predicted_classes contains labels in [0, num_classes)
```

Each prediction is the index of the largest predicted probability.

## How to compute categorical cross-entropy

```cpp
double loss = model.categorical_cross_entropy(X_test, y_test);
```

This evaluates the predicted probability distribution against the true class labels.

Lower categorical cross-entropy is better.

## How to evaluate predictions

Use multiclass classification metrics:

```cpp
#include "ml/common/multiclass_metrics.hpp"

double accuracy = ml::multiclass_accuracy_score(
    predicted_classes,
    y_test,
    num_classes
);

double macro_precision = ml::macro_precision(
    predicted_classes,
    y_test,
    num_classes
);

double macro_recall = ml::macro_recall(
    predicted_classes,
    y_test,
    num_classes
);

double macro_f1 = ml::macro_f1(
    predicted_classes,
    y_test,
    num_classes
);
```

Interpretation:

```txt
accuracy:
  fraction of correct predictions

macro_precision:
  average precision across classes

macro_recall:
  average recall across classes

macro_f1:
  average F1 score across classes
```

Macro metrics are useful because they evaluate all classes more evenly than raw accuracy.

## How to inspect learned parameters

After fitting:

```cpp
const ml::Matrix& weights = model.weights();
const ml::Vector& bias = model.bias();
```

Expected shapes:

```txt
weights.rows() == number of features
weights.cols() == num_classes
bias.size() == num_classes
```

Interpretation:

```txt
weights.col(j):
  linear feature weights for class j

bias(j):
  intercept term for class j
```

Weights are easiest to interpret when features are standardized.

## How to inspect the number of classes

```cpp
Eigen::Index class_count = model.num_classes();
```

This returns the number of classes used during training.

## How to inspect training history

If `store_loss_history` is enabled:

```cpp
const auto& history = model.training_history();

const std::vector<double>& losses = history.losses;
std::size_t iterations = history.iterations_run;
bool converged = history.converged;
```

This is useful for checking whether categorical cross-entropy decreased over time.

## How to read outputs

In Phase 11 practical workflows, `SoftmaxRegression` writes results through the multiclass classification comparison workflow.

Main output files:

```txt
outputs/practical-exercises/multiclass-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/predictions.csv
outputs/practical-exercises/multiclass-classification/probabilities.csv
outputs/practical-exercises/multiclass-classification/loss_history.csv
```

### `metrics.csv`

Contains rows such as:

```txt
run_id,workflow,dataset,model,split,metric,value
softmax_regression_baseline,multiclass_classification,wine,SoftmaxRegression,test,accuracy,...
softmax_regression_baseline,multiclass_classification,wine,SoftmaxRegression,test,macro_precision,...
softmax_regression_baseline,multiclass_classification,wine,SoftmaxRegression,test,macro_recall,...
softmax_regression_baseline,multiclass_classification,wine,SoftmaxRegression,test,macro_f1,...
```

### `predictions.csv`

Contains:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,y_pred,correct
```

where:

```txt
correct = 1 if y_pred == y_true else 0
```

### `probabilities.csv`

Contains:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1,probability_class_2
```

The number of probability columns depends on the number of classes.

For the Wine workflow:

```txt
num_classes = 3
```

### `loss_history.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,iteration,loss
```

This is used by notebooks to plot training loss.

## Practical workflow example

`SoftmaxRegression` is used in the Phase 11 multiclass classification workflow:

```txt
include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp
```

The corresponding notebook is:

```txt
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
```

The notebook visualizes:

```txt
metric comparison
prediction correctness
confusion-matrix-style tables
confusion-matrix-style plots
probability distributions
probability assigned to the true class
loss history
```

## Common mistakes to avoid

### 1. Using one-based class labels

`SoftmaxRegression` expects labels in:

```txt
0, 1, ..., num_classes - 1
```

Do not pass labels like:

```txt
1, 2, 3
```

unless they have been converted first.

### 2. Confusing logits with probabilities

`logits` are raw scores.

`predict_proba` converts logits into probabilities using softmax.

Use `predict_proba` when you need class probabilities.

### 3. Forgetting that each probability row should sum to one

For each sample, softmax produces a probability distribution across classes.

If probabilities do not sum close to one, that indicates a numerical or implementation problem.

### 4. Interpreting accuracy alone

Accuracy can hide class-level problems.

Use macro precision, macro recall, and macro F1 to check class-balanced behavior.

### 5. Using unscaled features with gradient descent

Feature scaling affects optimization behavior. Standardize numeric features before training.

### 6. Interpreting weights without scaling

Feature weights are easier to compare when inputs are standardized.

### 7. Calling prediction methods before `fit`

The model must be trained before calling:

```txt
logits
predict_proba
predict_classes
categorical_cross_entropy
weights
bias
training_history
```

### 8. Treating softmax regression as a nonlinear classifier

Softmax regression learns linear class boundaries in the input feature space.

Nonlinear behavior requires engineered nonlinear features or a different model family.

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating this model are:

```txt
outputs/practical-exercises/multiclass-classification/metrics.csv
outputs/practical-exercises/multiclass-classification/predictions.csv
outputs/practical-exercises/multiclass-classification/probabilities.csv
outputs/practical-exercises/multiclass-classification/loss_history.csv
```

The main visualization notebook is:

```txt
notebooks/practical-workflows/03_multiclass_classification_outputs.ipynb
```

The main workflow implementation is:

```txt
include/ml/workflows/multiclass_classification_comparison.hpp
src/workflows/multiclass_classification_comparison.cpp
```

## When to use this model

Use `SoftmaxRegression` when:

```txt
the task is multiclass classification
you want probability estimates for each class
you want an interpretable linear multiclass baseline
you want a simple model for standardized numeric tabular data
you want to connect logistic regression to neural-network output layers
```

Avoid relying on it alone when:

```txt
the class boundary is strongly nonlinear
feature interactions are important but not engineered
classes are highly overlapping
the dataset requires complex non-linear decision regions
probability calibration is critical and has not been evaluated
```
