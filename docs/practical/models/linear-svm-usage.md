# LinearSVM Usage

## What the model does

`LinearSVM` is a binary margin-based classification model.

It learns a linear decision boundary using the hinge-loss objective and L2 regularization.

Conceptually, the model computes a decision score:

```txt
score = X * weights + bias
```

and predicts the positive class when the score is non-negative:

```txt
prediction = 1 if score >= 0 else 0
```

Internally, binary labels are mapped from:

```txt
0, 1
```

to:

```txt
-1, +1
```

so the hinge-loss margin condition can be written as:

```txt
y_svm * score >= 1
```

In practical terms, `LinearSVM` is useful when you want:

```txt
a binary classifier
a linear margin-based baseline
decision scores
regularized separating hyperplanes
a comparison point against LogisticRegression
```

## Supported task type

`LinearSVM` supports:

```txt
binary classification
```

It does not support:

```txt
regression
multiclass classification directly
probability prediction
clustering
dimensionality reduction
```

For probability estimates, use:

```txt
LogisticRegression
GaussianNaiveBayes
RandomForestClassifier
TinyMLPBinaryClassifier
```

For multiclass classification, use:

```txt
SoftmaxRegression
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
```

where:

```txt
X.rows() == y.size()
X.cols() == number of numeric features
y contains binary labels encoded as 0 or 1
```

Typical shapes:

```txt
X: n_samples x n_features
y: n_samples
```

Example:

```cpp
ml::Matrix X(4, 2);
X << 0.0, 0.0,
     0.0, 1.0,
     1.0, 0.0,
     1.0, 1.0;

ml::Vector y(4);
y << 0.0, 0.0, 1.0, 1.0;
```

For real dataset workflows, use the CSV loader to create `X` and `y` from a processed CSV file.

## Target encoding

Targets must be encoded as:

```txt
0 = negative class
1 = positive class
```

For the Phase 11 NASA KC1 workflow:

```txt
0 = no defect
1 = defect
```

The model maps these labels internally to:

```txt
0 -> -1
1 -> +1
```

This mapping is needed for the hinge-loss margin objective.

## Preprocessing usually needed

`LinearSVM` is sensitive to feature scale because the margin and regularization depend on the geometry of the feature space.

Recommended preprocessing:

```txt
standardize numeric features
reject missing values or preprocess them before loading
avoid non-numeric columns
ensure binary target encoding is 0/1
use train-fitted preprocessing statistics only
```

For practical workflows in this project, feature standardization is handled in the workflow layer before training.

## How to instantiate the model

Basic model:

```cpp
#include "ml/linear_models/linear_svm.hpp"

ml::LinearSVMOptions options;
options.learning_rate = 0.01;
options.max_epochs = 100;
options.l2_lambda = 0.01;

ml::LinearSVM model(options);
```

Default constructor is also available:

```cpp
ml::LinearSVM model;
```

## Main options

`LinearSVMOptions` contains:

```txt
learning_rate
max_epochs
l2_lambda
```

Interpretation:

```txt
learning_rate:
  step size used during training

max_epochs:
  number of training passes

l2_lambda:
  strength of L2 regularization
```

Higher `l2_lambda` penalizes large weights more strongly.

## How to call `fit`

```cpp
model.fit(X_train, y_train);
```

Expected behavior:

```txt
- validates options
- validates non-empty matrix/vector inputs
- validates matching number of rows
- validates binary target labels encoded as 0/1
- maps labels internally to -1/+1
- trains using hinge-loss margin logic with L2 regularization
- stores training loss history
- marks the model as fitted
```

The target vector must contain binary labels encoded as `0` or `1`.

## How to call `decision_function`

```cpp
ml::Vector scores = model.decision_function(X_test);
```

Expected output:

```txt
scores.size() == X_test.rows()
```

Each score is the raw margin score:

```txt
score = w^T x + b
```

Interpretation:

```txt
score >= 0:
  model predicts class 1

score < 0:
  model predicts class 0

larger positive score:
  stronger positive-class margin

larger negative score:
  stronger negative-class margin

score close to 0:
  sample is close to the learned decision boundary
```

Calling `decision_function` before `fit` should be rejected.

## How to call `predict`

```cpp
ml::Vector predicted_classes = model.predict(X_test);
```

Expected output:

```txt
predicted_classes.size() == X_test.rows()
predicted_classes contains 0/1 labels
```

Prediction rule:

```txt
prediction = 1 if decision_score >= 0 else 0
```

## Probability output

The current `LinearSVM` API does not expose:

```txt
predict_proba
```

This is expected. A linear SVM is a margin-based classifier, not a probabilistic model.

If probability estimates are needed, use:

```txt
LogisticRegression
GaussianNaiveBayes
RandomForestClassifier
TinyMLPBinaryClassifier
```

or add a later calibration layer.

## How to evaluate predictions

Use binary classification metrics:

```cpp
#include "ml/common/classification_metrics.hpp"

double accuracy = ml::accuracy_score(predicted_classes, y_test);
double precision = ml::precision_score(predicted_classes, y_test);
double recall = ml::recall_score(predicted_classes, y_test);
double f1 = ml::f1_score(predicted_classes, y_test);
```

Interpretation:

```txt
accuracy:
  fraction of correct predictions

precision:
  among predicted positives, how many were truly positive

recall:
  among true positives, how many were detected

f1:
  balance between precision and recall
```

For imbalanced datasets, accuracy can be misleading. Recall and F1 are often more informative.

## How to inspect learned parameters

After fitting:

```cpp
const ml::Vector& weights = model.weights();
double bias = model.bias();
```

Interpretation:

```txt
weights:
  normal vector of the linear decision boundary

bias:
  intercept term

decision boundary:
  w^T x + b = 0
```

Weights are easiest to interpret when features are standardized.

## How to inspect training loss history

After fitting:

```cpp
const std::vector<double>& losses = model.training_loss_history();
```

The loss history stores the objective value over training.

It is useful for checking whether training is stable and whether loss decreases over epochs.

## How to inspect model state

```cpp
bool fitted = model.is_fitted();
const ml::LinearSVMOptions& used_options = model.options();
```

These are useful for verifying that the model was trained and which configuration was used.

## How to read outputs

In Phase 11 practical workflows, `LinearSVM` writes results through the binary classification comparison workflow.

Main output files:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/decision_scores.csv
outputs/practical-exercises/binary-classification/loss_history.csv
```

`LinearSVM` does not write probability rows because the model does not expose `predict_proba`.

### `metrics.csv`

Contains rows such as:

```txt
run_id,workflow,dataset,model,split,metric,value
linear_svm_baseline,binary_classification,nasa_kc1_software_defects,LinearSVM,test,accuracy,...
linear_svm_baseline,binary_classification,nasa_kc1_software_defects,LinearSVM,test,precision,...
linear_svm_baseline,binary_classification,nasa_kc1_software_defects,LinearSVM,test,recall,...
linear_svm_baseline,binary_classification,nasa_kc1_software_defects,LinearSVM,test,f1,...
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

### `decision_scores.csv`

Contains:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,decision_score
```

For `LinearSVM`:

```txt
decision_score = w^T x + b
```

### `loss_history.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,iteration,loss
```

This is used by notebooks to plot training loss.

## Practical workflow example

`LinearSVM` is used in the Phase 11 binary classification workflow:

```txt
include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp
```

The corresponding notebook is:

```txt
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
```

The notebook visualizes:

```txt
metric comparison
prediction correctness
decision-score distributions
decision scores by true class
loss history
```

## Common mistakes to avoid

### 1. Using labels other than 0 and 1

The public API expects binary labels encoded as:

```txt
0
1
```

Do not pass labels as:

```txt
-1
+1
```

The model performs the `0/1` to `-1/+1` mapping internally.

### 2. Expecting probabilities

`LinearSVM` exposes decision scores, not probabilities.

A high positive score means a strong class-1 margin, but it is not the same thing as:

```txt
P(y = 1 | x)
```

### 3. Interpreting accuracy alone on imbalanced data

On imbalanced datasets, a model can achieve high accuracy by mostly predicting the majority class.

Always inspect:

```txt
precision
recall
f1
```

### 4. Using unscaled features

Margins and distances depend on feature scale.

Standardize numeric features before training.

### 5. Choosing `l2_lambda` without validation

Regularization strength changes the learned margin and weight magnitude.

Compare different values experimentally.

### 6. Calling prediction methods before `fit`

The model must be trained before calling:

```txt
decision_function
predict
weights
bias
training_loss_history
```

### 7. Treating linear SVM as a nonlinear kernel SVM

This implementation is a primal linear SVM.

It does not implement:

```txt
kernel SVM
dual optimization
SMO
nonlinear kernel decision functions
```

Kernel SVM was explicitly deferred in this project.

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating this model are:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/decision_scores.csv
outputs/practical-exercises/binary-classification/loss_history.csv
```

The main visualization notebook is:

```txt
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
```

The main workflow implementation is:

```txt
include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp
```

## When to use this model

Use `LinearSVM` when:

```txt
the task is binary classification
you want a linear margin-based classifier
you want decision scores rather than probabilities
you want a comparison point against LogisticRegression
you want to study margin behavior
features are numeric and standardized
```

Avoid relying on it alone when:

```txt
you need calibrated probabilities
the class boundary is strongly nonlinear
the dataset is highly imbalanced and threshold/margin behavior has not been evaluated
feature scaling is not controlled
a kernel SVM is required
the task is multiclass classification
```
