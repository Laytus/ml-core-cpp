# LogisticRegression Usage

## What the model does

`LogisticRegression` is a binary classification model.

It learns a linear decision function and converts that score into a probability using the sigmoid function:

```txt
score = X * weights + bias
probability_class_1 = sigmoid(score)
```

The model predicts class `1` when the predicted probability is greater than or equal to a chosen threshold, usually:

```txt
threshold = 0.5
```

In practical terms, logistic regression is useful when you want:

```txt
a binary classifier
probability estimates
an interpretable linear baseline
decision scores / logits
a simple model for imbalanced classification analysis
```

## Supported task type

`LogisticRegression` supports:

```txt
binary classification
```

It does not support multiclass classification directly. For multiclass linear classification, use:

```txt
SoftmaxRegression
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

The model should reject labels outside `{0, 1}`.

## Preprocessing usually needed

Logistic regression is sensitive to feature scale when trained with gradient descent.

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
#include "ml/linear_models/logistic_regression.hpp"

ml::LogisticRegressionOptions options;
options.learning_rate = 0.01;
options.max_iterations = 1000;
options.tolerance = 1e-8;
options.store_loss_history = true;

ml::LogisticRegression model(options);
```

Default constructor is also available:

```cpp
ml::LogisticRegression model;
```

## Regularized configuration

`LogisticRegression` supports regularization through `RegularizationConfig`.

Ridge-style regularization:

```cpp
#include "ml/linear_models/logistic_regression.hpp"
#include "ml/linear_models/regularization.hpp"

ml::LogisticRegressionOptions options;
options.learning_rate = 0.01;
options.max_iterations = 1000;
options.tolerance = 1e-8;
options.regularization = ml::RegularizationConfig::ridge(0.01);
options.store_loss_history = true;

ml::LogisticRegression model(options);
```

Use regularization when you want to penalize large weights and reduce sensitivity to noisy or correlated features.

## How to call `fit`

```cpp
model.fit(X_train, y_train);
```

Expected behavior:

```txt
- validates non-empty matrix/vector inputs
- validates matching number of rows
- validates binary target labels
- initializes weights and bias
- trains using gradient descent
- stores loss history if enabled
- marks the model as fitted
```

The target vector must contain binary labels encoded as `0` or `1`.

## How to call `logits`

```cpp
ml::Vector scores = model.logits(X_test);
```

Expected output:

```txt
scores.size() == X_test.rows()
```

Each score is the raw linear value before sigmoid conversion:

```txt
score = w^T x + b
```

Interpretation:

```txt
score > 0:
  model leans toward class 1

score < 0:
  model leans toward class 0

larger absolute score:
  stronger confidence before sigmoid thresholding
```

Calling `logits` before `fit` should be rejected.

## How to call `predict_proba`

```cpp
ml::Vector probabilities = model.predict_proba(X_test);
```

Expected output:

```txt
probabilities.size() == X_test.rows()
```

Each value is:

```txt
P(y = 1 | x)
```

Values are in:

```txt
[0, 1]
```

For binary classification workflows, this is exported as:

```txt
probability_class_1
```

and:

```txt
probability_class_0 = 1 - probability_class_1
```

## How to call `predict_classes`

Default threshold:

```cpp
ml::Vector predicted_classes = model.predict_classes(X_test);
```

Custom threshold:

```cpp
ml::Vector predicted_classes = model.predict_classes(X_test, 0.3);
```

Expected output:

```txt
predicted_classes.size() == X_test.rows()
predicted_classes contains 0/1 labels
```

Threshold interpretation:

```txt
lower threshold:
  predicts more positives
  may increase recall
  may reduce precision

higher threshold:
  predicts fewer positives
  may increase precision
  may reduce recall
```

## How to compute binary cross-entropy

```cpp
double loss = model.binary_cross_entropy(X_test, y_test);
```

This evaluates the model probability outputs against the binary targets.

Lower binary cross-entropy is better.

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
positive weight:
  increasing the feature increases the log-odds of class 1

negative weight:
  increasing the feature decreases the log-odds of class 1

larger absolute weight:
  stronger linear influence, assuming features are similarly scaled
```

Weights are easiest to interpret when features are standardized.

## How to inspect training history

If `store_loss_history` is enabled:

```cpp
const auto& history = model.training_history();

const std::vector<double>& losses = history.losses;
std::size_t iterations = history.iterations_run;
bool converged = history.converged;
```

This is useful for checking whether the binary cross-entropy decreased over time.

## How to read outputs

In Phase 11 practical workflows, `LogisticRegression` writes results through the binary classification comparison workflow.

Main output files:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/probabilities.csv
outputs/practical-exercises/binary-classification/decision_scores.csv
outputs/practical-exercises/binary-classification/loss_history.csv
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
```

### `metrics.csv`

Contains rows such as:

```txt
run_id,workflow,dataset,model,split,metric,value
logistic_regression_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,accuracy,...
logistic_regression_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,precision,...
logistic_regression_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,recall,...
logistic_regression_baseline,binary_classification,nasa_kc1_software_defects,LogisticRegression,test,f1,...
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
run_id,row_id,workflow,dataset,model,split,y_true,probability_class_0,probability_class_1
```

For logistic regression:

```txt
probability_class_1 = sigmoid(logit)
probability_class_0 = 1 - probability_class_1
```

### `decision_scores.csv`

Contains:

```txt
run_id,row_id,workflow,dataset,model,split,y_true,decision_score
```

For logistic regression:

```txt
decision_score = logit = w^T x + b
```

### `loss_history.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,iteration,loss
```

This is used by notebooks to plot training loss.

### `hyperparameter_sweep.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

For logistic regression sweeps, parameters include:

```txt
learning_rate
ridge_lambda
```

## Practical workflow example

`LogisticRegression` is used in the Phase 11 binary classification workflow:

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
probability distributions
probability distributions by true class
decision-score distributions
loss history
```

The hyperparameter sweep notebook is:

```txt
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

## Common mistakes to avoid

### 1. Using labels other than 0 and 1

`LogisticRegression` expects binary labels encoded as:

```txt
0
1
```

Do not use:

```txt
-1
+1
```

For `-1/+1` margin-based logic, use `LinearSVM`.

### 2. Interpreting accuracy alone on imbalanced data

On imbalanced datasets, a model can achieve high accuracy by mostly predicting the majority class.

Always inspect:

```txt
precision
recall
f1
```

### 3. Forgetting that probabilities depend on thresholding

`predict_proba` returns probabilities.

`predict_classes` applies a threshold.

Changing the threshold can strongly change precision and recall.

### 4. Using unscaled features with gradient descent

Feature scaling affects optimization behavior. Standardize numeric features before training.

### 5. Interpreting weights without scaling

Feature weights are easier to compare when inputs are standardized.

### 6. Calling prediction methods before `fit`

The model must be trained before calling:

```txt
logits
predict_proba
predict_classes
binary_cross_entropy
weights
bias
training_history
```

### 7. Treating logistic regression as a nonlinear classifier

Logistic regression learns a linear decision boundary in the input feature space. Nonlinear behavior requires engineered nonlinear features or a different model family.

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating this model are:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/probabilities.csv
outputs/practical-exercises/binary-classification/decision_scores.csv
outputs/practical-exercises/binary-classification/loss_history.csv
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
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

Use `LogisticRegression` when:

```txt
the task is binary classification
you want probability estimates
you want an interpretable linear baseline
you want decision scores / logits
you want a fast model for tabular data
you want to study threshold trade-offs
```

Avoid relying on it alone when:

```txt
the class boundary is strongly nonlinear
feature interactions are important but not engineered
the dataset is highly imbalanced and threshold/class weighting is not handled
probability calibration is critical and has not been evaluated
the task is multiclass classification
```
