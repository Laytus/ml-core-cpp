# TinyMLPBinaryClassifier Usage

## What the model does

`TinyMLPBinaryClassifier` is a small neural-network-style binary classifier.

It is part of the `dl_bridge` module and exists to connect the classical ML Core project to Deep Learning concepts.

The model uses a simple multilayer perceptron architecture:

```txt
input features
  -> hidden linear layer
  -> ReLU activation
  -> output linear layer
  -> sigmoid activation
  -> binary probability
```

Conceptually:

```txt
Z1 = X * W1 + b1
A1 = ReLU(Z1)
Z2 = A1 * W2 + b2
A2 = sigmoid(Z2)
```

where:

```txt
A2 = P(y = 1 | x)
```

The model predicts class `1` when the output probability is greater than or equal to `0.5`.

In practical terms, `TinyMLPBinaryClassifier` is useful when you want:

```txt
a minimal neural-network bridge
a binary classifier with nonlinear hidden features
probability outputs
manual forward/backpropagation intuition
a comparison point against LogisticRegression and LinearSVM
```

## Supported task type

`TinyMLPBinaryClassifier` supports:

```txt
binary classification
```

It does not support:

```txt
regression
multiclass classification
clustering
dimensionality reduction
decision scores / margins
```

For linear binary classification, use:

```txt
LogisticRegression
LinearSVM
```

For multiclass classification, use:

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
y << 0.0, 1.0, 1.0, 0.0;
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

`TinyMLPBinaryClassifier` is trained with gradient-based optimization, so it is sensitive to feature scale.

Recommended preprocessing:

```txt
standardize numeric features
reject missing values or preprocess them before loading
avoid non-numeric feature columns
ensure binary target encoding is 0/1
use train-fitted preprocessing statistics only
```

For practical workflows in this project, feature standardization is handled in the workflow layer before training.

## How to instantiate the model

Basic model:

```cpp
#include "ml/dl_bridge/mlp.hpp"

ml::TinyMLPBinaryClassifierOptions options;
options.hidden_units = 8;
options.learning_rate = 0.05;
options.max_epochs = 100;
options.batch_size = 16;
options.random_seed = 42;

ml::TinyMLPBinaryClassifier model(options);
```

Default constructor is also available:

```cpp
ml::TinyMLPBinaryClassifier model;
```

## Main options

`TinyMLPBinaryClassifierOptions` contains:

```txt
hidden_units
learning_rate
max_epochs
batch_size
random_seed
```

### `hidden_units`

Number of neurons in the hidden layer.

Interpretation:

```txt
more hidden units:
  higher model capacity
  can capture more nonlinear structure
  higher overfitting risk
  higher compute cost

fewer hidden units:
  simpler model
  lower capacity
  may underfit
```

### `learning_rate`

Step size used during training.

Interpretation:

```txt
larger learning_rate:
  faster updates
  can become unstable

smaller learning_rate:
  slower updates
  may require more epochs
```

### `max_epochs`

Number of training passes over the dataset.

Interpretation:

```txt
more epochs:
  more training
  lower training loss possible
  higher overfitting risk

fewer epochs:
  faster
  may underfit
```

### `batch_size`

Number of samples used per mini-batch update.

Interpretation:

```txt
smaller batch_size:
  noisier updates
  can help exploration
  more update steps

larger batch_size:
  smoother updates
  fewer update steps
```

### `random_seed`

Controls deterministic initialization and mini-batch shuffling.

Use a fixed seed when comparing experiments.

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
- initializes W1, b1, W2, b2
- trains using mini-batch gradient descent
- applies forward propagation
- applies manual backpropagation
- stores loss history
- marks the model as fitted
```

The target vector must contain binary labels encoded as `0` or `1`.

## How to call `forward`

```cpp
ml::TinyMLPForwardCache cache = model.forward(X_test);
```

The forward cache contains:

```txt
X
Z1
A1
Z2
A2
```

Interpretation:

```txt
Z1:
  hidden-layer pre-activation

A1:
  hidden-layer activation after ReLU

Z2:
  output-layer pre-activation

A2:
  output probability after sigmoid
```

This method is useful for inspecting the neural-network computation.

Calling `forward` before `fit` should be rejected.

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
prediction = 1 if probability >= 0.5 else 0
```

## How to inspect learned parameters

After fitting:

```cpp
const ml::Matrix& W1 = model.W1();
const ml::Vector& b1 = model.b1();
const ml::Matrix& W2 = model.W2();
double b2 = model.b2();
```

Expected shapes:

```txt
W1.rows() == number of input features
W1.cols() == hidden_units

b1.size() == hidden_units

W2.rows() == hidden_units
W2.cols() == 1

b2 is scalar
```

Interpretation:

```txt
W1, b1:
  hidden-layer parameters

W2, b2:
  output-layer parameters
```

Unlike linear regression or logistic regression, these parameters are harder to interpret directly because the model contains a nonlinear hidden layer.

## How to inspect training loss history

After fitting:

```cpp
const std::vector<double> losses = model.loss_history();
```

The loss history stores binary cross-entropy over epochs.

It is useful for checking:

```txt
whether training loss decreases
whether learning rate is stable
whether the model appears to underfit
whether training may be unstable
```

## How to inspect fitted state and feature count

```cpp
bool fitted = model.is_fitted();
Eigen::Index feature_count = model.num_features();
const ml::TinyMLPBinaryClassifierOptions& used_options = model.options();
```

These are useful for verifying that the model was trained and which configuration was used.

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

## How to read outputs

In Phase 11 practical workflows, `TinyMLPBinaryClassifier` writes results through the binary classification comparison and hyperparameter sweep workflows.

Main output files:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/probabilities.csv
outputs/practical-exercises/binary-classification/loss_history.csv
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
```

`TinyMLPBinaryClassifier` does not currently export:

```txt
decision_scores
```

because its main public confidence output is probability from the sigmoid output.

### `metrics.csv`

Contains rows such as:

```txt
run_id,workflow,dataset,model,split,metric,value
tiny_mlp_binary_classifier_baseline,binary_classification,nasa_kc1_software_defects,TinyMLPBinaryClassifier,test,accuracy,...
tiny_mlp_binary_classifier_baseline,binary_classification,nasa_kc1_software_defects,TinyMLPBinaryClassifier,test,precision,...
tiny_mlp_binary_classifier_baseline,binary_classification,nasa_kc1_software_defects,TinyMLPBinaryClassifier,test,recall,...
tiny_mlp_binary_classifier_baseline,binary_classification,nasa_kc1_software_defects,TinyMLPBinaryClassifier,test,f1,...
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

For `TinyMLPBinaryClassifier`:

```txt
probability_class_1 = sigmoid(output_logit)
probability_class_0 = 1 - probability_class_1
```

### `loss_history.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,iteration,loss
```

For `TinyMLPBinaryClassifier`, each iteration corresponds to a training epoch in the exported workflow.

### `hyperparameter_sweep.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

For TinyMLP sweeps, parameters include:

```txt
hidden_units
learning_rate
max_epochs
```

Example:

```txt
tiny_mlp_h8_lr0.05_e50,binary_classification,nasa_kc1_software_defects,TinyMLPBinaryClassifier,test,hidden_units,8,f1,...
tiny_mlp_h8_lr0.05_e50,binary_classification,nasa_kc1_software_defects,TinyMLPBinaryClassifier,test,learning_rate,0.05,f1,...
tiny_mlp_h8_lr0.05_e50,binary_classification,nasa_kc1_software_defects,TinyMLPBinaryClassifier,test,max_epochs,50,f1,...
```

## Practical workflow example

`TinyMLPBinaryClassifier` is used in the Phase 11 binary classification workflow:

```txt
include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp
```

It is also used in the hyperparameter sweep workflow:

```txt
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

The corresponding notebooks are:

```txt
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The notebooks visualize:

```txt
metric comparison
prediction correctness
probability distributions
probability distributions by true class
loss history
parameter-vs-metric sweep behavior
```

## Hyperparameter behavior

Important hyperparameters:

```txt
hidden_units
learning_rate
max_epochs
batch_size
```

Common behavior:

```txt
more hidden_units:
  more capacity, but more overfitting risk

larger learning_rate:
  faster training, but possible instability

more epochs:
  more training, but possible overfitting

batch_size:
  controls update noise and training dynamics
```

In the Phase 11 NASA KC1 sweep, the best tested TinyMLP configuration was:

```txt
run_id: tiny_mlp_h8_lr0.05_e50
hidden_units: 8
learning_rate: 0.05
max_epochs: 50
```

with:

```txt
accuracy: 0.866666666667
precision: 1.0
recall: 0.0588235294118
f1: 0.111111111111
```

This result illustrates an important lesson:

```txt
high accuracy and high precision can still hide very low recall
```

On an imbalanced dataset, the model may predict very few positive samples and still appear accurate.

## Common mistakes to avoid

### 1. Interpreting high accuracy alone

On imbalanced datasets, a model can achieve high accuracy by mostly predicting the majority class.

Always inspect:

```txt
precision
recall
f1
```

### 2. Assuming a neural model is automatically better

`TinyMLPBinaryClassifier` is intentionally small and educational.

It is not guaranteed to outperform classical models such as:

```txt
LogisticRegression
RandomForestClassifier
GaussianNaiveBayes
```

### 3. Using unscaled features

Neural-network training is sensitive to feature scale.

Standardize numeric features before training.

### 4. Using too high a learning rate

A high learning rate can make training unstable or prevent convergence.

Inspect the loss history.

### 5. Using too few epochs

Too few epochs can underfit.

Compare training loss and validation/test metrics.

### 6. Using too many hidden units without validation

More hidden units can increase capacity, but may overfit or behave poorly on small datasets.

### 7. Treating probabilities as calibrated by default

The sigmoid output is probability-like, but it is not guaranteed to be calibrated.

Calibration should be evaluated separately if probability quality matters.

### 8. Calling prediction methods before `fit`

The model must be trained before calling:

```txt
forward
predict_proba
predict
W1
b1
W2
b2
loss_history
num_features
```

### 9. Forgetting this is a bridge model

This model exists to connect ML Core to Deep Learning.

It is useful for learning forward propagation, activation functions, binary cross-entropy, mini-batch training, and backpropagation.

It is not intended to become a full deep-learning framework.

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating `TinyMLPBinaryClassifier` are:

```txt
outputs/practical-exercises/binary-classification/metrics.csv
outputs/practical-exercises/binary-classification/predictions.csv
outputs/practical-exercises/binary-classification/probabilities.csv
outputs/practical-exercises/binary-classification/loss_history.csv
outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv
```

The main visualization notebooks are:

```txt
notebooks/practical-workflows/02_binary_classification_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The main workflow implementations are:

```txt
include/ml/workflows/binary_classification_comparison.hpp
src/workflows/binary_classification_comparison.cpp
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

The model implementation files are:

```txt
include/ml/dl_bridge/mlp.hpp
src/dl_bridge/mlp.cpp
```

The related theory doc is:

```txt
docs/theory/dl-bridge.md
```

## When to use this model

Use `TinyMLPBinaryClassifier` when:

```txt
the task is binary classification
you want a minimal neural-network classifier
you want probability outputs
you want to study nonlinear hidden-layer behavior
you want to connect ML Core concepts to Deep Learning
you want to compare a tiny MLP against classical baselines
```

Avoid relying on it alone when:

```txt
you need a production neural-network framework
you need multiclass classification
you need deep architectures
the dataset is highly imbalanced and threshold/class weighting is not handled
you need calibrated probabilities
you have not inspected loss history and recall/F1
```
