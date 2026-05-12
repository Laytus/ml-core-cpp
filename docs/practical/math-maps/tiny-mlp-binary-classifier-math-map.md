# TinyMLPBinaryClassifier Math Map

## Purpose

This document maps the public `TinyMLPBinaryClassifier` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/dl_bridge/mlp.hpp
src/dl_bridge/mlp.cpp
include/ml/dl_bridge/activation.hpp
src/dl_bridge/activation.cpp
include/ml/common/classification_metrics.hpp
```

Related practical usage doc:

```txt
docs/practical/models/tiny-mlp-binary-classifier-usage.md
```

Related theory doc:

```txt
docs/theory/dl-bridge.md
```

## Model idea

`TinyMLPBinaryClassifier` is a small neural-network-style binary classifier.

It is intentionally minimal and educational.

The architecture is:

```txt
input features
    ↓
linear hidden layer
    ↓
ReLU activation
    ↓
linear output layer
    ↓
sigmoid activation
    ↓
binary probability
```

For a matrix of samples `X`, the forward computation is:

```txt
Z1 = XW1 + b1
A1 = ReLU(Z1)
Z2 = A1W2 + b2
A2 = sigmoid(Z2)
```

where:

```txt
X  = input feature matrix
W1 = hidden-layer weight matrix
b1 = hidden-layer bias vector
Z1 = hidden-layer pre-activation
A1 = hidden-layer activation
W2 = output-layer weight matrix
b2 = output-layer scalar bias
Z2 = output-layer pre-activation
A2 = predicted probability of class 1
```

For binary classification:

```txt
A2_i = P(y_i = 1 | x_i)
```

The predicted class is:

```txt
y_pred_i = 1 if A2_i >= 0.5 else 0
```

## Training objective

The training objective is binary cross-entropy:

```txt
BCE = -(1 / n) * sum_i [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
```

where:

```txt
p_i = A2_i
```

The model trains with mini-batch gradient descent and manual backpropagation.

For each mini-batch:

```txt
1. run forward pass
2. compute binary cross-entropy signal
3. backpropagate gradients through sigmoid, linear output layer, ReLU, and linear hidden layer
4. update W1, b1, W2, b2
```

## Public API to math mapping

## `TinyMLPBinaryClassifierOptions`

```cpp
struct TinyMLPBinaryClassifierOptions {
    std::size_t hidden_units;
    double learning_rate;
    std::size_t max_epochs;
    std::size_t batch_size;
    unsigned int random_seed;
};
```

### Mathematical role

`TinyMLPBinaryClassifierOptions` controls model capacity and optimization.

Important fields:

```txt
hidden_units
learning_rate
max_epochs
batch_size
random_seed
```

Math meaning:

```txt
hidden_units:
  number of neurons in the hidden layer

learning_rate:
  gradient descent step size

max_epochs:
  number of full training passes over the dataset

batch_size:
  number of samples used per mini-batch update

random_seed:
  controls parameter initialization and mini-batch shuffling
```

The number of hidden units controls the shapes:

```txt
W1: num_features x hidden_units
b1: hidden_units
W2: hidden_units x 1
b2: scalar
```

## `validate_tiny_mlp_binary_classifier_options`

```cpp
void validate_tiny_mlp_binary_classifier_options(
    const TinyMLPBinaryClassifierOptions& options,
    const std::string& context
);
```

### Mathematical role

Ensures the neural-network configuration is meaningful.

Typical requirements:

```txt
hidden_units >= 1
learning_rate > 0
max_epochs >= 1
batch_size >= 1
```

Invalid values would make the architecture or optimization impossible.

Examples:

```txt
hidden_units = 0:
  no hidden representation exists

learning_rate <= 0:
  gradient descent cannot move in a useful minimizing direction

batch_size = 0:
  no samples are available for a training update
```

### Infrastructure role

It also provides:

```txt
consistent error messages
context-specific validation
early failure for invalid configuration
```

## `TinyMLPForwardCache`

```cpp
struct TinyMLPForwardCache {
    Matrix X;
    Matrix Z1;
    Matrix A1;
    Matrix Z2;
    Matrix A2;
};
```

### Mathematical role

`TinyMLPForwardCache` stores intermediate values from the forward pass.

These values are needed for backpropagation.

Stored values:

```txt
X:
  input mini-batch

Z1:
  hidden pre-activation

A1:
  hidden activation after ReLU

Z2:
  output pre-activation

A2:
  output probability after sigmoid
```

Backpropagation needs these because gradients are computed layer by layer.

For example:

```txt
ReLU gradient depends on Z1
output-layer gradients depend on A1
binary cross-entropy output signal depends on A2 and y
```

## Constructor

```cpp
TinyMLPBinaryClassifier();
explicit TinyMLPBinaryClassifier(
    TinyMLPBinaryClassifierOptions options
);
```

### Mathematical role

The constructor does not train the network.

It only stores the architecture and optimization configuration.

Math implemented:

```txt
none directly
```

Infrastructure role:

```txt
configure hidden layer size
configure learning rate
configure epoch count
configure batch size
configure random seed
validate options
```

## `fit`

```cpp
void fit(const Matrix& X, const Vector& y);
```

### Mathematical role

`fit` is the core training method.

It estimates:

```txt
W1
b1
W2
b2
```

by minimizing binary cross-entropy through mini-batch gradient descent.

The training process is:

```txt
1. initialize W1, b1, W2, b2
2. for each epoch:
   - shuffle training indices
   - split data into mini-batches
   - for each mini-batch:
       run forward pass
       compute backpropagation gradients
       update parameters
   - compute full training loss
   - append loss to history
```

Forward pass for a mini-batch:

```txt
Z1 = X_batch W1 + b1
A1 = ReLU(Z1)
Z2 = A1 W2 + b2
A2 = sigmoid(Z2)
```

Binary cross-entropy:

```txt
BCE = -(1 / m) * sum_i [y_i log(A2_i) + (1 - y_i) log(1 - A2_i)]
```

where:

```txt
m = mini-batch size
```

For sigmoid output with binary cross-entropy, the output-layer gradient signal simplifies to:

```txt
dZ2 = A2 - y
```

Then:

```txt
dW2 = (1 / m) * A1^T dZ2
db2 = mean(dZ2)
```

Backpropagate into hidden layer:

```txt
dA1 = dZ2 W2^T
dZ1 = dA1 * ReLU'(Z1)
```

where:

```txt
ReLU'(z) = 1 if z > 0 else 0
```

Then:

```txt
dW1 = (1 / m) * X_batch^T dZ1
db1 = column_mean(dZ1)
```

Parameter updates:

```txt
W1 := W1 - learning_rate * dW1
b1 := b1 - learning_rate * db1
W2 := W2 - learning_rate * dW2
b2 := b2 - learning_rate * db2
```

### What `fit` does mathematically

`fit` implements:

```txt
parameter initialization
mini-batch forward propagation
binary cross-entropy computation
manual backpropagation
gradient descent parameter updates
epoch-level loss tracking
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
option validation
input validation
binary target validation
finite-value validation
shape checking
mini-batch index shuffling
batch slicing
random-seed management
fitted-state tracking
loss-history storage
```

These are necessary for correctness but are not the neural-network equations themselves.

## `predict_proba`

```cpp
Vector predict_proba(const Matrix& X) const;
```

### Mathematical role

`predict_proba` returns the network output probability for class `1`.

Math implemented:

```txt
A2 = sigmoid(A1W2 + b2)
```

where:

```txt
A1 = ReLU(XW1 + b1)
```

Expected output:

```txt
probabilities.size() == X.rows()
```

Each value is:

```txt
P(y = 1 | x)
```

The binary workflow exports:

```txt
probability_class_1 = A2
probability_class_0 = 1 - A2
```

### Infrastructure role

`predict_proba` also handles:

```txt
calling forward
extracting A2 as a vector
fitted-state validation through forward
input validation through forward
```

## `predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` thresholds predicted probabilities into binary class labels.

Math implemented:

```txt
p = predict_proba(X)
y_pred_i = 1 if p_i >= 0.5 else 0
```

Expected output:

```txt
predicted_classes.size() == X.rows()
```

### Infrastructure role

`predict` also handles:

```txt
probability computation
thresholding at 0.5
output vector allocation
```

## `forward`

```cpp
TinyMLPForwardCache forward(const Matrix& X) const;
```

### Mathematical role

`forward` computes and returns all intermediate values in the network.

Math implemented:

```txt
Z1 = XW1 + b1
A1 = ReLU(Z1)
Z2 = A1W2 + b2
A2 = sigmoid(Z2)
```

The returned cache contains:

```txt
X
Z1
A1
Z2
A2
```

### Why this matters

The forward pass is the direct prediction computation.

The same intermediate values are also needed to compute gradients in backpropagation.

### Infrastructure role

`forward` also handles:

```txt
fitted-state validation
input validation
feature-count validation
finite-value validation
matrix allocation
```

## `is_fitted`

```cpp
bool is_fitted() const;
```

### Mathematical role

None directly.

This is state-management infrastructure.

It answers whether the network has initialized and trained parameters:

```txt
W1
b1
W2
b2
```

available for prediction.

## `options`

```cpp
const TinyMLPBinaryClassifierOptions& options() const;
```

### Mathematical role

Returns the architecture and optimization configuration.

This configuration defines:

```txt
hidden layer width
learning rate
number of epochs
batch size
random initialization seed
```

## `W1`

```cpp
const Matrix& W1() const;
```

### Mathematical role

Returns the hidden-layer weight matrix.

Expected shape:

```txt
W1.rows() == num_features
W1.cols() == hidden_units
```

Math role:

```txt
Z1 = XW1 + b1
```

Each column of `W1` defines the input weights for one hidden unit.

### Infrastructure role

Also checks fitted state.

## `b1`

```cpp
const Vector& b1() const;
```

### Mathematical role

Returns the hidden-layer bias vector.

Expected shape:

```txt
b1.size() == hidden_units
```

Math role:

```txt
Z1 = XW1 + b1
```

where `b1` is broadcast across rows.

### Infrastructure role

Also checks fitted state.

## `W2`

```cpp
const Matrix& W2() const;
```

### Mathematical role

Returns the output-layer weight matrix.

Expected shape:

```txt
W2.rows() == hidden_units
W2.cols() == 1
```

Math role:

```txt
Z2 = A1W2 + b2
```

Each row of `W2` weights one hidden unit's contribution to the output logit.

### Infrastructure role

Also checks fitted state.

## `b2`

```cpp
double b2() const;
```

### Mathematical role

Returns the output-layer scalar bias.

Math role:

```txt
Z2 = A1W2 + b2
```

The output bias shifts the final logit before sigmoid.

### Infrastructure role

Also checks fitted state.

## `loss_history`

```cpp
const std::vector<double> loss_history() const;
```

### Mathematical role

Returns the training loss recorded across epochs.

Each value is usually:

```txt
binary cross-entropy over the training set after an epoch
```

This is useful for checking whether training is improving.

Expected behavior:

```txt
loss should generally decrease if the learning rate and architecture are reasonable
```

### Infrastructure role

Stores diagnostic information for plotting and practical workflow exports.

## `num_features`

```cpp
Eigen::Index num_features() const;
```

### Mathematical role

Returns the number of input features used during fitting.

This controls:

```txt
W1 row count
valid input shape for forward/predict
```

### Infrastructure role

Supports feature-count validation during prediction.

## Important internal math concepts

## Hidden linear layer

The hidden pre-activation is:

```txt
Z1 = XW1 + b1
```

This is the same linear-algebra pattern used in linear regression and logistic regression, but applied to multiple hidden units at once.

For one hidden unit `h`:

```txt
Z1_ih = dot(x_i, W1_col_h) + b1_h
```

## ReLU activation

ReLU is:

```txt
ReLU(z) = max(0, z)
```

Applied element-wise:

```txt
A1 = ReLU(Z1)
```

ReLU introduces nonlinearity.

Without the activation, two linear layers would collapse into one linear model.

Derivative:

```txt
ReLU'(z) = 1 if z > 0 else 0
```

This derivative controls which hidden units receive gradient during backpropagation.

## Output linear layer

The output pre-activation is:

```txt
Z2 = A1W2 + b2
```

This combines the hidden representation into one scalar logit per sample.

## Sigmoid activation

The sigmoid maps the output logit to a probability:

```txt
sigmoid(z) = 1 / (1 + exp(-z))
```

Output:

```txt
A2 = sigmoid(Z2)
```

Interpretation:

```txt
A2_i = P(y_i = 1 | x_i)
```

## Binary cross-entropy

Binary cross-entropy measures probability prediction error:

```txt
BCE = -(1 / n) * sum_i [y_i log(A2_i) + (1 - y_i) log(1 - A2_i)]
```

Effects:

```txt
confident correct prediction:
  low loss

uncertain prediction:
  moderate loss

confident wrong prediction:
  high loss
```

## Backpropagation

Backpropagation applies the chain rule from output to input.

For this network:

```txt
BCE
 ↓
sigmoid output
 ↓
output linear layer
 ↓
ReLU activation
 ↓
hidden linear layer
```

The simplified output gradient for sigmoid + BCE is:

```txt
dZ2 = A2 - y
```

Then gradients propagate backward:

```txt
dW2 = A1^T dZ2 / m
db2 = mean(dZ2)

dA1 = dZ2 W2^T
dZ1 = dA1 * ReLU'(Z1)

dW1 = X^T dZ1 / m
db1 = column_mean(dZ1)
```

## Mini-batch gradient descent

The model updates parameters using mini-batches.

Instead of using the full dataset for every update, each update uses:

```txt
batch_size samples
```

This creates noisier but often more efficient updates.

For each parameter `theta`:

```txt
theta := theta - learning_rate * gradient(theta)
```

where:

```txt
theta ∈ {W1, b1, W2, b2}
```

## Parameter initialization

The model initializes weights before training.

Initialization matters because:

```txt
if all hidden units start identically:
  they may learn the same features

random initialization:
  breaks symmetry between hidden units
```

The random seed makes initialization reproducible.

## Nonlinear decision boundary

Because of the hidden ReLU layer, the model can learn nonlinear boundaries.

The structure is:

```txt
x -> ReLU(linear transform) -> sigmoid(linear transform)
```

This is more expressive than logistic regression, which uses:

```txt
x -> sigmoid(single linear transform)
```

However, this tiny MLP is still small and educational, not a production deep-learning framework.

## Relationship to LogisticRegression

Both models output:

```txt
P(y = 1 | x)
```

using sigmoid.

Logistic regression:

```txt
z = Xw + b
p = sigmoid(z)
```

Tiny MLP:

```txt
Z1 = XW1 + b1
A1 = ReLU(Z1)
Z2 = A1W2 + b2
p = sigmoid(Z2)
```

The MLP adds a learned nonlinear hidden representation before the sigmoid output.

## Method classification

| Method / Struct | Forward math | Backprop math | Optimization math | Probability/loss math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `TinyMLPBinaryClassifierOptions` | Partial | No | Yes | No | Yes |
| `validate_tiny_mlp_binary_classifier_options` | Partial | No | Partial | No | Yes |
| `TinyMLPForwardCache` | Yes | Yes | No | No | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Yes | Yes | Yes | Yes | Yes |
| `predict_proba` | Yes | No | No | Yes | Yes |
| `predict` | Yes | No | No | Yes | Yes |
| `forward` | Yes | No | No | Yes | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `options` | Partial | No | Partial | No | Yes |
| `W1` | Yes | No | No | No | Yes |
| `b1` | Yes | No | No | No | Yes |
| `W2` | Yes | No | No | No | Yes |
| `b2` | Yes | No | No | No | Yes |
| `loss_history` | No | No | Diagnostic | Diagnostic | Yes |
| `num_features` | Partial | No | No | No | Yes |

## Output files and math meaning

## `metrics.csv`

Relevant rows:

```txt
model = TinyMLPBinaryClassifier
metric = accuracy
metric = precision
metric = recall
metric = f1
```

Math meaning:

```txt
accuracy:
  fraction of correct thresholded predictions

precision:
  true positives / predicted positives

recall:
  true positives / actual positives

f1:
  harmonic mean of precision and recall
```

For imbalanced datasets, recall and F1 are especially important.

## `predictions.csv`

Relevant columns:

```txt
y_true
y_pred
correct
```

Math meaning:

```txt
y_true:
  actual binary label

y_pred:
  thresholded MLP prediction

correct:
  1 if y_pred == y_true else 0
```

## `probabilities.csv`

Relevant columns:

```txt
probability_class_0
probability_class_1
```

Math meaning:

```txt
probability_class_1:
  A2 = sigmoid(A1W2 + b2)

probability_class_0:
  1 - A2
```

## `loss_history.csv`

Relevant columns:

```txt
iteration
loss
```

Math meaning:

```txt
iteration:
  training epoch index

loss:
  binary cross-entropy after that epoch
```

This is used to inspect optimization behavior.

## `hyperparameter_sweep.csv`

Relevant rows:

```txt
model = TinyMLPBinaryClassifier
param_name = hidden_units
param_name = learning_rate
param_name = max_epochs
metric = accuracy
metric = precision
metric = recall
metric = f1
```

Math meaning:

```txt
hidden_units:
  hidden-layer width / model capacity

learning_rate:
  gradient descent update scale

max_epochs:
  number of training passes

metric value:
  classification performance produced by that configuration
```

## Why TinyMLPBinaryClassifier does not export decision scores

The model internally computes an output logit:

```txt
Z2
```

but the public practical workflow focuses on probability output:

```txt
A2 = sigmoid(Z2)
```

Therefore, the main confidence output is:

```txt
probability_class_1
```

A future extension could export `Z2` as a decision score, but this is not part of the current workflow.

## Practical interpretation

`TinyMLPBinaryClassifier` is useful as a bridge from classical ML to neural networks.

It demonstrates:

```txt
layered computation
activation functions
binary cross-entropy
mini-batch training
manual backpropagation
nonlinear hidden representations
```

If it performs well, the hidden layer may be capturing useful nonlinear structure.

If it performs poorly, possible causes include:

```txt
features are not scaled
learning rate is poor
too few or too many hidden units
too few epochs
class imbalance
small/noisy dataset
lack of class weighting or threshold tuning
```

In the Phase 11 NASA KC1 workflow, TinyMLP can show high accuracy while still having very low recall. This illustrates why binary classification evaluation must include precision, recall, and F1, not accuracy alone.

## Educational scope

This model is not intended to become a full deep-learning framework.

It is designed to make the transition to Deep Learning natural by showing how previous concepts connect:

```txt
linear models:
  matrix multiplications

logistic regression:
  sigmoid probability + BCE

optimization:
  gradient descent

calculus:
  chain rule

DL:
  layered differentiable computation + backpropagation
```

## Summary

`TinyMLPBinaryClassifier` maps to the following mathematical pipeline:

```txt
data matrix X
binary target vector y
        ↓
initialize W1, b1, W2, b2
        ↓
forward pass:
  Z1 = XW1 + b1
  A1 = ReLU(Z1)
  Z2 = A1W2 + b2
  A2 = sigmoid(Z2)
        ↓
binary cross-entropy loss
        ↓
backpropagation:
  compute gradients for W2, b2, W1, b1
        ↓
mini-batch gradient descent updates
        ↓
learned tiny neural classifier
        ↓
probabilities, thresholded predictions, and binary metrics
```

The core math lives in:

```txt
fit
forward
predict_proba
predict
W1
b1
W2
b2
loss_history
```

The supporting infrastructure lives in:

```txt
options
validate_tiny_mlp_binary_classifier_options
is_fitted
num_features
batch selection helpers
input validation
random-seed management
output export workflows
```
