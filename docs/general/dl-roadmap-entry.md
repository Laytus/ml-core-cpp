# Deep Learning Roadmap Entry

## Purpose

This document defines the handoff from `ml-core-cpp` to the next Deep Learning project.

`ml-core-cpp` is now complete as the classical ML foundation layer. The next project should not restart from zero. It should build directly on the concepts, workflows, and implementation discipline developed here.

The goal of the next project is to move from:

```txt
classical ML models implemented manually in C++
```

to:

```txt
modern Deep Learning workflows with tensors, autograd, neural architectures, and serious training loops
```

---

## Starting point from ML Core

The following concepts are already covered by `ml-core-cpp` and should be treated as prerequisites for the DL project.

### Mathematical foundations already covered

```txt
vectors and matrices
matrix multiplication
dot products
linear transformations
gradients
partial derivatives
chain rule
expectation
variance
covariance
bias / variance intuition
train / validation / test roles
```

These should not be relearned from zero.

The DL project should reuse them immediately.

### Model foundations already covered

```txt
LinearRegression
LogisticRegression
SoftmaxRegression
LinearSVM
KNNClassifier
DecisionTreeClassifier
DecisionTreeRegressor
RandomForestClassifier
GradientBoostingRegressor
GaussianNaiveBayes
PCA
KMeans
Perceptron
TinyMLPBinaryClassifier
```

The DL project does not need to expand the classical ML model inventory.

### Optimization foundations already covered

```txt
batch gradient descent
SGD
mini-batch gradient descent
momentum
learning-rate sensitivity
loss histories
scaled vs unscaled training
regularization intuition
```

The DL project should start by extending these ideas to neural-network training.

### Evaluation foundations already covered

```txt
train/test split
train/validation/test split
cross-validation
preprocessing discipline
data leakage prevention
baseline-vs-model comparison
regression metrics
binary classification metrics
multiclass metrics
probability outputs
decision scores
structured output exports
```

The DL project should keep this evaluation discipline.

### Practical workflow foundations already covered

```txt
real dataset loading
numeric CSV workflows
Matrix / Vector conversion
structured metrics exports
prediction/probability/loss exports
Python/Pandas verification
Jupyter visualization
hyperparameter sweep summaries
model usage docs
method-to-math mapping docs
```

The DL project should use the same style:

```txt
theory first
implementation second
experiments third
visualization fourth
interpretation last
```

---

## Main conceptual transition

The central transition is:

```txt
classical ML:
  manually implemented models and gradients

Deep Learning:
  layered differentiable computation with tensors and autograd
```

In ML Core, each model had its own explicit math:

```txt
LinearRegression:
  Xw + b

LogisticRegression:
  sigmoid(Xw + b)

SoftmaxRegression:
  softmax(XW + b)

TinyMLPBinaryClassifier:
  X -> Linear -> ReLU -> Linear -> Sigmoid
```

In Deep Learning, this generalizes to:

```txt
input tensor
  -> many differentiable layers
  -> loss function
  -> automatic backpropagation
  -> optimizer step
```

The next project should make that generalization explicit.

---

## What stays the same from ML to DL

The following ideas remain the same:

```txt
data must be split correctly
features/inputs must be transformed consistently
training loss is optimized on training data
validation metrics guide model selection
test metrics estimate final generalization
learning rate matters
batch size matters
initialization matters
overfitting matters
regularization matters
metrics must match the task
plots and diagnostics matter
```

Deep Learning adds more scale and more architectural complexity, but the evaluation discipline remains the same.

---

## What changes in DL

The following ideas become more important or new:

```txt
tensors instead of only matrices/vectors
automatic differentiation
computational graphs
deeper architectures
activation choices
weight initialization schemes
normalization layers
dropout
optimizers such as Adam
GPU/accelerator execution
data loaders
mini-batch pipelines
checkpointing
training loops
model serialization
larger datasets
experiment tracking
```

The DL project should focus on these new ideas rather than re-implementing classical ML.

---

## Recommended next project identity

Recommended repo identity:

```txt
dl-foundations
```

or:

```txt
deep-learning-foundations
```

If the implementation is Python/PyTorch-based, a good name would be:

```txt
dl-foundations-pytorch
```

Recommended goal:

```txt
Build a serious Deep Learning foundation project that starts from the concepts learned in ml-core-cpp and progresses toward modern neural-network workflows.
```

Recommended project positioning:

```txt
not a toy tutorial repo
not a Kaggle-only notebook collection
not a production MLOps platform
not a research framework

a structured DL learning and implementation project with theory, experiments, notebooks, and reusable training workflows
```

---

## Recommended technology choice

For the DL project, use a real DL framework instead of implementing everything manually.

Recommended:

```txt
Python
PyTorch
NumPy
Pandas
Matplotlib
Jupyter
```

Optional later:

```txt
TensorBoard
Weights & Biases or MLflow
TorchVision
Hugging Face Transformers
PyTorch Lightning
```

### Why PyTorch

PyTorch is recommended because it makes the transition from ML Core very clear:

```txt
Tensor:
  generalization of Matrix / Vector

nn.Module:
  reusable model abstraction

autograd:
  automatic backpropagation

optimizer:
  generalized gradient update logic

DataLoader:
  mini-batch data workflow

training loop:
  structured version of the optimization loops studied in ML Core
```

---

## What not to do in the DL project

Avoid starting with:

```txt
Transformers immediately
large language models immediately
diffusion models immediately
GPU optimization immediately
MLOps/deployment immediately
too many datasets at once
too many frameworks at once
```

The DL project should first build a stable foundation.

Do not skip:

```txt
tensors
autograd
training loops
MLP
overfitting experiments
regularization experiments
CNN basics
```

---

## Phase 0 – DL Project Setup and Learning Contract

**Goal:** Create the repo and define the DL project identity.

**Level:** A

**Estimated effort:** 4–8 hours

### Tasks

- [ ] Define repo structure
- [ ] Create Python environment
- [ ] Add dependencies
- [ ] Add README
- [ ] Add action plan
- [ ] Add theory docs folder
- [ ] Add experiments folder
- [ ] Add notebooks folder
- [ ] Add outputs folder
- [ ] Add project rules:
  - theory first
  - implementation second
  - experiment third
  - notebook visualization fourth
- [ ] Define what is reused conceptually from `ml-core-cpp`

### Expected folders

```txt
docs/general/
docs/theory/
experiments/
notebooks/
src/
outputs/
data/
scripts/
```

### Exit criteria

- [ ] Repo is ready
- [ ] PyTorch environment works
- [ ] First tensor sanity script runs
- [ ] Project identity is documented

---

## Phase 1 – Tensors, Autograd, and Computational Graphs

**Goal:** Understand the real replacement for manual gradients.

**Level:** A

**Estimated effort:** 8–14 hours

### Why this phase exists

In ML Core, gradients were implemented manually.

In DL, gradients are computed by autograd.

This phase should make autograd feel understandable rather than magical.

### Tasks

- [ ] Write theory for tensors
- [ ] Write theory for computational graphs
- [ ] Write theory for automatic differentiation
- [ ] Compare manual gradient from ML Core to PyTorch autograd
- [ ] Implement small tensor demos
- [ ] Implement scalar autograd demos
- [ ] Implement vector/matrix autograd demos
- [ ] Recreate a simple linear regression gradient example in PyTorch
- [ ] Show `.backward()`, `.grad`, and optimizer steps
- [ ] Document what autograd replaces and what it does not replace

### Key experiments

```txt
manual gradient vs autograd gradient
linear regression loss backward pass
gradient accumulation behavior
requires_grad behavior
detach / no_grad behavior
```

### Exit criteria

- [ ] You can explain autograd as chain rule over a computation graph
- [ ] You can inspect gradients
- [ ] You understand when gradients are tracked and when they are not

---

## Phase 2 – Neural Network Training Loop Fundamentals

**Goal:** Build the standard DL training loop from first principles.

**Level:** A

**Estimated effort:** 10–16 hours

### Tasks

- [ ] Write theory for `nn.Module`
- [ ] Write theory for parameters
- [ ] Write theory for loss functions
- [ ] Write theory for optimizers
- [ ] Write theory for train/eval modes
- [ ] Implement a simple linear model with `nn.Module`
- [ ] Implement a manual training loop:
  - forward
  - loss
  - zero gradients
  - backward
  - optimizer step
  - metric logging
- [ ] Add train/validation split
- [ ] Add loss-history plotting
- [ ] Add metric export
- [ ] Compare against equivalent ML Core concepts

### Key experiments

```txt
linear regression in PyTorch
binary logistic regression in PyTorch
softmax regression in PyTorch
manual loop vs helper function loop
```

### Exit criteria

- [ ] You can write a complete PyTorch training loop without copying blindly
- [ ] You can explain every line of the loop
- [ ] You understand train vs eval mode

---

## Phase 3 – MLPs for Tabular and Simple Classification

**Goal:** Move from linear models to real neural networks.

**Level:** A

**Estimated effort:** 12–20 hours

### Tasks

- [ ] Write theory for MLPs
- [ ] Write theory for hidden layers
- [ ] Write theory for activations
- [ ] Write theory for nonlinear decision boundaries
- [ ] Implement MLP for binary classification
- [ ] Implement MLP for multiclass classification
- [ ] Compare:
  - logistic regression vs MLP
  - softmax regression vs MLP
  - TinyMLPBinaryClassifier from ML Core vs PyTorch MLP
- [ ] Add experiments for:
  - hidden units
  - number of layers
  - learning rate
  - batch size
  - overfitting
- [ ] Add notebooks for visualization

### Key experiments

```txt
two-moons or synthetic nonlinear classification
MNIST-like simple dataset if appropriate
tabular binary classification
tabular multiclass classification
```

### Exit criteria

- [ ] You understand why hidden layers create nonlinear models
- [ ] You can train an MLP for binary and multiclass tasks
- [ ] You can diagnose underfitting and overfitting in an MLP

---

## Phase 4 – Regularization, Initialization, and Training Stability

**Goal:** Understand why neural networks train well or poorly.

**Level:** A

**Estimated effort:** 12–20 hours

### Tasks

- [ ] Write theory for initialization
- [ ] Write theory for vanishing/exploding gradients
- [ ] Write theory for dropout
- [ ] Write theory for weight decay
- [ ] Write theory for batch normalization
- [ ] Write theory for learning-rate schedules
- [ ] Compare optimizers:
  - SGD
  - SGD + momentum
  - Adam
- [ ] Add experiments for:
  - poor vs good initialization
  - no regularization vs weight decay
  - dropout effects
  - learning-rate schedules
  - batch normalization effects

### Key experiments

```txt
loss curves under different learning rates
SGD vs Adam
dropout vs no dropout
weight decay sweeps
initialization sensitivity
```

### Exit criteria

- [ ] You can diagnose unstable neural-network training
- [ ] You understand the most common regularization tools
- [ ] You can choose a reasonable optimizer and learning rate

---

## Phase 5 – CNN Foundations

**Goal:** Understand convolutional neural networks for images.

**Level:** A

**Estimated effort:** 16–28 hours

### Tasks

- [ ] Write theory for convolution
- [ ] Write theory for channels, kernels, padding, stride
- [ ] Write theory for pooling
- [ ] Write theory for feature maps
- [ ] Implement simple CNN in PyTorch
- [ ] Train CNN on MNIST or Fashion-MNIST
- [ ] Compare MLP vs CNN on images
- [ ] Visualize filters or feature maps if useful
- [ ] Add notebook for image classification outputs

### Key experiments

```txt
MLP vs CNN on MNIST/Fashion-MNIST
kernel size comparison
depth comparison
pooling comparison
feature map visualization
```

### Exit criteria

- [ ] You understand why CNNs are better than MLPs for image data
- [ ] You can train and evaluate a simple CNN
- [ ] You can explain feature maps, kernels, stride, padding, and pooling

---

## Phase 6 – Transfer Learning and Practical Image Models

**Goal:** Learn practical DL workflows using pretrained models.

**Level:** A

**Estimated effort:** 12–20 hours

### Tasks

- [ ] Write theory for transfer learning
- [ ] Write theory for frozen backbones vs fine-tuning
- [ ] Use a pretrained CNN backbone
- [ ] Train a small classifier head
- [ ] Fine-tune selected layers
- [ ] Compare:
  - training from scratch
  - frozen feature extractor
  - fine-tuning
- [ ] Add model checkpointing
- [ ] Add practical image dataset workflow

### Key experiments

```txt
small custom image classification dataset
frozen backbone vs fine-tuning
few-shot behavior
data augmentation effects
```

### Exit criteria

- [ ] You can use pretrained models correctly
- [ ] You understand when to freeze vs fine-tune
- [ ] You can build a practical image classification workflow

---

## Phase 7 – Sequence Models and Attention Bridge

**Goal:** Build intuition for sequence modeling before Transformers.

**Level:** B+

**Estimated effort:** 16–28 hours

### Tasks

- [ ] Write theory for sequence data
- [ ] Write theory for embeddings
- [ ] Write theory for RNNs and LSTMs conceptually
- [ ] Implement a small RNN or GRU text classifier
- [ ] Write theory for attention
- [ ] Implement a minimal attention demo
- [ ] Compare bag-of-words / MLP / RNN-style model if useful

### Exit criteria

- [ ] You understand the sequence modeling problem
- [ ] You understand why attention was introduced
- [ ] You are ready to study Transformers without treating them as black boxes

---

## Phase 8 – Transformer Foundations

**Goal:** Understand the core architecture behind modern DL.

**Level:** A

**Estimated effort:** 20–35 hours

### Tasks

- [ ] Write theory for self-attention
- [ ] Write theory for queries, keys, values
- [ ] Write theory for positional encoding
- [ ] Write theory for multi-head attention
- [ ] Write theory for feed-forward blocks and residual connections
- [ ] Implement small self-attention module
- [ ] Implement a tiny Transformer block
- [ ] Use PyTorch modules for a practical Transformer-style experiment
- [ ] Compare manual attention intuition with framework implementation

### Exit criteria

- [ ] You can explain self-attention mathematically
- [ ] You can implement a minimal attention block
- [ ] You understand the structure of Transformer blocks

---

## Phase 9 – Final DL Practical Project

**Goal:** Build one serious end-to-end DL project.

**Level:** A

**Estimated effort:** 25–50 hours

### Candidate project options

Choose one:

```txt
image classifier with transfer learning
tabular + neural baseline comparison
text classifier with embeddings/Transformer
small time-series forecasting neural model
MLP/CNN/Transformer comparison notebook suite
```

### Requirements

- [ ] Real dataset
- [ ] Clean data pipeline
- [ ] Train/validation/test split
- [ ] Baselines
- [ ] Neural model
- [ ] Hyperparameter study
- [ ] Saved metrics
- [ ] Saved plots
- [ ] Final report
- [ ] README update
- [ ] Honest limitations section

### Exit criteria

- [ ] The DL project demonstrates a real end-to-end workflow
- [ ] The project is portfolio-quality
- [ ] The results are interpreted honestly
- [ ] The connection to ML Core is explicit

---

## Recommended first milestone

The first milestone of the DL project should be:

```txt
Create the repo, environment, action plan, and tensor/autograd sanity demos.
```

Do not start with a big neural architecture.

Start with:

```txt
Tensor
Autograd
Linear model
Training loop
Loss curve
Validation metric
```

This mirrors how `ml-core-cpp` succeeded: by building foundations in dependency order.

---

## Final handoff statement

`ml-core-cpp` answered:

```txt
How do classical ML models work internally?
How do losses, gradients, metrics, and evaluation workflows fit together?
How do we use models on real datasets?
```

The DL project should answer:

```txt
How do neural networks generalize these ideas?
How does autograd automate gradient computation?
How do modern architectures train and generalize?
How do we build serious DL workflows without treating frameworks as magic?
```

The correct next step is not more classical ML expansion.

The correct next step is a focused Deep Learning foundation project.
