# Optimization for Machine Learning

This document covers the optimization foundations needed before Deep Learning.

Optimization is the mechanism that turns a model from a fixed mathematical function into a trainable system.

In previous phases, models such as `LinearRegression`, `LogisticRegression`, and `SoftmaxRegression` already used gradient-based training internally.

Phase 5 makes optimization explicit and reusable.

The goal is to understand and implement:

```txt
batch gradient descent
stochastic gradient descent
mini-batch gradient descent
momentum
conditioning and scaling effects
initialization sensitivity
convergence behavior
adaptive optimizer intuition
```

---

## 1. Batch Gradient Descent, SGD, and Mini-Batch Gradient Descent

Gradient descent is an iterative optimization method.

It updates model parameters by moving them in the opposite direction of the loss gradient.

For a parameter vector:

$$
\theta
$$

and a loss function:

$$
J(\theta)
$$

the generic update rule is:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)
$$

where:

```txt
θ = trainable parameters
J(θ) = loss function
∇J(θ) = gradient of the loss with respect to parameters
α = learning rate
```

The gradient points in the direction of steepest increase of the loss.

Therefore, subtracting the gradient moves the parameters in a direction that should reduce the loss.

---

### Batch Gradient Descent

Batch Gradient Descent computes the gradient using the full training dataset at every update.

For a dataset:

```txt
X: m x n feature matrix
y: target vector or target matrix
```

each update uses all `m` training samples.

For linear regression with MSE:

$$
\hat{y} = Xw + b
$$

$$
r = \hat{y} - y
$$

The batch gradient is:

$$
\nabla_w J = \frac{2}{m}X^\top r
$$

and:

$$
\frac{\partial J}{\partial b} =
\frac{2}{m}\sum_{i=1}^{m}r_i
$$

The update is:

$$
w \leftarrow w - \alpha \nabla_w J
$$

$$
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
$$

Batch Gradient Descent has stable updates because every step sees the full dataset.

Advantages:

```txt
stable loss curve
deterministic when initialization is fixed
simple to reason about mathematically
good for small and medium datasets
works naturally with vectorized formulas
```

Disadvantages:

```txt
each update can be expensive for large datasets
parameters are updated only once per full dataset pass
can be slower when many cheap approximate updates would work
```

In ML Core, Batch GD is the cleanest first reusable optimizer because it matches the vectorized model formulas used throughout the project.

---

### Stochastic Gradient Descent

Stochastic Gradient Descent, or SGD, updates parameters using one sample at a time.

Instead of computing the gradient over the full dataset, SGD uses a single training example:

```txt
x_i, y_i
```

Conceptually:

```txt
for each epoch:
    shuffle samples
    for each sample:
        compute gradient on one sample
        update parameters
```

The update rule is still:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta J_i(\theta)
$$

but now:

```txt
J_i(θ) = loss on one sample
```

The gradient is noisy because one sample may not represent the full dataset.

However, this noise can be useful.

Advantages:

```txt
cheap updates
can start improving quickly
works well for large datasets
introduces useful stochasticity
is the conceptual base of many modern training methods
```

Disadvantages:

```txt
loss curve is noisy
convergence is less smooth
learning rate choice is more sensitive
single-sample gradients can be unstable
```

SGD usually does not move directly toward the minimum.

Instead, it follows a noisy path that trends toward lower loss over time.

---

### Mini-Batch Gradient Descent

Mini-batch Gradient Descent is a compromise between Batch GD and SGD.

Instead of using:

```txt
all samples
```

or:

```txt
one sample
```

it uses a small subset of samples per update:

```txt
batch size = 16, 32, 64, 128, ...
```

For a mini-batch:

```txt
X_batch: k x n
y_batch: k targets
```

the update is:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta J_{batch}(\theta)
$$

where:

```txt
J_batch = loss computed on the current mini-batch
```

Advantages:

```txt
more stable than SGD
more frequent updates than Batch GD
efficient with matrix operations
standard practical default in deep learning
works well with hardware acceleration
```

Disadvantages:

```txt
requires choosing batch size
loss curve can still be noisy
implementation is more complex than Batch GD
```

Mini-batch GD is usually the best practical default for larger models and datasets.

---

### Epochs and iterations

An iteration is one parameter update.

An epoch means the optimizer has processed the full training dataset once.

For Batch GD:

```txt
1 epoch = 1 update
```

For SGD:

```txt
1 epoch = m updates
```

where `m` is the number of samples.

For Mini-batch GD:

```txt
1 epoch = ceil(m / batch_size) updates
```

This distinction matters when comparing optimizers.

A fair experiment should track:

```txt
iterations
epochs
loss history
final loss
converged flag
stopping reason
```

---

### Comparison summary

| Method | Samples per update | Update stability | Cost per update | Typical use |
|---|---:|---:|---:|---|
| Batch GD | Full dataset | High | High | Small/medium datasets, clean theory |
| SGD | 1 sample | Low | Very low | Large datasets, noisy optimization |
| Mini-batch GD | Small batch | Medium | Medium | Practical ML and DL default |

---

### ML Core implementation goal

Phase 5 should implement reusable optimizers that can train more than one model type.

The optimizer should not be hardcoded only for linear regression.

The goal is to support at least two trainable models through shared optimizer-aware structures.

The initial target models are:

```txt
LinearRegression
LogisticRegression
```

and possibly later:

```txt
SoftmaxRegression
tiny neural-network bridge models
```

---

## 2. Momentum and Why It Helps

Momentum is an optimization technique that smooths gradient descent updates.

Instead of updating parameters using only the current gradient, momentum accumulates a velocity vector.

The basic momentum equations are:

$$
v_t = \beta v_{t-1} + \nabla J(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \alpha v_t
$$

where:

```txt
v_t = velocity at step t
β = momentum coefficient
α = learning rate
∇J(θ_t) = current gradient
```

Typical values:

```txt
β = 0.9
β = 0.99
```

---

### Intuition

Ordinary gradient descent only reacts to the current gradient.

Momentum remembers the direction of recent gradients.

If gradients keep pointing in a similar direction, momentum accelerates movement in that direction.

If gradients alternate directions, momentum dampens oscillation.

This is useful in narrow valleys.

Without momentum, gradient descent may zigzag:

```txt
left wall
right wall
left wall
right wall
```

With momentum, the optimizer can move more smoothly along the valley.

---

### Momentum as velocity

A useful analogy is a ball rolling downhill.

Without momentum:

```txt
the update reacts only to the current slope
```

With momentum:

```txt
the update has inertia
```

The optimizer builds speed in directions that consistently reduce loss.

This can make convergence faster and smoother.

---

### Why momentum helps

Momentum can help with:

```txt
faster convergence
less zigzagging
better behavior in poorly conditioned problems
smoother optimization paths
less sensitivity to noisy mini-batch gradients
```

Momentum is especially useful when the loss surface has long, narrow valleys.

It is also useful with stochastic or mini-batch gradients because it averages update direction over time.

---

### Momentum and learning rate

Momentum changes the effective update size.

A learning rate that works without momentum may be too large with momentum.

This happens because momentum accumulates past gradients.

So optimizer experiments should compare:

```txt
same learning rate without momentum
same learning rate with momentum
smaller learning rate with momentum
```

Momentum should not be treated as an isolated switch.

It interacts directly with learning rate and convergence behavior.

---

### ML Core implementation goal

Momentum should be implemented as reusable optimizer state, not copied separately into each model.

The optimizer should track:

```txt
current parameters
current gradients
velocity state
learning rate
momentum coefficient
```

The project should be able to compare:

```txt
Batch GD without momentum
Batch GD with momentum
SGD without momentum
SGD with momentum
Mini-batch GD without momentum
Mini-batch GD with momentum
```

The result should be visible in structured training history.

---

## 3. Conditioning, Scaling, and Optimization Geometry

Optimization behavior depends strongly on the geometry of the loss surface.

Even when a loss function is convex, gradient descent can still be slow if the problem is poorly conditioned.

---

### Loss surface geometry

For linear regression with MSE, the loss surface is convex.

That means there is one global minimum.

However, the shape of the surface can vary.

A well-conditioned loss surface looks like a round bowl.

A poorly conditioned loss surface looks like a long narrow valley.

In a round bowl:

```txt
gradient descent can move directly toward the minimum
```

In a narrow valley:

```txt
gradient descent may zigzag and converge slowly
```

---

### Conditioning

Conditioning describes how stretched or distorted the loss surface is.

Poor conditioning often happens when features have very different scales or are highly correlated.

Example:

```txt
feature 1 values around 1
feature 2 values around 1,000,000
```

The large-scale feature can dominate the gradient.

This can make updates unstable or inefficient.

---

### Feature scaling and optimization

Feature scaling improves optimization geometry.

Standardization transforms each feature using training statistics:

$$
x' = \frac{x - \mu}{\sigma}
$$

where:

```txt
μ = training feature mean
σ = training feature standard deviation
```

After scaling, features tend to have comparable magnitude.

This often makes the loss surface less stretched.

As a result:

```txt
gradient descent can converge faster
larger learning rates may become stable
updates become more balanced across features
regularization behaves more consistently
```

---

### Scaling and leakage

Scaling must be done without data leakage.

Correct workflow:

```txt
1. split data
2. fit scaler on training data only
3. transform training data
4. transform validation/test data using training scaler
5. train model
6. evaluate model
```

Incorrect workflow:

```txt
1. fit scaler on all data
2. split data
3. train and evaluate
```

The incorrect workflow leaks information from validation/test data into training preprocessing.

Phase 2 already established this rule.

Phase 5 uses this rule to compare optimization behavior fairly.

---

### Conditioning and learning rate

Poor conditioning makes learning-rate selection harder.

In a narrow valley:

```txt
a large learning rate may overshoot across the steep direction
a small learning rate may move too slowly along the flat direction
```

So there may be no single learning rate that is both:

```txt
stable along steep directions
fast along flat directions
```

Momentum and adaptive optimizers partly address this problem.

---

### Optimization geometry and regularization

Regularization also interacts with scaling.

Ridge penalties depend on weight magnitude:

$$
\lambda \|w\|_2^2
$$

If feature scales differ greatly, then weight magnitudes are not directly comparable.

This means Ridge can penalize features unevenly unless data is scaled.

Therefore, regularized models should usually be trained with scaled features.

---

### ML Core implementation goal

Phase 5 experiments should explicitly compare:

```txt
scaled vs unscaled data
different learning rates
Batch GD vs SGD vs Mini-batch GD
with vs without momentum
```

The goal is not only to get a lower loss.

The goal is to connect optimizer behavior to the geometry of the problem.

---

## 4. Initialization Sensitivity and Convergence Behavior

Initialization is the choice of parameter values before training begins.

For a parameter vector:

$$
\theta
$$

initialization defines:

$$
\theta_0
$$

where optimization starts.

---

### Initialization in linear models

For convex linear regression with MSE, initialization usually does not affect the final optimum if:

```txt
learning rate is valid
training runs long enough
loss is convex
```

Common initialization:

```txt
weights = zeros
bias = 0
```

This is acceptable for linear regression.

However, initialization can still affect:

```txt
initial loss
training trajectory
number of iterations needed
interaction with learning rate
```

So it is still useful to track convergence behavior.

---

### Initialization in logistic and softmax regression

For logistic regression and softmax regression, zero initialization can still work because these models are shallow linear classifiers.

However, training behavior can still depend on:

```txt
learning rate
feature scale
regularization
class separability
initial parameter scale
```

In nearly separable data, weights can grow very large without regularization.

This affects convergence and probability confidence.

---

### Initialization in neural networks

Initialization becomes much more important in neural networks.

Neural networks are non-convex.

Different initial parameters can lead to:

```txt
different convergence speeds
different final solutions
unstable training
vanishing or exploding activations
symmetry problems between hidden units
```

This is why initialization is a major topic in Deep Learning.

Phase 5 should introduce the concept now, even if ML Core models are mostly convex or shallow.

---

### Convergence behavior

Convergence means the optimizer has reached a point where training is no longer changing meaningfully.

Common convergence criteria:

```txt
loss improvement below tolerance
gradient norm below tolerance
maximum iterations reached
maximum epochs reached
non-finite loss detected
```

A robust optimizer should not only return final parameters.

It should return structured information about the training run.

---

### Loss history patterns

Loss history is one of the main tools for diagnosing optimization.

#### Healthy convergence

```txt
loss decreases steadily
loss eventually flattens
no sudden explosions
```

#### Learning rate too small

```txt
loss decreases very slowly
many iterations produce tiny progress
training may stop before reaching a useful solution
```

#### Learning rate too large

```txt
loss oscillates strongly
loss increases
loss becomes NaN or infinity
training diverges
```

#### Noisy stochastic training

```txt
individual updates fluctuate
loss trend decreases overall
mini-batch loss is not perfectly smooth
```

This is expected for SGD and mini-batch GD.

---

### Stopping reasons

A reusable optimizer should record why training stopped.

Possible stopping reasons:

```txt
max_iterations_reached
max_epochs_reached
loss_tolerance_reached
gradient_tolerance_reached
non_finite_loss
manual_stop_or_unknown
```

This makes convergence analysis reproducible.

It avoids relying on manual inspection only.

---

### ML Core implementation goal

Phase 5 should introduce reusable training history structures.

These should record:

```txt
loss history
iteration count
epoch count
converged flag
stopping reason
optimizer name
learning rate
batch size
momentum coefficient
```

This will allow fair comparison across optimizers and models.

---

## 5. Adaptive Optimizers — Conceptual Bridge

Adaptive optimizers adjust updates using information from past gradients.

They are especially important in Deep Learning.

Common adaptive optimizers include:

```txt
AdaGrad
RMSProp
Adam
AdamW
```

Phase 5 does not need to fully implement all of them.

However, it should explain why they exist and how they relate to GD, momentum, and conditioning.

---

### Why adaptive optimizers exist

A single global learning rate may not work equally well for all parameters.

Some parameters may receive large gradients.

Others may receive small or sparse gradients.

Adaptive optimizers adjust the effective learning rate per parameter.

This can help when:

```txt
features have different scales
gradients are sparse
loss geometry is poorly conditioned
training is noisy
models are deep or non-convex
```

---

### AdaGrad intuition

AdaGrad accumulates squared gradients for each parameter.

Parameters with large historical gradients get smaller effective learning rates.

Parameters with small historical gradients keep larger effective learning rates.

Conceptually:

```txt
frequently updated parameters slow down
rarely updated parameters can still move
```

AdaGrad can be useful for sparse features.

However, the accumulated squared gradients only grow.

So effective learning rates can become too small over time.

---

### RMSProp intuition

RMSProp modifies AdaGrad by using a moving average of squared gradients instead of an ever-growing sum.

Conceptually:

```txt
track recent gradient scale
divide updates by recent gradient magnitude
```

This prevents learning rates from shrinking too aggressively.

RMSProp is useful for noisy and non-stationary training behavior.

---

### Adam intuition

Adam combines two ideas:

```txt
momentum over gradients
adaptive scaling using squared gradients
```

It tracks:

```txt
first moment:
    moving average of gradients

second moment:
    moving average of squared gradients
```

Adam is popular because it often works well with little tuning.

However, it is not magic.

It can still require careful learning-rate selection and can generalize differently from SGD with momentum.

---

### AdamW intuition

AdamW modifies Adam by decoupling weight decay from the adaptive gradient update.

This matters because L2 regularization and weight decay are not exactly equivalent under adaptive optimizers.

AdamW is widely used in modern deep learning.

For ML Core, AdamW is mainly a conceptual bridge to later DL training.

---

### Implementation decision for ML Core

Phase 5 should focus implementation on:

```txt
Batch GD
SGD
Mini-batch GD
Momentum
Reusable training history
```

Adaptive optimizers should be documented conceptually.

A full Adam or AdamW implementation can be deferred to the Deep Learning bridge or a later extension.

The important goal is to understand why adaptive methods exist and how they relate to:

```txt
learning rates
momentum
gradient scale
conditioning
deep learning optimization
```

---

## 6. Optimizer Behavior Summary

Phase 5 produced a reusable optimization layer and a set of comparison outputs that make optimizer behavior explicit instead of relying on manual inspection.

The main generated summaries are stored in:

```txt
outputs/phase-5-optimization/
```

They compare:

```txt
batch_vs_sgd_vs_mini_batch
momentum_comparison
learning_rate_comparison
scaled_vs_unscaled_comparison
```

These outputs should be read as small controlled optimization studies. They are not meant to prove universal optimizer superiority. Their purpose is to show how optimizer choices affect convergence behavior under controlled synthetic settings.

---

### Batch GD vs SGD vs Mini-batch GD

The comparison between Batch Gradient Descent, Stochastic Gradient Descent, and Mini-batch Gradient Descent shows the trade-off between update stability and update frequency.

Batch Gradient Descent uses the full dataset for every update. Its loss trajectory is usually the smoothest because each gradient is computed from all available samples. This makes it easy to reason about and useful for small datasets or theory-focused experiments.

SGD updates after each individual sample. Its updates are noisier because each gradient is based on only one example. However, it performs many more parameter updates per epoch and introduces stochasticity that can be useful in larger or more complex training settings.

Mini-batch Gradient Descent sits between the two. It uses small groups of samples per update, giving a compromise between noisy single-sample updates and expensive full-dataset updates. This is the practical default in many modern ML and DL workflows.

In this project, all three optimizers are implemented through the same `OptimizationProblem` interface and can train:

```txt
LinearRegressionOptimizationProblem
LogisticRegressionOptimizationProblem
SoftmaxRegressionOptimizationProblem
```

This confirms that optimization logic is now reusable instead of being hardcoded separately inside each model.

---

### Momentum vs No Momentum

The momentum comparison shows how adding a velocity term changes the update behavior.

Without momentum, each update depends only on the current gradient. With momentum, the optimizer accumulates a running direction of recent gradients:

```txt
velocity = momentum * velocity + gradient
parameters = parameters - learning_rate * velocity
```

Momentum can accelerate movement in consistent descent directions and reduce zigzagging in poorly conditioned regions. It is especially useful when gradients repeatedly point in similar directions or when stochastic updates are noisy.

Momentum is not automatically better in every situation. It interacts strongly with the learning rate. A learning rate that is stable without momentum may become too aggressive with momentum because the velocity accumulates past gradients.

For this reason, the Phase 5 optimizer options explicitly track:

```txt
learning_rate
momentum
batch_size
random_seed
shuffle
```

and the training history records the momentum used in each run.

---

### Learning-rate comparison

The learning-rate comparison demonstrates that optimization quality depends strongly on step size.

A very small learning rate tends to produce slow progress. The loss may decrease, but many epochs may be needed to reach a useful solution.

A moderate learning rate usually reaches a good solution more quickly.

A too-large learning rate can cause oscillation, instability, or divergence, especially on poorly scaled data or when momentum is enabled.

This is why Phase 5 records structured training histories rather than only final parameters. A useful optimization run should be interpreted using:

```txt
initial_loss
final_loss
best_loss
loss_improvement
iterations_run
epochs_run
converged
stop_reason
gradient_norms
```

These fields make convergence behavior inspectable and comparable.

---

### Scaled vs Unscaled Optimization Behavior

The scaled-vs-unscaled comparison shows why feature scaling is an optimization issue, not only a preprocessing detail.

When features have very different magnitudes, the loss surface can become poorly conditioned. Gradient descent may need a very small learning rate to avoid unstable updates along large-scale feature directions.

After scaling, feature magnitudes become more comparable. This usually improves optimization geometry, allows more useful learning rates, and makes gradient updates more balanced across parameters.

The Phase 5 experiment uses a controlled linear-regression dataset with one feature at a much larger scale than the other. The unscaled run requires a very small learning rate, while the scaled run can use a much larger learning rate.

This reinforces one of the main rules from the evaluation methodology phase:

```txt
fit preprocessing only on training data
transform validation/test data using training-fitted preprocessing
```

Scaling improves optimization, but it must still be done without data leakage.

---

### Main Phase 5 conclusion

The main outcome of Phase 5 is that optimization is now a reusable framework in the project.

The project no longer depends only on isolated model-specific training loops. Instead, models can expose their loss and gradients through optimizer-aware adapters, while optimizers handle the training loop, parameter updates, stopping behavior, and history logging.

The important architectural split is:

```txt
OptimizationProblem adapter:
    model-specific loss
    model-specific gradients
    parameter access and updates

Optimizer:
    training loop
    batch selection
    learning rate
    momentum
    stopping rules
    history logging
```

This design is the correct bridge toward Deep Learning, where models become more complex but the same optimization principles remain central.

Phase 5 is successful when optimizer behavior can be compared through structured outputs instead of informal observation. The generated files in `outputs/phase-5-optimization/` satisfy that goal.