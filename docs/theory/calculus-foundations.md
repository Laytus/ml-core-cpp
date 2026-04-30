# Calculus Foundations for Machine Learning

This document summarizes the minimum calculus foundations needed for ML Core.

The goal is not to study calculus in full generality. The goal is to understand the ideas required to reason about trainable models, losses, gradients, and optimization.

---

## 1. Why Calculus Matters in Machine Learning

Machine Learning models often contain parameters that must be learned from data.

For example, in a linear model:

$$
\hat{y} = Xw + b
$$

the parameters are:

```txt
w: weight vector
b: bias/intercept
```

Training means finding values of $w$ and $b$ that make predictions close to targets.

To do that, we define a loss function:

$$
L(w, b)
$$

The loss measures how bad the current model is.

Calculus tells us how the loss changes when the parameters change.

That is the core idea behind gradient-based learning.

---

## 2. Derivatives

A derivative measures how a function changes when its input changes.

For a function:

$$
f(x)
$$

the derivative is written:

$$
\frac{df}{dx}
$$

It measures the slope of the function at a point.

Example:

$$
f(x) = x^2
$$

The derivative is:

$$
\frac{df}{dx} = 2x
$$

So:

```txt
at x = 1, slope = 2
at x = 2, slope = 4
at x = -1, slope = -2
```

The derivative tells us both:

```txt
direction of change
rate of change
```

---

## 3. Derivatives and Minimization

In Machine Learning, we usually want to minimize a loss function.

If:

$$
\frac{dL}{dw} > 0
$$

then increasing $w$ increases the loss, so to reduce the loss we should decrease $w$.

If:

$$
\frac{dL}{dw} < 0
$$

then increasing $w$ decreases the loss, so to reduce the loss we should increase $w$.

This is why gradient descent updates parameters by moving in the opposite direction of the derivative.

For one parameter:

$$
w \leftarrow w - \alpha \frac{dL}{dw}
$$

where:

```txt
alpha = learning rate
```

The learning rate controls the size of the update.

---

## 4. Partial Derivatives

Most ML models have more than one parameter.

For example:

$$
L(w_1, w_2, b)
$$

depends on multiple variables.

A partial derivative measures how the function changes with respect to one variable while holding the others fixed.

Examples:

$$
\frac{\partial L}{\partial w_1}
$$

$$
\frac{\partial L}{\partial w_2}
$$

$$
\frac{\partial L}{\partial b}
$$

Each partial derivative answers one question:

```txt
If only this parameter changes slightly, how does the loss change?
```

---

## 5. Gradients

The gradient is the vector of all partial derivatives.

For a parameter vector:

$$
w =
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n
\end{bmatrix}
$$

the gradient of the loss with respect to $w$ is:

$$
\nabla_w L =
\begin{bmatrix}
\frac{\partial L}{\partial w_1} \\
\frac{\partial L}{\partial w_2} \\
\vdots \\
\frac{\partial L}{\partial w_n}
\end{bmatrix}
$$

The gradient points in the direction of steepest increase of the loss.

Therefore, gradient descent moves in the opposite direction:

$$
w \leftarrow w - \alpha \nabla_w L
$$

This is the core update rule behind many ML training algorithms.

---

## 6. Vectorized Linear Regression Example

For linear regression:

$$
\hat{y} = Xw + b
$$

where:

```txt
X: m x n design matrix
w: n-dimensional weight vector
b: scalar bias
y: m-dimensional target vector
```

The residual vector is:

$$
r = \hat{y} - y
$$

The mean squared error loss is:

$$
L(w, b) = \frac{1}{m} r^\top r
$$

or equivalently:

$$
L(w, b) = \frac{1}{m} \sum_{i=1}^{m}(\hat{y}_i - y_i)^2
$$

The gradient with respect to the weights is:

$$
\nabla_w L = \frac{2}{m} X^\top r
$$

The gradient with respect to the bias is:

$$
\frac{\partial L}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} r_i
$$

This shows why matrix operations matter:

```txt
Xw gives predictions
r gives prediction errors
Xᵀr aggregates feature-error relationships
```

---

## 7. Gradient Interpretation

The gradient tells us how each parameter contributes to the loss.

If one component of the gradient is large and positive:

```txt
increasing that parameter increases the loss strongly
```

If one component is large and negative:

```txt
increasing that parameter decreases the loss strongly
```

If one component is close to zero:

```txt
changing that parameter has little local effect on the loss
```

In training, the parameter update:

$$
w \leftarrow w - \alpha \nabla_w L
$$

adjusts each parameter according to its contribution to the loss.

---

## 8. Chain Rule

The chain rule explains how to differentiate composed functions.

If:

$$
z = f(g(x))
$$

then:

$$
\frac{dz}{dx} = \frac{df}{dg} \frac{dg}{dx}
$$

In words:

```txt
change of output with respect to input
=
change of output with respect to intermediate value
times
change of intermediate value with respect to input
```

This is essential because ML models are compositions of operations.

Example:

$$
\hat{y} = Xw + b
$$

$$
L = \frac{1}{m}(\hat{y} - y)^\top(\hat{y} - y)
$$

The loss depends on predictions, and predictions depend on parameters.

The chain rule lets us connect:

```txt
parameters → predictions → residuals → loss
```

---

## 9. Why Chain Rule Matters for Trainable Models

Trainable models are built from layers of computation.

Even simple models have structure:

```txt
inputs
→ linear combination
→ prediction
→ loss
```

Deep Learning extends this idea:

```txt
inputs
→ layer 1
→ activation
→ layer 2
→ activation
→ output
→ loss
```

Backpropagation is essentially the chain rule applied efficiently through this computational graph.

ML Core does not need full backpropagation yet, but the chain rule must be understood before the Deep Learning bridge.

---

## 10. Gradient Descent Intuition

Gradient descent is an iterative optimization method.

Starting from initial parameters:

$$
w^{(0)}
$$

we repeatedly update:

$$
w^{(t+1)} = w^{(t)} - \alpha \nabla_w L(w^{(t)})
$$

where:

```txt
t = iteration number
alpha = learning rate
```

The learning rate matters:

```txt
too small  → training is slow
too large  → training may diverge
```

Gradient descent does not magically find the best answer in one step. It follows local slope information repeatedly.

---

## 11. What This Means for ML Core

For ML Core, calculus is needed to understand:

```txt
loss functions
gradients
parameter updates
optimization
training loops
regularization
backpropagation later
```

The immediate practical connection is:

```txt
linear model:
    predictions = Xw + b

loss:
    MSE = mean squared error

gradient:
    tells how to update w and b

optimizer:
    repeatedly applies updates
```

This foundation will be used directly in:

```txt
Phase 3 – Linear Models
Phase 4 – Logistic Regression
Phase 5 – Optimization for ML
Phase 10 – Bridge to Deep Learning
```

---

## 12. Summary

The essential ideas are:

```txt
derivative:
    slope of a function with one input

partial derivative:
    effect of one variable while holding others fixed

gradient:
    vector of partial derivatives

gradient descent:
    move parameters opposite the gradient

chain rule:
    differentiate composed functions

backpropagation:
    efficient chain rule through layered models
```

These ideas are the mathematical basis of trainable Machine Learning models.