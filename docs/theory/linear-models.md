# Linear Models

This document covers the linear model foundations used in ML Core.

The goal is to understand linear regression as a serious multivariate model, not as a scalar toy example.

Linear models are the first major model family in ML Core because they connect directly to:

```txt
matrix operations
loss functions
gradients
optimization
regularization
evaluation methodology
```

---

## 1. Multivariate Linear Regression

Multivariate linear regression predicts a continuous target using multiple input features.

For one sample with `n` features:

$$
x =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

and weights:

$$
w =
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n
\end{bmatrix}
$$

the prediction is:

$$
\hat{y} = w^\top x + b
$$

Expanded:

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

where:

```txt
ŷ = predicted target
x = feature vector for one sample
w = weight vector
b = bias/intercept
```

Each weight controls how strongly one feature contributes to the prediction.

For example:

```txt
positive weight:
    increasing the feature increases the prediction

negative weight:
    increasing the feature decreases the prediction

weight near zero:
    feature has little linear effect on the prediction
```

The bias term shifts predictions independently of the features.

---

### Dataset-level representation

ML Core does not implement linear regression as a scalar single-feature model.

The default representation is dataset-level and multivariate:

```txt
X: m x n feature matrix
y: m-dimensional target vector
w: n-dimensional weight vector
b: scalar bias
```

where:

```txt
m = number of samples
n = number of features
```

The convention is:

```txt
rows of X    = samples
columns of X = features
```

So:

$$
X =
\begin{bmatrix}
    - & x_1^\top & - \\ 
    - & x_2^\top & - \\
      & \vdots   &   \\
    - & x_m^\top & -
\end{bmatrix}
$$

Each row is one sample.

The target vector is:

$$
y =
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{bmatrix}
$$

The model produces one prediction per row.

---

### What the model learns

Training linear regression means finding:

```txt
weights w
bias b
```

so that predictions are close to the true targets.

The model does not memorize rows directly. It learns a linear relationship between features and target values.

A linear model is appropriate when the target can be reasonably approximated as:

```txt
weighted sum of input features + bias
```

It can still be useful even when the real relationship is not perfectly linear because it often provides:

```txt
strong baseline behavior
fast training
interpretable coefficients
good behavior with regularization
```

---

## 2. Linear Models as Matrix Operations

For the full dataset, predictions can be computed at once.

Given:

```txt
X: m x n
w: n
b: scalar
```

the vectorized prediction is:

$$
\hat{y} = Xw + b
$$

More explicitly, the bias is added to every prediction:

$$
\hat{y} = Xw + b\mathbf{1}
$$

where:

```txt
1 = m-dimensional vector of ones
```

So:

$$
\hat{y} =
\begin{bmatrix}
x_1^\top w + b \\
x_2^\top w + b \\
\vdots \\
x_m^\top w + b
\end{bmatrix}
$$

This is the main reason ML Core uses Eigen.

Instead of looping sample by sample in model code, linear prediction should be implemented as:

```cpp
ml::Vector predictions = X * weights;
predictions.array() += bias;
```

This matches the Phase 1 utility:

```cpp
linear_prediction(X, weights, bias);
```

---

### Shape requirements

The operation:

$$
Xw
$$

is valid only when:

```txt
X.cols() == w.size()
```

If:

```txt
X is m x n
w is n-dimensional
```

then:

```txt
Xw is m-dimensional
```

The result aligns with the target vector:

```txt
predictions.size() == y.size()
X.rows() == y.size()
```

These shape assumptions must be validated before fitting or predicting.

---

### Why vectorization matters

Vectorized implementation is important because it gives:

```txt
clearer mathematical structure
less duplicated scalar code
better performance through Eigen
cleaner gradient formulas
easier extension to multivariate data
```

A scalar-first implementation such as:

```txt
for each sample:
    prediction = weight * x + bias
```

does not scale well to the real ML Core design.

The correct default is:

```txt
operate on matrices and vectors
validate shapes
return dataset-level predictions
```

---

### Design matrix with explicit bias column

There are two common ways to handle the bias term.

Option 1: keep bias separate:

$$
\hat{y} = Xw + b
$$

Option 2: add a column of ones to the feature matrix:

$$
\tilde{X} =
\begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1n} \\
1 & x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \vdots & & \vdots \\
1 & x_{m1} & x_{m2} & \cdots & x_{mn}
\end{bmatrix}
$$

and include the bias inside the parameter vector:

$$
\theta =
\begin{bmatrix}
b \\
w_1 \\
w_2 \\
\vdots \\
w_n
\end{bmatrix}
$$

Then:

$$
\hat{y} = \tilde{X}\theta
$$

ML Core will keep the bias separate in the model interface because it is explicit and easy to interpret:

```txt
weights = feature coefficients
bias    = intercept
```

The normal equation section may still use the augmented design matrix concept because it is mathematically convenient.

---

## 3. Vectorized MSE

The prediction error vector is called the residual vector.

$$
r = \hat{y} - y
$$

where:

```txt
r: m-dimensional residual vector
ŷ: m-dimensional prediction vector
y: m-dimensional target vector
```

For sample `i`:

$$
r_i = \hat{y}_i - y_i
$$

The Mean Squared Error is:

$$
MSE = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2
$$

Using the residual vector:

$$
MSE = \frac{1}{m}r^\top r
$$

because:

$$
r^\top r = \sum_{i=1}^{m}r_i^2
$$

So the vectorized expression is:

$$
MSE = \frac{1}{m}(\hat{y} - y)^\top(\hat{y} - y)
$$

Substituting the linear model:

$$
MSE(w,b) = \frac{1}{m}(Xw + b - y)^\top(Xw + b - y)
$$

More explicitly:

$$
MSE(w,b) = \frac{1}{m}\|Xw + b - y\|_2^2
$$

---

### Implementation form

In ML Core, MSE can be computed from predictions and targets:

```cpp
double mse = mean_squared_error(predictions, targets);
```

Internally, this should be equivalent to:

```cpp
ml::Vector r = predictions - targets;
double mse = r.squaredNorm() / static_cast<double>(r.size());
```

This matches the mathematical form:

$$
\frac{1}{m}r^\top r
$$

---

### Why MSE is useful for linear regression

MSE is commonly used with linear regression because it is:

```txt
smooth
differentiable
convex for ordinary linear regression
easy to express with matrix operations
strongly penalizes large errors
```

The smoothness and convexity are important because they make optimization easier.

For ordinary least squares linear regression, the MSE loss has a single global minimum.

That means gradient-based training can converge to the best solution if configured correctly.

---

### Gradient connection

For linear regression with MSE:

$$
r = Xw + b - y
$$

The gradient with respect to the weights is:

$$
\nabla_w MSE = \frac{2}{m}X^\top r
$$

The gradient with respect to the bias is:

$$
\frac{\partial MSE}{\partial b} = \frac{2}{m}\sum_{i=1}^{m}r_i
$$

These formulas will drive gradient descent training.

The important interpretation is:

```txt
Xᵀr measures how each feature is aligned with the current prediction errors
```

If a feature tends to be positive when residuals are positive, the corresponding gradient component will be positive.

If a feature tends to be positive when residuals are negative, the corresponding gradient component will be negative.

This is how the model knows how to adjust each weight.

---

## 4. Normal Equation Concept

Linear regression can be solved in two major ways:

```txt
1. closed-form solution
2. iterative optimization
```

The closed-form solution is known as the normal equation.

For linear regression without separating the bias, using an augmented matrix:

$$
\tilde{X}
$$

and parameter vector:

$$
\theta
$$

the least-squares solution is:

$$
\theta = (\tilde{X}^\top\tilde{X})^{-1}\tilde{X}^\top y
$$

This equation gives the parameters that minimize squared error, assuming the inverse exists.

---

### Why the normal equation matters

Even if ML Core mainly trains models using optimization, the normal equation is still important conceptually.

It shows that ordinary least squares has a direct mathematical solution.

It also helps explain:

```txt
why matrix structure matters
why XᵀX appears in linear regression
why feature correlation can cause numerical problems
why regularization improves stability
```

The normal equation is also a useful reference point when testing gradient-based training.

For small datasets, a closed-form solution can be used to verify whether gradient descent is converging toward the correct parameters.

---

### Limitations of the normal equation

The normal equation is not always the best computational route.

Main limitations:

```txt
requires forming XᵀX
may require matrix inversion or solving a linear system
can be numerically unstable if features are highly correlated
can be expensive for large feature counts
does not generalize naturally to all model families
```

In practice, explicitly computing the inverse is usually discouraged.

Instead of:

$$
(\tilde{X}^\top\tilde{X})^{-1}
$$

numerical code should prefer solving the linear system:

$$
\tilde{X}^\top\tilde{X}\theta = \tilde{X}^\top y
$$

This is more stable than manually computing the inverse.

---

### Ridge version of the normal equation

Ridge regression modifies the least-squares objective by adding an L2 penalty.

Conceptually, the Ridge closed-form solution is:

$$
\theta = (\tilde{X}^\top\tilde{X} + \lambda I)^{-1}\tilde{X}^\top y
$$

where:

```txt
λ = regularization strength
I = identity matrix
```

This improves numerical stability because the matrix being solved becomes better conditioned.

In practice, the bias term is often not regularized.

That means the identity penalty should apply to feature weights, not necessarily to the intercept.

This detail matters for implementation.

---

### ML Core implementation decision

ML Core will primarily implement training through gradient-based optimization.

The normal equation is included because it gives a conceptual and testing reference.

For Phase 3, the implementation should focus on:

```txt
multivariate prediction
MSE loss
gradient-based fitting
Ridge support
evaluation against baseline
```

The normal equation may be implemented later as an optional comparison path if useful, but it is not the main training route.

---

## 5. Regularization

Regularization adds a penalty to the training objective to discourage overly complex parameter values.

For linear regression, the unregularized objective is:

$$
MSE(w,b) = \frac{1}{m}\|Xw + b - y\|_2^2
$$

Regularized linear regression modifies the objective by adding a penalty term:

$$
Loss(w,b) = MSE(w,b) + \text{regularization penalty}
$$

The goal is not only to fit the training data, but to find parameters that generalize better.

Regularization is especially useful when:

```txt
features are correlated
there are many features
the dataset is small relative to the number of features
the model overfits training data
the normal equation is numerically unstable
```

A key implementation decision is that ML Core will regularize feature weights, but not the bias term.

The bias only shifts predictions up or down. Penalizing it usually does not control model complexity in the same meaningful way as penalizing feature weights.

So the regularization penalty applies to:

```txt
w
```

not:

```txt
b
```

---

## 6. Ridge Regression

Ridge regression is linear regression with L2 regularization.

The Ridge objective is:

$$
J(w,b) =
\frac{1}{m}\|Xw + b - y\|_2^2
+
\lambda \|w\|_2^2
$$

where:

```txt
λ = regularization strength
w = feature weights
b = bias/intercept
```

The L2 penalty is:

$$
\|w\|_2^2 = w^\top w = \sum_{j=1}^{n}w_j^2
$$

So Ridge penalizes large weight magnitudes.

When:

```txt
λ = 0
```

Ridge becomes ordinary linear regression.

When:

```txt
λ is large
```

the model is pushed toward smaller weights.

---

### Ridge intuition

Ridge discourages the model from relying too strongly on any one feature.

This usually improves generalization when the training data is noisy or when features are highly correlated.

Example intuition:

```txt
without Ridge:
    model may assign very large positive and negative weights to correlated features

with Ridge:
    model prefers smaller, more stable weights
```

Ridge does not usually set weights exactly to zero.

Instead, it shrinks weights continuously toward zero.

This means Ridge is useful for stability and generalization, but not mainly for feature selection.

---

### Ridge gradient

For ordinary MSE:

$$
\nabla_w MSE = \frac{2}{m}X^\top r
$$

where:

$$
r = Xw + b - y
$$

For the Ridge penalty:

$$
\lambda \|w\|_2^2
$$

the gradient is:

$$
\nabla_w \lambda \|w\|_2^2 = 2\lambda w
$$

Therefore, the full Ridge gradient with respect to the weights is:

$$
\nabla_w J =
\frac{2}{m}X^\top r + 2\lambda w
$$

The gradient with respect to the bias remains:

$$
\frac{\partial J}{\partial b} =
\frac{2}{m}\sum_{i=1}^{m}r_i 
$$

because the bias is not regularized.

---

### Ridge update rule

Using gradient descent:

$$
w \leftarrow w - \alpha
\left(
\frac{2}{m}X^\top r + 2\lambda w
\right)
$$

and:

$$
b \leftarrow b - \alpha
\left(
\frac{2}{m}\sum_{i=1}^{m}r_i
\right)
$$

where:

```txt
α = learning rate
λ = regularization strength
```

The Ridge term pulls weights toward zero at every update.

This is sometimes called weight decay.

---

### Ridge and feature scaling

Ridge is sensitive to feature scale.

If one feature has values around:

```txt
1,000,000
```

and another has values around:

```txt
0.01
```

then the same weight magnitude does not mean the same effect.

Because Ridge penalizes weight magnitude, features should usually be scaled before Ridge regression.

A leakage-safe workflow is:

```txt
1. split data
2. fit scaler on training features
3. transform train/validation/test using training scaler
4. train Ridge model
5. evaluate model against baseline
```

This connects directly to the Phase 2 preprocessing pipeline.

---

### Ridge normal equation

Using an augmented design matrix, the Ridge closed-form solution is:

$$
\theta =
(\tilde{X}^\top\tilde{X} + \lambda I)^{-1}\tilde{X}^\top y
$$

However, if the first component of `θ` represents the bias, then the bias should usually not be regularized.

So conceptually, the penalty matrix should be:

$$
P =
\begin{bmatrix}
0 & 0 & \cdots & 0 \\
0 & 1 &        & 0 \\
\vdots &   & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
$$

and the solution becomes:

$$
\theta =
(\tilde{X}^\top\tilde{X} + \lambda P)^{-1}\tilde{X}^\top y
$$

This detail matters because regularizing the intercept can shift the model unnecessarily.

ML Core will primarily use gradient-based training for Ridge, but the normal equation form is useful for understanding why Ridge improves numerical stability.

---

### Ridge implementation decision in ML Core

For Phase 3, Ridge support should be implemented through the same linear regression model by exposing a regularization parameter.

A clean configuration is:

```txt
regularization = none or ridge
lambda = non-negative regularization strength
```

Rules:

```txt
lambda = 0:
    ordinary linear regression

lambda > 0 with Ridge:
    add L2 penalty to the weight gradient

bias:
    never regularized
```

This keeps ordinary linear regression and Ridge regression in one reusable implementation.

---

## 7. Lasso Regression

Lasso regression is linear regression with L1 regularization.

The Lasso objective is:

$$
J(w,b) =
\frac{1}{m}\|Xw + b - y\|_2^2
+
\lambda \|w\|_1
$$

where:

$$
\|w\|_1 = \sum_{j=1}^{n}|w_j|
$$

Like Ridge, Lasso penalizes the weights but usually does not penalize the bias.

---

### Lasso intuition

Lasso encourages sparsity.

This means it can push some weights exactly to zero.

A zero weight means the corresponding feature is effectively removed from the model.

So Lasso can be useful for:

```txt
feature selection
sparse models
high-dimensional datasets
interpretable models with fewer active features
```

This is the main conceptual difference:

```txt
Ridge:
    shrinks weights toward zero

Lasso:
    can shrink weights exactly to zero
```

---

### Why Lasso is harder to optimize

The L1 penalty uses absolute values:

$$
|w_j|
$$

The absolute value function is not differentiable at zero.

That means ordinary gradient descent is not as directly clean as with Ridge.

Lasso is usually optimized with methods such as:

```txt
coordinate descent
subgradient methods
proximal gradient methods
```

Because Phase 3 is focused on linear regression and Ridge, ML Core will treat Lasso conceptually for now.

A full Lasso implementation is deferred.

---

### Lasso implementation decision in ML Core

For Phase 3:

```txt
Ridge:
    implement in detail

Lasso:
    document conceptually
    do not implement yet
```

This keeps the scope controlled.

Lasso may be revisited later after the optimization phase introduces more appropriate tools.

---

## 8. Residual Behavior and Interpretation

Residuals are the prediction errors made by a model.

For regression, the residual vector is:

$$
r = \hat{y} - y
$$

where:

```txt
r = residual vector
ŷ = prediction vector
y = target vector
```

For one sample:

$$
r_i = \hat{y}_i - y_i
$$

Interpretation:

```txt
r_i > 0:
    the model overpredicted sample i

r_i < 0:
    the model underpredicted sample i

r_i = 0:
    the model predicted sample i exactly
```

Residual analysis helps us understand not only how large the model error is, but also whether the model is making systematic mistakes.

---

### Why residuals matter

Metrics such as MSE, RMSE, MAE, and R² summarize model performance into a few numbers.

Those metrics are useful, but they can hide important behavior.

Two models can have similar MSE while having very different residual patterns.

Residual analysis helps answer questions such as:

```txt
Are errors small and randomly distributed?

Does the model systematically overpredict?

Does the model systematically underpredict?

Are errors larger for certain target ranges?

Are there outliers with very large errors?

Is there evidence that the linear model is missing a nonlinear pattern?
```

A model with random-looking residuals is usually healthier than a model with structured residuals.

---

### Good residual behavior

For a well-behaved linear regression model, residuals should ideally look like noise around zero.

That means:

```txt
mean residual close to 0
no strong trend in residuals
no obvious systematic overprediction or underprediction
no clear pattern with respect to target values or features
few extreme outliers
```

In simple terms:

```txt
the model should be wrong in small, unsystematic ways
```

This does not mean every residual must be tiny.

It means the remaining errors should not reveal a clear structure that the model failed to capture.

---

### Problematic residual behavior

Residuals can reveal problems that metrics alone may hide.

#### Systematic overprediction

If most residuals are positive:

```txt
ŷ - y > 0
```

then the model tends to predict values that are too high.

This may indicate:

```txt
bias/intercept issue
training distribution mismatch
missing explanatory features
poor model fit
```

#### Systematic underprediction

If most residuals are negative:

```txt
ŷ - y < 0
```

then the model tends to predict values that are too low.

This may indicate similar issues:

```txt
bias/intercept issue
missing features
non-representative training data
incorrect preprocessing
```

#### Residual trend

If residuals increase or decrease with the target value or with one feature, the model may be missing a nonlinear relationship.

Example:

```txt
small targets:
    residuals mostly positive

large targets:
    residuals mostly negative
```

This suggests the model may be compressing predictions toward the center.

#### Large outliers

A few very large residuals can dominate MSE and RMSE.

This matters because squared-error metrics penalize large errors strongly.

Outliers may come from:

```txt
bad data
rare cases
missing features
true nonlinear behavior
measurement noise
```

---

### Residual summary statistics

A minimal residual analysis can report:

```txt
mean residual
mean absolute residual
minimum residual
maximum residual
largest absolute residual
```

Useful interpretations:

```txt
mean residual:
    detects average overprediction or underprediction

mean absolute residual:
    gives average error magnitude

minimum residual:
    largest underprediction

maximum residual:
    largest overprediction

largest absolute residual:
    worst individual error
```

For a fitted model, ML Core can export a small residual table:

```txt
sample_index
target
prediction
residual
absolute_residual
```

This table helps inspect concrete prediction errors.

---

### Residuals and linearity

Linear regression assumes the target can be approximated as:

```txt
weighted sum of features + bias
```

If residuals show structure, the linear model may not be expressive enough.

Possible signs:

```txt
curved residual pattern
errors grow as target values grow
errors depend strongly on one feature
clusters of positive and negative residuals
```

Potential responses include:

```txt
add better features
scale features correctly
try polynomial or interaction features
try regularization
use a different model family later
```

For Phase 3, the goal is not to solve all residual problems.

The goal is to learn how to inspect them and connect residual patterns to model behavior.

---

### Residuals and Ridge regression

Ridge regularization can change residual behavior.

Because Ridge shrinks weights, it may:

```txt
increase training error slightly
reduce unstable weight values
improve validation/test behavior
reduce sensitivity to correlated features
```

On training data, unregularized linear regression may produce smaller residuals than Ridge.

That does not automatically mean it is better.

The real question is whether Ridge improves generalization on validation or test data.

Residual analysis should therefore be interpreted together with:

```txt
train/test split
baseline comparison
MSE/RMSE/MAE/R²
weight magnitudes
loss history
```

---

### Residual interpretation in ML Core

For Phase 3, residual behavior should be inspected using small controlled experiments.

A minimal residual analysis should:

```txt
1. train a LinearRegression model
2. generate predictions
3. compute residuals = predictions - targets
4. export a residual table
5. export summary statistics
6. write a short interpretation note
```

The output should help answer:

```txt
Are residuals small?

Are residuals centered around zero?

Does the model mostly overpredict or underpredict?

Are there unusually large errors?

Does the result make sense for the synthetic dataset?
```

This completes the connection between linear model theory, metrics, and concrete prediction errors.