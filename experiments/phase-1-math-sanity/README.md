# Phase 1 Math Sanity Notes

This folder contains the concise validation/demo layer for Phase 1.

The goal is not to add new reusable code. The goal is to connect the Phase 1 theory and utilities to small, concrete ML examples.

This sanity layer covers:

```txt
1. vectorized expressions
2. gradient interpretation
3. variance / covariance intuition
```

---

## 1. Vectorized Expressions

ML Core uses matrix and vector operations instead of scalar-first implementations.

For a supervised dataset:

```txt
X: m x n matrix
y: m-dimensional vector
w: n-dimensional weight vector
b: scalar bias
```

a linear model is written as:

$$
\hat{y} = Xw + b
$$

This means:

```txt
one matrix-vector operation produces all predictions
```

Instead of computing each prediction manually:

```txt
prediction_1 = x_1^T w + b
prediction_2 = x_2^T w + b
...
prediction_m = x_m^T w + b
```

we compute all predictions at once:

```cpp
ml::Vector predictions = ml::linear_prediction(X, weights, bias);
```

This is important because later models should operate at dataset level, not one sample at a time.

---

## 2. Example: Matrix-Vector Prediction

Example dataset:

$$
X =
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$

Weights:

$$
w =
\begin{bmatrix}
0.5 \\
1.0
\end{bmatrix}
$$

Bias:

$$
b = 2.0
$$

Matrix-vector product:

$$
Xw =
\begin{bmatrix}
1(0.5) + 2(1.0) \\
3(0.5) + 4(1.0) \\
5(0.5) + 6(1.0)
\end{bmatrix}
=
\begin{bmatrix}
2.5 \\
5.5 \\
8.5
\end{bmatrix}
$$

Adding the bias:

$$
\hat{y} = Xw + b =
\begin{bmatrix}
4.5 \\
7.5 \\
10.5
\end{bmatrix}
$$

This is exactly the behavior validated by the Phase 1.2 math operation tests.

---

## 3. Residuals and Loss

Given targets:

$$
y =
\begin{bmatrix}
5.0 \\
7.0 \\
11.0
\end{bmatrix}
$$

and predictions:

$$
\hat{y} =
\begin{bmatrix}
4.5 \\
7.5 \\
10.5
\end{bmatrix}
$$

the residual vector is:

$$
r = \hat{y} - y =
\begin{bmatrix}
-0.5 \\
0.5 \\
-0.5
\end{bmatrix}
$$

The mean squared error is:

$$
MSE = \frac{1}{m} r^\top r
$$

So:

$$
MSE = \frac{1}{3}(0.25 + 0.25 + 0.25) = 0.25
$$

This connects directly to:

```cpp
ml::residuals(predictions, targets);
ml::mean_squared_error(predictions, targets);
```

---

## 4. Gradient Interpretation

For linear regression with mean squared error:

$$
\hat{y} = Xw + b
$$

$$
r = \hat{y} - y
$$

$$
L(w, b) = \frac{1}{m} r^\top r
$$

The gradient with respect to the weights is:

$$
\nabla_w L = \frac{2}{m}X^\top r
$$

The gradient with respect to the bias is:

$$
\frac{\partial L}{\partial b} = \frac{2}{m}\sum_{i=1}^{m}r_i
$$

Interpretation:

```txt
r tells us the prediction errors.
Xᵀr tells us how each feature is correlated with those errors.
```

If one gradient component is positive, increasing that weight increases the loss locally.

If one gradient component is negative, increasing that weight decreases the loss locally.

Gradient descent updates parameters in the opposite direction:

$$
w \leftarrow w - \alpha \nabla_w L
$$

This is the bridge between Phase 1 math and Phase 3/5 model training.

---

## 5. Gradient Example

Using:

$$
X =
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$

and:

$$
r =
\begin{bmatrix}
-0.5 \\
0.5 \\
-0.5
\end{bmatrix}
$$

First compute:

$$
X^\top r =
\begin{bmatrix}
1 & 3 & 5 \\
2 & 4 & 6
\end{bmatrix}
\begin{bmatrix}
-0.5 \\
0.5 \\
-0.5
\end{bmatrix}
$$

Feature 1 contribution:

$$
1(-0.5) + 3(0.5) + 5(-0.5) = -1.5
$$

Feature 2 contribution:

$$
2(-0.5) + 4(0.5) + 6(-0.5) = -2.0
$$

So:

$$
X^\top r =
\begin{bmatrix}
-1.5 \\
-2.0
\end{bmatrix}
$$

With $m = 3$:

$$
\nabla_w L =
\frac{2}{3}
\begin{bmatrix}
-1.5 \\
-2.0
\end{bmatrix}
=
\begin{bmatrix}
-1.0 \\
-1.3333
\end{bmatrix}
$$

The negative gradient means increasing both weights would locally reduce the loss.

---

## 6. Variance Intuition

Variance measures spread around the mean.

For:

$$
x =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

the mean is:

$$
\mu = 2
$$

Centered values:

$$
x - \mu =
\begin{bmatrix}
-1 \\
0 \\
1
\end{bmatrix}
$$

Population variance:

$$
\sigma^2 = \frac{1}{3}(1 + 0 + 1) = \frac{2}{3}
$$

Population standard deviation:

$$
\sigma = \sqrt{\frac{2}{3}}
$$

This is why standardizing the vector gives:

$$
z =
\begin{bmatrix}
-1.2247 \\
0 \\
1.2247
\end{bmatrix}
$$

Phase 1.4 validates this through `standardize_columns`.

---

## 7. Covariance Intuition

Covariance measures whether two variables move together.

Given two feature columns:

$$
x =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

and:

$$
y =
\begin{bmatrix}
10 \\
20 \\
30
\end{bmatrix}
$$

Both increase together.

Their centered values are:

$$
x - \bar{x} =
\begin{bmatrix}
-1 \\
0 \\
1
\end{bmatrix}
$$

and:

$$
y - \bar{y} =
\begin{bmatrix}
-10 \\
0 \\
10
\end{bmatrix}
$$

The element-wise products are:

$$
\begin{bmatrix}
10 \\
0 \\
10
\end{bmatrix}
$$

The average product is positive, so covariance is positive.

Interpretation:

```txt
positive covariance:
    features tend to move together

negative covariance:
    one feature tends to increase while the other decreases

near-zero covariance:
    no strong linear co-movement
```

Covariance will become important later for PCA, probabilistic models, and finance-oriented datasets.

---

## 8. What This Sanity Layer Confirms

This sanity layer confirms that Phase 1 gives us the necessary foundation for the next phases.

We now have:

```txt
vectorized dataset representation
matrix-vector prediction
residual computation
MSE computation
gradient interpretation
mean / variance / standard deviation
feature scaling
basic covariance intuition
```

The corresponding reusable utilities are implemented in:

```txt
include/ml/common/
src/common/
```

The corresponding tests are run through:

```bash
./build/ml_core_tests
```

---

## 9. Phase 1 Exit Check

Phase 1 is complete when:

```txt
- the theory docs exist
- the sanity layer exists
- the common utilities compile
- ml_core_tests pass
- the remaining Phase 1 action-plan checkboxes are marked correctly
```

At this point, the project is ready to move to Phase 2.

Phase 2 should focus on:

```txt
dataset abstractions
train / validation / test splitting
evaluation discipline
data leakage prevention
baseline-vs-model evaluation
```