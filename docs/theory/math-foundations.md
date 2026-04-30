# Math Foundations

This document contains the mathematical foundations used throughout ML Core.

ML Core uses Eigen for linear algebra, but the project still defines its own mathematical conventions so that all models, losses, metrics, preprocessing utilities, and optimization routines use the same interpretation of vectors and matrices.

---

## 1. Vector and Matrix Representation in ML Core

Machine Learning algorithms operate on structured numerical data.

In ML Core, we represent data using Eigen matrices and vectors through the project aliases defined in:

```cpp
#include "ml/common/types.hpp"
```

```cpp
namespace ml {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

} // namespace ml
```

The main dataset convention is:

$$
X \in \mathbb{R}^{m \times n}
$$

where:

- $m$ is the number of samples
- $n$ is the number of features
- each row of $X$ is one training example
- each column of $X$ is one feature

A dataset with $m$ samples and $n$ features is represented as:

$$
X =
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1n} \\
x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \cdots & x_{mn}
\end{bmatrix}
$$

The target vector is represented as:

$$
y \in \mathbb{R}^{m}
$$

or:

$$
y =
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{bmatrix}
$$

This convention is used throughout the project.

---

## 2. Feature Vectors

A single sample is represented as a feature vector:

$$
x_i \in \mathbb{R}^{n}
$$

For example, if a sample has three features:

$$
x_i =
\begin{bmatrix}
x_{i1} \\
x_{i2} \\
x_{i3}
\end{bmatrix}
$$

In the design matrix $X$, this same sample appears as one row:

$$
X_i =
\begin{bmatrix}
x_{i1} & x_{i2} & x_{i3}
\end{bmatrix}
$$

This means ML Core follows a row-wise dataset convention:

```txt
rows    = samples
columns = features
```

Even though Eigen handles the internal memory layout, the mathematical convention remains fixed.

This convention matters because many ML formulas are written in matrix form. For example, if $X$ contains all samples and $w$ contains one coefficient per feature, then $Xw$ produces one prediction per sample.

---

## 3. Parameter Vectors

Many ML models use a parameter vector:

$$
w \in \mathbb{R}^{n}
$$

where each coefficient corresponds to one feature.

For a linear model, predictions are computed as:

$$
\hat{y} = Xw + b
$$

where:

- $X \in \mathbb{R}^{m \times n}$
- $w \in \mathbb{R}^{n}$
- $b \in \mathbb{R}$
- $\hat{y} \in \mathbb{R}^{m}$

The shape compatibility is essential:

$$
(m \times n)(n \times 1) = (m \times 1)
$$

So the number of columns in $X$ must match the number of elements in $w$.

In code, this means:

```cpp
X.cols() == weights.size()
```

A valid prediction operation has this form:

```cpp
ml::Vector predictions = X * weights;
predictions.array() += bias;
```

The scalar bias is added element-wise to every prediction, which is why Eigen's `.array()` mode is used.

---

## 4. Shape Conventions

ML Core uses the following shape conventions by default:

```txt
X           : m x n matrix
y           : m-dimensional vector
weights     : n-dimensional vector
predictions : m-dimensional vector
residuals   : m-dimensional vector
gradients   : usually n-dimensional for weights, scalar for bias
```

These conventions will be reused in:

- linear regression
- logistic regression
- loss functions
- evaluation metrics
- optimization routines
- dimensionality reduction
- probabilistic models

The most important compatibility rules are:

```txt
X.rows() == y.size()
X.cols() == weights.size()
predictions.size() == y.size()
residuals.size() == y.size()
```

Violating these assumptions usually means there is a dataset, model, or implementation error.

---

## 5. Why Shape Validation Matters

Many ML errors are not syntax errors.

They are shape errors.

For example:

- using a target vector with the wrong number of samples
- using a weight vector with the wrong number of features
- passing an empty dataset into training
- computing metrics between vectors of different sizes

These errors should be caught explicitly.

Therefore, ML Core includes reusable validation helpers for common shape checks.

These helpers are not meant to replace Eigen. They exist to make ML-specific assumptions explicit and to provide readable error messages.

The shape validation helpers live in:

```txt
include/ml/common/shape_validation.hpp
src/common/shape_validation.cpp
```

The current validation functions include:

```cpp
validate_same_number_of_rows(X, y, context);
validate_same_size(a, b, context);
validate_feature_count(X, weights, context);
validate_non_empty_matrix(X, context);
validate_non_empty_vector(v, context);
validate_min_vector_size(v, min_size, context);
validate_min_matrix_rows(X, min_rows, context);
```

The `context` argument identifies where the validation happened.

Example:

```cpp
validate_same_number_of_rows(X, y, "LinearRegression::fit");
```

If the validation fails, the error message should be explicit, for example:

```txt
LinearRegression::fit: X rows must match y size. Got X.rows() = 100, y.size() = 90.
```

This makes debugging faster and prevents silent mathematical mistakes.

---

## 6. Matrix Multiplication and Geometric Interpretation

Matrix-vector multiplication is one of the central operations in Machine Learning.

For a dataset:

$$
X \in \mathbb{R}^{m \times n}
$$

and a parameter vector:

$$
w \in \mathbb{R}^{n}
$$

the product:

$$
Xw
$$

produces a vector in:

$$
\mathbb{R}^{m}
$$

Each element of the result is the dot product between one sample row and the parameter vector:

$$
(Xw)_i = x_i^\top w
$$

where:

- $x_i$ is the feature vector for sample $i$
- $w$ is the weight vector
- $x_i^\top w$ is a scalar score

For a linear model, predictions are computed as:

$$
\hat{y} = Xw + b
$$

This means each prediction is a weighted combination of input features plus a bias term.

### Dot Product Interpretation

The dot product between two vectors is:

$$
a^\top b = \sum_{j=1}^{n} a_j b_j
$$

The dot product measures alignment between vectors.

If two vectors point in a similar direction, their dot product is large and positive. If they point in opposite directions, their dot product is negative. If they are orthogonal, their dot product is zero.

This interpretation is important for:

- linear models
- cosine similarity
- nearest-neighbor methods
- projections
- kernels
- neural network layers

### Linear Combination Interpretation

The product $Xw$ can also be interpreted as a linear combination of the columns of $X$:

$$
Xw = w_1 X_{:,1} + w_2 X_{:,2} + \cdots + w_n X_{:,n}
$$

Each feature column contributes to the prediction vector according to its weight.

This is the core idea behind linear models: the output is constructed by weighting and combining input features.

### Geometric Interpretation

For a single sample $x_i$, the value:

$$
x_i^\top w
$$

is the projection-like score of the sample along the direction defined by $w$.

In binary classification, the sign of this score can define which side of a decision boundary the sample lies on.

In regression, this score defines the predicted continuous value before adding the bias.

Therefore, matrix multiplication is not just a computational operation. It is the mathematical bridge between raw features and model predictions.

---

## 7. Descriptive Statistics

Descriptive statistics summarize numerical data.

In Machine Learning, they are used to understand datasets, preprocess features, detect scale problems, and compute evaluation metrics.

For a vector:

$$
x =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_m
\end{bmatrix}
$$

the mean is:

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

The mean describes the central value of the data.

The variance measures average squared deviation from the mean.

For a population variance:

$$
\sigma^2 = \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu)^2
$$

For a sample variance:

$$
s^2 = \frac{1}{m - 1} \sum_{i=1}^{m}(x_i - \mu)^2
$$

The standard deviation is the square root of the variance:

$$
\sigma = \sqrt{\sigma^2}
$$

These quantities are important because many ML algorithms are sensitive to feature scale.

### Feature-wise Statistics

For a dataset:

$$
X \in \mathbb{R}^{m \times n}
$$

we often compute statistics column by column.

The mean of feature $j$ is:

$$
\mu_j = \frac{1}{m} \sum_{i=1}^{m} X_{ij}
$$

The result is a vector:

$$
\mu \in \mathbb{R}^{n}
$$

where each entry corresponds to one feature.

Feature-wise statistics are used for:

- standardization
- normalization
- exploratory data analysis
- outlier detection
- covariance computation
- PCA

ML Core uses column-wise statistics by default because columns represent features.

The current statistics utilities include:

```cpp
mean(values);
variance_population(values);
variance_sample(values);
standard_deviation_population(values);
standard_deviation_sample(values);
column_means(X);
column_variance_population(X);
column_variance_sample(X);
column_standard_deviation_population(X);
column_standard_deviation_sample(X);
```

---

## 8. Normalization and Standardization

Machine Learning algorithms are often sensitive to feature scale.

If one feature has values around $1$ and another feature has values around $100000$, optimization algorithms such as gradient descent may behave poorly.

Feature scaling makes numerical features more comparable.

### Standardization

Standardization transforms a feature so that it has approximately zero mean and unit variance.

For a feature column $x_j$, the standardized value is:

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

where:

- $x_{ij}$ is the value of feature $j$ for sample $i$
- $\mu_j$ is the mean of feature $j$
- $\sigma_j$ is the standard deviation of feature $j$

After standardization, each feature usually has:

$$
\mu \approx 0
$$

and:

$$
\sigma \approx 1
$$

This is useful for:

- gradient descent
- linear models
- logistic regression
- PCA
- distance-based methods

In ML Core, the function responsible for this transformation is:

```cpp
Matrix standardize_columns(const Matrix& X);
```

It standardizes each feature column independently.

### Min-Max Normalization

Min-max normalization rescales a feature to a fixed range, usually $[0, 1]$.

For a feature column $x_j$:

$$
x'_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)}
$$

This preserves the relative ordering of values while changing the scale.

In ML Core, the function responsible for this transformation is:

```cpp
Matrix normalize_min_max_columns(const Matrix& X);
```

It normalizes each feature column independently.

### Zero-Variance and Constant Features

A feature has zero variance when all its values are identical.

For example:

$$
x_j =
\begin{bmatrix}
5 \\
5 \\
5
\end{bmatrix}
$$

In this case, the standard deviation is zero, so standardization would require division by zero.

Similarly, min-max normalization would have a zero denominator because:

$$
\max(x_j) - \min(x_j) = 0
$$

ML Core handles these columns explicitly.

For this project, the default rule is:

```txt
If a feature has zero variance or zero range, its scaled values are set to 0.
```

This is equivalent to saying that the feature carries no variation across samples and should not affect scaled computations.

---

## 9. Current Phase 1 Implementation Scope

Phase 1 establishes the mathematical and statistical utilities required before implementing complete ML models.

It includes:

- project-level Eigen aliases
- dataset shape conventions
- reusable shape validation helpers
- matrix/vector math operations
- descriptive statistics
- feature standardization and min-max normalization
- a reusable `ml_core` CMake library target

It does not implement:

- a custom matrix library
- a complete data pipeline
- model training abstractions
- train/test splitting
- cross-validation
- optimization algorithms

Those will be added later in dependency order.

The purpose of Phase 1 is to make sure every future ML component starts from a consistent mathematical and statistical foundation.