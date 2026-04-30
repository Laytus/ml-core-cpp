# Eigen Quick Reference for ML Core

This document summarizes the Eigen features used in ML Core.

ML Core uses Eigen for linear algebra. We do not build our own matrix library.

The project aliases are defined in:

```cpp
#include "ml/common/types.hpp"
```

```cpp
namespace ml {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

} // namespace ml
```

So, in ML Core, prefer:

```cpp
ml::Matrix X;
ml::Vector y;
```

instead of writing Eigen types directly everywhere.

---

## 1. Core Types

### Matrix

```cpp
ml::Matrix X(3, 2);
```

This creates a matrix with:

```txt
3 rows
2 columns
```

In ML Core convention:

```txt
rows    = samples
columns = features
```

So this represents 3 samples with 2 features each.

### Vector

```cpp
ml::Vector y(3);
```

This creates a vector with 3 values.

In ML Core, vectors are used for:

```txt
targets
predictions
weights
gradients
residuals
statistics
```

---

## 2. Creating Matrices and Vectors

### Matrix initialization

```cpp
ml::Matrix X(3, 2);

X << 1.0, 2.0,
     3.0, 4.0,
     5.0, 6.0;
```

This creates:

$$
X =
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$

### Vector initialization

```cpp
ml::Vector y(3);

y << 10.0, 20.0, 30.0;
```

This creates:

$$
y =
\begin{bmatrix}
10 \\
20 \\
30
\end{bmatrix}
$$

---

## 3. Shape Information

### Number of rows

```cpp
X.rows();
```

### Number of columns

```cpp
X.cols();
```

### Vector size

```cpp
y.size();
```

### Total number of matrix elements

```cpp
X.size();
```

For a `3 x 2` matrix, this returns `6`.

---

## 4. Accessing Elements

Eigen uses zero-based indexing.

### Matrix element access

```cpp
double value = X(0, 1);
```

This means:

```txt
row 0, column 1
```

### Vector element access

```cpp
double value = y(0);
```

or:

```cpp
double value = y[0];
```

For consistency in mathematical code, prefer:

```cpp
y(i);
```

---

## 5. Rows and Columns

### Get one row

```cpp
auto row = X.row(0);
```

### Get one column

```cpp
auto col = X.col(1);
```

### Modify one row

```cpp
X.row(0) << 7.0, 8.0;
```

### Modify one column

```cpp
X.col(0) << 1.0, 2.0, 3.0;
```

---

## 6. Matrix and Vector Operations

### Matrix-vector multiplication

```cpp
ml::Vector predictions = X * weights;
```

If:

```txt
X       = m x n
weights = n
```

then:

```txt
X * weights = m
```

This is the base operation for linear models.

### Dot product

```cpp
double result = a.dot(b);
```

Mathematically:

$$
a^\top b
$$

### Vector norm

```cpp
double norm = a.norm();
```

This computes the Euclidean norm:

$$
\|a\|_2
$$

### Squared norm

```cpp
double squared_norm = a.squaredNorm();
```

This computes:

$$
a^\top a
$$

This is useful for squared error losses.

### Transpose

```cpp
ml::Matrix Xt = X.transpose();
```

If:

```txt
X = m x n
```

then:

```txt
X.transpose() = n x m
```

A common ML expression is:

```cpp
ml::Vector gradient = X.transpose() * residuals;
```

If:

```txt
X         = m x n
residuals = m
```

then:

```txt
X.transpose() * residuals = n
```

---

## 7. Element-wise Operations with `.array()`

Eigen separates:

```txt
.matrix() mode = linear algebra operations
.array() mode  = element-wise operations
```

### Add scalar to every element

```cpp
predictions.array() += bias;
```

This is useful for:

```cpp
ml::Vector predictions = X * weights;
predictions.array() += bias;
```

### Element-wise square

```cpp
ml::Vector squared = residuals.array().square();
```

### Element-wise multiplication

```cpp
ml::Vector result = a.array() * b.array();
```

### Element-wise division

```cpp
ml::Vector result = a.array() / b.array();
```

### Convert array expression back to matrix/vector mode

```cpp
ml::Vector result = some_array_expression.matrix();
```

Example:

```cpp
ml::Vector squared = residuals.array().square().matrix();
```

In many cases Eigen can infer the conversion, but using `.matrix()` can make the intent explicit.

---

## 8. Coefficient-wise Matrix Operations

For element-wise operations between matrices or vectors, Eigen also provides `cwise...` methods.

### Coefficient-wise division

```cpp
ml::Matrix result = A.cwiseQuotient(B);
```

This computes:

```txt
result(i, j) = A(i, j) / B(i, j)
```

### Coefficient-wise product

```cpp
ml::Matrix result = A.cwiseProduct(B);
```

This computes:

```txt
result(i, j) = A(i, j) * B(i, j)
```

These are useful when operating on two matrices of the same shape.

---

## 9. Reductions

### Sum

```cpp
double total = v.sum();
```

### Mean

```cpp
double mean = v.mean();
```

### Minimum

```cpp
double min_value = v.minCoeff();
```

### Maximum

```cpp
double max_value = v.maxCoeff();
```

These are useful for statistics and metrics.

---

## 10. Column-wise Operations

ML Core uses columns as features, so column-wise operations are important.

### Column means

```cpp
ml::Vector means = X.colwise().mean().transpose();
```

`X.colwise().mean()` gives one value per column. Transpose it into a column vector if you want to store it as `ml::Vector`.

### Column minimum values

```cpp
ml::Vector mins = X.colwise().minCoeff().transpose();
```

### Column maximum values

```cpp
ml::Vector maxs = X.colwise().maxCoeff().transpose();
```

### Center a matrix column-wise

If `means` is an `ml::Vector` of size `X.cols()`:

```cpp
ml::Matrix centered = X.rowwise() - means.transpose();
```

This subtracts the feature mean from every row.

### Divide each column by a scale vector

If `scales` is an `ml::Vector` of size `X.cols()`:

```cpp
ml::Matrix scaled =
    (centered.array().rowwise() / scales.transpose().array()).matrix();
```

This divides each column by the corresponding scale value.

This pattern is useful for standardization.

---

## 11. Creating Special Matrices and Vectors

### Zero vector

```cpp
ml::Vector v = ml::Vector::Zero(5);
```

### Ones vector

```cpp
ml::Vector v = ml::Vector::Ones(5);
```

### Constant vector

```cpp
ml::Vector v = ml::Vector::Constant(5, 3.14);
```

### Zero matrix

```cpp
ml::Matrix X = ml::Matrix::Zero(3, 2);
```

### Ones matrix

```cpp
ml::Matrix X = ml::Matrix::Ones(3, 2);
```

### Constant matrix

```cpp
ml::Matrix X = ml::Matrix::Constant(3, 2, 3.14);
```

### Identity matrix

```cpp
ml::Matrix I = ml::Matrix::Identity(3, 3);
```

---

## 12. Resizing

### Resize vector

```cpp
ml::Vector v;
v.resize(5);
```

### Resize matrix

```cpp
ml::Matrix X;
X.resize(10, 3);
```

Warning:

```txt
resize() does not necessarily preserve existing values.
```

Prefer constructing with the correct shape when possible.

---

## 13. Common ML Patterns

### Linear prediction

```cpp
ml::Vector predictions = X * weights;
predictions.array() += bias;
```

Mathematically:

$$
\hat{y} = Xw + b
$$

### Residuals

```cpp
ml::Vector residuals = predictions - y;
```

Mathematically:

$$
r = \hat{y} - y
$$

### Mean squared error

```cpp
double mse = residuals.squaredNorm() / static_cast<double>(residuals.size());
```

Mathematically:

$$
MSE = \frac{1}{m} \sum_{i=1}^{m}(\hat{y}_i - y_i)^2
$$

### Linear regression gradient

```cpp
ml::Vector gradient_w =
    (2.0 / static_cast<double>(X.rows())) * X.transpose() * residuals;

double gradient_b =
    (2.0 / static_cast<double>(X.rows())) * residuals.sum();
```

### Feature standardization

```cpp
ml::Vector means = X.colwise().mean().transpose();
ml::Vector stds = /* one standard deviation per column */;

ml::Matrix centered = X.rowwise() - means.transpose();

ml::Matrix standardized =
    (centered.array().rowwise() / stds.transpose().array()).matrix();
```

### Min-max normalization

```cpp
ml::Vector mins = X.colwise().minCoeff().transpose();
ml::Vector maxs = X.colwise().maxCoeff().transpose();
ml::Vector ranges = maxs - mins;

ml::Matrix centered = X.rowwise() - mins.transpose();

ml::Matrix normalized =
    (centered.array().rowwise() / ranges.transpose().array()).matrix();
```

For zero standard deviations or zero ranges, replace the scale value with `1.0` before dividing. If the column is constant, centering already produces zeros.

---

## 14. Passing Eigen Objects to Functions

Prefer passing matrices and vectors by const reference when the function does not modify them:

```cpp
double mean_squared_error(
    const ml::Vector& y_true,
    const ml::Vector& y_pred
);
```

Use non-const reference only when the function modifies the object:

```cpp
void normalize_in_place(ml::Matrix& X);
```

Return by value when producing a new vector or matrix:

```cpp
ml::Vector predict(const ml::Matrix& X);
```

Eigen is optimized for this style.

---

## 15. Important ML Core Convention

ML Core uses this convention everywhere:

```txt
X: m x n matrix
m: number of samples
n: number of features

y: m-dimensional vector
weights: n-dimensional vector
predictions: m-dimensional vector
residuals: m-dimensional vector
```

So for a linear model:

```cpp
ml::Vector predictions = X * weights;
predictions.array() += bias;
```

requires:

```txt
X.cols() == weights.size()
X.rows() == predictions.size()
X.rows() == y.size()
```

These assumptions should be checked using shape validation helpers.

---

## 16. Recommended Include Pattern

In headers:

```cpp
#include "ml/common/types.hpp"
```

In implementation files:

```cpp
#include "ml/some_module/some_file.hpp"
```

Avoid including Eigen directly across the codebase unless necessary.

Prefer using ML Core aliases:

```cpp
ml::Matrix
ml::Vector
```

---

## 17. Common Mistakes to Avoid

### Mistake 1: Wrong multiplication shape

Invalid:

```cpp
ml::Vector result = X * y;
```

unless:

```txt
X.cols() == y.size()
```

### Mistake 2: Trying to add scalar directly to vector

Avoid:

```cpp
predictions += bias;
```

Prefer:

```cpp
predictions.array() += bias;
```

### Mistake 3: Confusing rows and columns

Remember:

```txt
rows = samples
columns = features
```

### Mistake 4: Forgetting zero-based indexing

```cpp
X(0, 0)
```

is the first row and first column.

### Mistake 5: Confusing column vectors and row vectors

`ml::Vector` is a column vector.

For row-wise broadcasting, transpose it:

```cpp
X.rowwise() - means.transpose();
```

### Mistake 6: Mixing `.array()` and `.matrix()` modes accidentally

If you use `.array()` for element-wise operations, convert back with `.matrix()` when returning or assigning to a matrix/vector if needed:

```cpp
return some_array_expression.matrix();
```

### Mistake 7: Not validating shapes before model operations

Before fitting or predicting, validate:

```cpp
validate_same_number_of_rows(X, y, "Model::fit");
validate_feature_count(X, weights, "Model::predict");
```

---

## 18. Quick Example

```cpp
#include "ml/common/types.hpp"

#include <iostream>

int main() {
    ml::Matrix X(3, 2);

    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector weights(2);
    weights << 0.5, 1.0;

    double bias = 2.0;

    ml::Vector predictions = X * weights;
    predictions.array() += bias;

    std::cout << predictions << "\n";

    return 0;
}
```

Manual calculation:

```txt
sample 1: 1.0 * 0.5 + 2.0 * 1.0 + 2.0 = 4.5
sample 2: 3.0 * 0.5 + 4.0 * 1.0 + 2.0 = 7.5
sample 3: 5.0 * 0.5 + 6.0 * 1.0 + 2.0 = 10.5
```

Output:

```txt
4.5
7.5
10.5
```