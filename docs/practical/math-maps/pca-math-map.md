# PCA Math Map

## Purpose

This document maps the public `PCA` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/unsupervised/pca.hpp
src/unsupervised/pca.cpp
include/ml/common/types.hpp
```

Related practical usage doc:

```txt
docs/practical/models/pca-usage.md
```

Related theory doc:

```txt
docs/theory/unsupervised-learning.md
```

## Model idea

`PCA` means:

```txt
Principal Component Analysis
```

PCA is an unsupervised dimensionality-reduction method.

It finds orthogonal directions in the feature space that capture the largest variance in the data.

Given a data matrix:

```txt
X
```

PCA learns:

```txt
mean vector
principal components
explained variance
explained variance ratio
```

Then it transforms samples into a lower-dimensional representation:

```txt
Z = centered_X * components
```

where:

```txt
centered_X = X - mean
```

If `num_components = 2`, PCA produces a 2D representation that can be used for visualization.

## Core PCA objective

PCA searches for directions that maximize projected variance.

The first principal component is the direction:

```txt
v_1
```

that maximizes:

```txt
variance(X_centered * v_1)
```

subject to:

```txt
||v_1|| = 1
```

The second component maximizes remaining variance subject to being orthogonal to the first:

```txt
v_2 ⟂ v_1
```

More generally, the principal components are orthogonal directions:

```txt
v_1, v_2, ..., v_k
```

ordered by decreasing explained variance.

## Matrix formulation

Given:

```txt
X: n_samples x n_features
```

Compute the feature-wise mean:

```txt
mean_j = (1 / n) * sum_i X_ij
```

Center the data:

```txt
X_centered = X - mean
```

The covariance matrix is:

```txt
C = (1 / (n - 1)) * X_centered^T * X_centered
```

PCA finds eigenvectors of the covariance matrix:

```txt
C v_j = λ_j v_j
```

where:

```txt
v_j = principal component direction
λ_j = explained variance for that component
```

Components are sorted by decreasing eigenvalue:

```txt
λ_1 >= λ_2 >= ... >= λ_d
```

The top `num_components` eigenvectors form the component matrix.

## Projection equation

After fitting, PCA transforms data with:

```txt
Z = (X - mean) * components
```

where:

```txt
Z = projected low-dimensional representation
components = selected principal directions
```

Expected shape:

```txt
X: n_samples x n_features
components: n_features x num_components
Z: n_samples x num_components
```

## Reconstruction equation

PCA can approximately reconstruct original-space data from low-dimensional coordinates:

```txt
X_reconstructed = Z * components^T + mean
```

If fewer components are retained than original features, reconstruction is approximate.

If all components are retained, reconstruction can be exact up to numerical precision.

## Public API to math mapping

## `PCAOptions`

```cpp
struct PCAOptions {
    std::size_t num_components;
};
```

### Mathematical role

`PCAOptions` controls how many principal directions are retained.

Important field:

```txt
num_components
```

Math meaning:

```txt
num_components:
  number of eigenvectors / principal directions kept
```

This determines the dimensionality of the transformed representation:

```txt
Z.cols() == num_components
```

Larger `num_components` preserves more variance.

Smaller `num_components` gives stronger compression.

## `validate_pca_options`

```cpp
void validate_pca_options(
    const PCAOptions& options,
    const std::string& context
);
```

### Mathematical role

Ensures the PCA configuration is meaningful.

Typical mathematical requirements:

```txt
num_components >= 1
```

During fitting, there is also a practical constraint:

```txt
num_components <= number_of_features
```

A model cannot retain zero components, and it cannot retain more components than feature dimensions.

### Infrastructure role

It also provides:

```txt
consistent error messages
context-specific validation
early failure for invalid configuration
```

## Constructor

```cpp
PCA();
explicit PCA(PCAOptions options);
```

### Mathematical role

The constructor does not compute principal components.

It only stores the configuration:

```txt
num_components
```

Math implemented:

```txt
none directly
```

Infrastructure role:

```txt
configure retained dimensionality
validate options
```

## `fit`

```cpp
void fit(const Matrix& X);
```

### Mathematical role

`fit` is the core PCA training method.

It computes the principal component basis from the input data.

The mathematical process is:

```txt
1. compute feature-wise mean
2. center the matrix
3. compute covariance structure
4. compute eigenvalues and eigenvectors
5. sort components by explained variance
6. retain the top num_components directions
7. store explained variance and explained variance ratio
```

Feature-wise mean:

```txt
mean_j = (1 / n) * sum_i X_ij
```

Centered matrix:

```txt
X_centered = X - mean
```

Covariance matrix:

```txt
C = (1 / (n - 1)) * X_centered^T X_centered
```

Eigen decomposition:

```txt
C v_j = λ_j v_j
```

Top components:

```txt
components = [v_1, v_2, ..., v_k]
```

where:

```txt
k = num_components
```

### What `fit` does mathematically

`fit` implements:

```txt
mean estimation
data centering
covariance/eigenvalue structure extraction
component selection
explained variance computation
explained variance ratio computation
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
option validation
input validation
empty dataset rejection
component-count validation
feature-count tracking
matrix allocation
fitted-state tracking
```

These are necessary for correctness but are not the PCA objective itself.

## `transform`

```cpp
Matrix transform(const Matrix& X) const;
```

### Mathematical role

`transform` projects data into the learned principal-component space.

Math implemented:

```txt
Z = (X - mean) * components
```

Expected output shape:

```txt
Z.rows() == X.rows()
Z.cols() == num_components
```

Each row is the low-dimensional coordinate representation of one sample.

### Interpretation

The columns of `Z` are principal-component coordinates:

```txt
component_1
component_2
...
component_k
```

For `num_components = 2`, these are commonly plotted as a 2D projection.

### Infrastructure role

`transform` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output matrix allocation
```

## `fit_transform`

```cpp
Matrix fit_transform(const Matrix& X);
```

### Mathematical role

`fit_transform` combines fitting and projection.

Mathematically, it performs:

```txt
fit(X)
return transform(X)
```

Equivalent pipeline:

```txt
X
↓
compute PCA basis
↓
project X into that basis
↓
Z
```

### Infrastructure role

This is a convenience method.

It avoids writing:

```cpp
model.fit(X);
Matrix Z = model.transform(X);
```

## `inverse_transform`

```cpp
Matrix inverse_transform(const Matrix& Z) const;
```

### Mathematical role

`inverse_transform` maps PCA coordinates back into the original feature space.

Math implemented:

```txt
X_reconstructed = Z * components^T + mean
```

Expected shape:

```txt
X_reconstructed.rows() == Z.rows()
X_reconstructed.cols() == original_number_of_features
```

If `num_components < original_number_of_features`, reconstruction is approximate.

### Interpretation

Reconstruction shows how much original information is preserved by the retained components.

More components usually means better reconstruction.

### Infrastructure role

`inverse_transform` also handles:

```txt
fitted-state validation
input validation
component-count validation
output matrix allocation
```

## `is_fitted`

```cpp
bool is_fitted() const;
```

### Mathematical role

None directly.

This is state-management infrastructure.

It answers whether the model has learned:

```txt
mean
components
explained variance
explained variance ratio
```

and can transform new data.

## `options`

```cpp
const PCAOptions& options() const;
```

### Mathematical role

Returns the PCA configuration.

This configuration determines:

```txt
number of retained principal components
dimensionality of transformed output
```

## `mean`

```cpp
const Vector& mean() const;
```

### Mathematical role

Returns the feature-wise mean vector used for centering.

Expected shape:

```txt
mean.size() == number_of_features
```

The mean is used in:

```txt
transform:
  X_centered = X - mean

inverse_transform:
  X_reconstructed = Z * components^T + mean
```

### Infrastructure role

Also checks fitted state before exposing learned values.

## `components`

```cpp
const Matrix& components() const;
```

### Mathematical role

Returns the learned principal directions.

Expected shape:

```txt
components.rows() == number_of_features
components.cols() == num_components
```

Each column is a principal component direction:

```txt
components.col(0) = first principal component
components.col(1) = second principal component
...
```

The components are orthogonal directions ordered by explained variance.

### Infrastructure role

Also checks fitted state before exposing learned values.

## `explained_variance`

```cpp
const Vector& explained_variance() const;
```

### Mathematical role

Returns the variance captured by each retained component.

Expected shape:

```txt
explained_variance.size() == num_components
```

Each value corresponds to an eigenvalue:

```txt
λ_j
```

for the retained component.

Interpretation:

```txt
larger λ_j:
  component captures more variance
```

## `explained_variance_ratio`

```cpp
const Vector& explained_variance_ratio() const;
```

### Mathematical role

Returns the fraction of total variance captured by each retained component.

Expected shape:

```txt
explained_variance_ratio.size() == num_components
```

Formula:

```txt
explained_variance_ratio_j = λ_j / sum_all_eigenvalues
```

Cumulative explained variance ratio:

```txt
sum_j explained_variance_ratio_j
```

This tells how much of the original variance is preserved by the selected components.

## `num_features`

```cpp
Eigen::Index num_features() const;
```

### Mathematical role

Returns the number of original input features.

This controls:

```txt
mean vector size
component matrix row count
valid feature count for transform
output feature count for inverse_transform
```

### Infrastructure role

Also supports validation of future input matrices.

## Important internal math concepts

## Centering

PCA must center the data before computing principal directions.

Centered data:

```txt
X_centered = X - mean
```

Without centering, the first component may capture the offset from the origin rather than the direction of maximum variance around the data mean.

## Covariance structure

PCA is based on covariance.

The covariance matrix measures how features vary together:

```txt
C = (1 / (n - 1)) X_centered^T X_centered
```

Principal components are directions in feature space that diagonalize this covariance structure.

## Eigenvectors and eigenvalues

The covariance matrix eigenvectors are principal directions:

```txt
C v_j = λ_j v_j
```

where:

```txt
v_j:
  principal component direction

λ_j:
  variance captured by that direction
```

The eigenvectors are orthogonal for a symmetric covariance matrix.

## Explained variance

Explained variance measures how much variance a component captures.

If:

```txt
λ_1 > λ_2
```

then the first component captures more variance than the second.

## Explained variance ratio

Explained variance ratio normalizes explained variance by total variance:

```txt
λ_j / sum_k λ_k
```

This makes it easier to interpret how much information is retained.

Example:

```txt
component 1 ratio = 0.35
component 2 ratio = 0.28
```

means the first two components together retain approximately:

```txt
0.63 = 63%
```

of total variance.

## Projection

Projection maps original features to component coordinates:

```txt
Z = X_centered * components
```

Each PCA coordinate tells how much the sample aligns with a principal direction.

## Reconstruction

Reconstruction maps PCA coordinates back to original feature space:

```txt
X_reconstructed = Z * components^T + mean
```

If fewer components are kept, reconstruction loses information in discarded directions.

## Dimensionality reduction

If:

```txt
num_components < number_of_features
```

then PCA reduces dimensionality.

This can help:

```txt
visualization
compression
noise reduction
clustering workflows
```

but may discard information.

## Feature scaling

PCA is variance-based, so feature scale matters.

If one feature has much larger numeric variance, PCA may prioritize it.

For most practical workflows, standardize features before PCA:

```txt
mean 0
standard deviation 1
```

This makes each feature contribute more comparably.

## PCA vs supervised learning

PCA does not use targets.

It preserves high-variance directions, not necessarily predictive directions.

A low-variance direction can still be important for classification or regression, but PCA may discard it if only a few components are retained.

## PCA + KMeans

PCA is often combined with KMeans for visualization.

Example workflow:

```txt
standardized features
        ↓
PCA transform to 2D
        ↓
KMeans cluster labels
        ↓
2D cluster plot
```

PCA provides coordinates for visualization.

KMeans provides cluster assignments.

The combination is exploratory, not proof of ground-truth structure.

## Method classification

| Method / Struct | PCA math | Projection math | Reconstruction math | Diagnostics math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `PCAOptions` | Partial | Partial | No | No | Yes |
| `validate_pca_options` | Partial | No | No | No | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Yes | No | No | Yes | Yes |
| `transform` | Yes | Yes | No | No | Yes |
| `fit_transform` | Yes | Yes | No | Yes | Yes |
| `inverse_transform` | Yes | No | Yes | No | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `options` | Partial | Partial | Partial | No | Yes |
| `mean` | Yes | Yes | Yes | No | Yes |
| `components` | Yes | Yes | Yes | No | Yes |
| `explained_variance` | Yes | No | No | Yes | Yes |
| `explained_variance_ratio` | Yes | No | No | Yes | Yes |
| `num_features` | Partial | Partial | Partial | No | Yes |

## Output files and math meaning

## `metrics.csv`

Relevant rows:

```txt
model = PCA
metric = explained_variance_ratio_1
metric = explained_variance_ratio_2
...
```

Math meaning:

```txt
explained_variance_ratio_j:
  fraction of total variance captured by component j
```

For PCA + KMeans, the metrics file can also include clustering metrics such as:

```txt
model = PCA+KMeans
metric = inertia
metric = iterations
```

Those belong to the KMeans step, not the PCA step.

## `projections.csv`

Relevant columns:

```txt
component_1
component_2
label_reference
```

Math meaning:

```txt
component_1:
  coordinate along first principal component

component_2:
  coordinate along second principal component

label_reference:
  optional external label, if available
```

The projection columns come from:

```txt
Z = (X - mean) * components
```

## `clustering_assignments.csv`

PCA itself does not assign clusters.

However, the unsupervised workflow may export cluster assignments from:

```txt
KMeans
PCA+KMeans
```

These assignments can be visualized together with PCA coordinates.

## `hyperparameter_sweep.csv`

Relevant rows:

```txt
model = PCA
param_name = num_components
metric = explained_variance_ratio_1
metric = explained_variance_ratio_2
metric = cumulative_explained_variance_ratio
```

Math meaning:

```txt
num_components:
  number of retained principal components

cumulative_explained_variance_ratio:
  total fraction of variance retained by selected components
```

This sweep helps show the trade-off between compression and information retention.

## Why PCA does not export supervised metrics

PCA is unsupervised.

It does not know:

```txt
y_true
class labels
regression targets
```

during fitting.

Therefore, metrics such as:

```txt
accuracy
precision
recall
f1
mse
r2
```

are not part of core PCA evaluation unless PCA is used inside a downstream supervised workflow.

## Practical interpretation

`PCA` is useful when:

```txt
data has correlated numeric features
variance structure is meaningful
a lower-dimensional representation is needed
2D visualization is helpful
preprocessing before clustering is useful
```

If PCA works well for visualization, the first two or three components may reveal clear structure.

If PCA is hard to interpret, possible causes include:

```txt
variance is spread across many dimensions
features are not scaled
important structure is nonlinear
outliers dominate variance
high-variance directions are not semantically meaningful
```

In the Phase 11 stock OHLCV workflow, PCA is exploratory. The first components summarize variance in engineered financial features, but they should not be treated as guaranteed market factors without deeper validation.

## Summary

`PCA` maps to the following mathematical pipeline:

```txt
data matrix X
chosen num_components
        ↓
compute feature-wise mean
        ↓
center data
        ↓
compute covariance/eigen structure
        ↓
select top principal directions
        ↓
project data:
  Z = (X - mean) * components
        ↓
export explained variance and projection coordinates
        ↓
optionally combine with KMeans for visualization
```

The core math lives in:

```txt
fit
transform
fit_transform
inverse_transform
mean
components
explained_variance
explained_variance_ratio
num_features
```

The supporting infrastructure lives in:

```txt
options
validate_pca_options
is_fitted
input validation
component-count validation
output export workflows
```
