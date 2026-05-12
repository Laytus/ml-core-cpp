# PCA Usage

## What the model does

`PCA` is an unsupervised dimensionality-reduction model.

PCA means:

```txt
Principal Component Analysis
```

It finds orthogonal directions in the feature space that capture the largest amount of variance in the data.

Conceptually:

```txt
1. compute the feature-wise mean
2. center the data by subtracting the mean
3. compute principal directions
4. keep the top num_components directions
5. project the data onto those directions
```

The transformed representation is lower-dimensional:

```txt
Z = centered_X * components
```

In practical terms, `PCA` is useful when you want:

```txt
dimensionality reduction
2D or 3D visualization
variance analysis
feature compression
a preprocessing step before clustering
PCA + KMeans exploratory workflows
```

## Supported task type

`PCA` supports:

```txt
unsupervised dimensionality reduction
```

It does not support:

```txt
supervised regression
binary classification
multiclass classification
probability prediction
decision scores
clustering by itself
```

For clustering, use:

```txt
KMeans
```

For PCA followed by clustering, use:

```txt
PCA + KMeans
```

## Expected input format

The model expects:

```cpp
Matrix X;
```

where:

```txt
X.rows() == number of samples
X.cols() == number of numeric features
```

Typical shape:

```txt
X: n_samples x n_features
```

Example:

```cpp
ml::Matrix X(5, 3);
X << 1.0, 2.0, 3.0,
     1.2, 1.8, 2.9,
     5.0, 8.0, 7.5,
     6.0, 9.0, 8.2,
     9.0, 1.0, 2.0;
```

There is no target vector because PCA is unsupervised.

For real dataset workflows, use the CSV loader to create the feature matrix from a processed CSV file.

## Target format

PCA does not use a target.

In practical workflows, the processed dataset may still contain a supervised target column, but the unsupervised workflow should load only the selected feature columns.

## Preprocessing usually needed

PCA is sensitive to feature scale because it is based on variance.

Recommended preprocessing:

```txt
standardize numeric features
reject missing values or preprocess them before loading
avoid non-numeric feature columns
remove extreme invalid values
use only feature columns, not target columns
```

Feature scaling is important. Without standardization, features with larger numeric ranges can dominate the first principal components.

For practical workflows in this project, feature standardization is handled in the workflow layer before applying PCA.

## How to instantiate the model

Basic model:

```cpp
#include "ml/unsupervised/pca.hpp"

ml::PCAOptions options;
options.num_components = 2;

ml::PCA model(options);
```

Default constructor is also available:

```cpp
ml::PCA model;
```

## Main options

`PCAOptions` contains:

```txt
num_components
```

### `num_components`

Number of principal components to keep.

Interpretation:

```txt
fewer components:
  stronger compression
  easier visualization
  more information discarded

more components:
  more variance retained
  less compression
  harder to visualize directly
```

For 2D visualization, use:

```txt
num_components = 2
```

For 3D visualization or better information retention, use:

```txt
num_components = 3
```

## How to call `fit`

```cpp
model.fit(X_train);
```

Expected behavior:

```txt
- validates non-empty matrix input
- validates PCA options
- computes feature-wise mean
- centers the data
- computes principal components
- computes explained variance
- computes explained variance ratio
- marks the model as fitted
```

The input matrix must contain finite numeric values.

## How to call `transform`

```cpp
ml::Matrix Z = model.transform(X);
```

Expected output shape:

```txt
Z.rows() == X.rows()
Z.cols() == options.num_components
```

Each row is the sample represented in principal-component coordinates.

Calling `transform` before `fit` should be rejected.

## How to call `fit_transform`

```cpp
ml::Matrix Z = model.fit_transform(X);
```

This is a convenience method equivalent to:

```cpp
model.fit(X);
ml::Matrix Z = model.transform(X);
```

Use `fit_transform` when training PCA and immediately projecting the same dataset.

## How to call `inverse_transform`

```cpp
ml::Matrix X_reconstructed = model.inverse_transform(Z);
```

Expected output shape:

```txt
X_reconstructed.rows() == Z.rows()
X_reconstructed.cols() == original_number_of_features
```

This maps lower-dimensional PCA coordinates back into the original feature space approximation.

Interpretation:

```txt
inverse_transform does not perfectly recover X unless enough components are kept
reconstruction quality improves as more components are retained
```

## How to inspect the mean

After fitting:

```cpp
const ml::Vector& mean = model.mean();
```

Expected shape:

```txt
mean.size() == original_number_of_features
```

This is the feature-wise mean used to center the data.

## How to inspect components

After fitting:

```cpp
const ml::Matrix& components = model.components();
```

Expected shape:

```txt
components.rows() == original_number_of_features
components.cols() == num_components
```

Each component is a principal direction in the original feature space.

Interpretation:

```txt
component 1:
  direction of maximum variance

component 2:
  next orthogonal direction of maximum remaining variance
```

## How to inspect explained variance

After fitting:

```cpp
const ml::Vector& explained_variance = model.explained_variance();
const ml::Vector& explained_variance_ratio = model.explained_variance_ratio();
```

Expected shape:

```txt
explained_variance.size() == num_components
explained_variance_ratio.size() == num_components
```

Interpretation:

```txt
explained_variance:
  absolute variance captured by each component

explained_variance_ratio:
  fraction of total variance captured by each component
```

The cumulative explained variance ratio is useful for deciding how many components to keep:

```txt
cumulative = sum(explained_variance_ratio)
```

## How to inspect feature count

After fitting:

```cpp
Eigen::Index feature_count = model.num_features();
```

This returns the number of original input features.

## How to evaluate PCA

PCA is unsupervised, so there is no direct target-based accuracy.

Common checks:

```txt
explained variance ratio
cumulative explained variance ratio
2D projection visualization
reconstruction quality
downstream clustering behavior
```

In this project, the main practical checks are:

```txt
explained variance ratios
PCA 2D projection
PCA + KMeans visualization
```

## How to read outputs

In Phase 11 practical workflows, `PCA` writes results through the unsupervised comparison and hyperparameter sweep workflows.

Main output files:

```txt
outputs/practical-exercises/unsupervised/metrics.csv
outputs/practical-exercises/unsupervised/projections.csv
outputs/practical-exercises/unsupervised/clustering_assignments.csv
outputs/practical-exercises/unsupervised/hyperparameter_sweep.csv
```

### `metrics.csv`

Contains rows such as:

```txt
run_id,workflow,dataset,model,split,metric,value
pca_2d_baseline,unsupervised,stock_ohlcv_engineered,PCA,full,explained_variance_ratio_1,...
pca_2d_baseline,unsupervised,stock_ohlcv_engineered,PCA,full,explained_variance_ratio_2,...
```

For PCA + KMeans, the metrics file can also contain:

```txt
pca_kmeans_baseline,unsupervised,stock_ohlcv_engineered,PCA+KMeans,full,inertia,...
pca_kmeans_baseline,unsupervised,stock_ohlcv_engineered,PCA+KMeans,full,iterations,...
```

### `projections.csv`

Contains:

```txt
run_id,row_id,workflow,dataset,method,split,component_1,component_2,label_reference
```

Important columns:

```txt
component_1:
  first principal component coordinate

component_2:
  second principal component coordinate

label_reference:
  optional reference label if available; empty for purely unsupervised workflows
```

This file is used by notebooks for 2D visualization.

### `clustering_assignments.csv`

Contains KMeans and PCA+KMeans cluster assignments:

```txt
run_id,row_id,workflow,dataset,method,split,cluster,label_reference
```

PCA itself does not assign clusters. This file is relevant when PCA is combined with KMeans.

### `hyperparameter_sweep.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

For PCA sweeps, the main parameter is:

```txt
num_components
```

Example:

```txt
pca_components_2,unsupervised,stock_ohlcv_engineered,PCA,full,num_components,2,cumulative_explained_variance_ratio,...
pca_components_3,unsupervised,stock_ohlcv_engineered,PCA,full,num_components,3,cumulative_explained_variance_ratio,...
```

## Practical workflow example

`PCA` is used in the Phase 11 unsupervised workflow:

```txt
include/ml/workflows/unsupervised_comparison.hpp
src/workflows/unsupervised_comparison.cpp
```

It is also used in the hyperparameter sweep workflow:

```txt
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

The corresponding notebooks are:

```txt
notebooks/practical-workflows/04_unsupervised_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The notebooks visualize:

```txt
PCA explained variance
PCA 2D projection
KMeans clusters in PCA space
PCA + KMeans combined plot
num_components vs cumulative explained variance
```

## PCA + KMeans

A common workflow is:

```txt
1. standardize features
2. fit PCA
3. transform data to 2D
4. run KMeans on either:
   - original standardized features
   - PCA-transformed features
5. visualize clusters in PCA space
```

Interpretation:

```txt
PCA:
  creates a low-dimensional representation

KMeans:
  assigns cluster IDs

PCA + KMeans:
  helps visualize cluster structure
```

This is exploratory. It should not be interpreted as proof of true semantic groups without further validation.

## Hyperparameter behavior

The main PCA hyperparameter is:

```txt
num_components
```

Common behavior:

```txt
as num_components increases:
  cumulative explained variance increases
  compression decreases
  reconstruction improves
  visualization becomes less simple
```

In the Phase 11 stock OHLCV sweep:

```txt
pca_components_2:
  cumulative_explained_variance_ratio = 0.629482245536

pca_components_3:
  cumulative_explained_variance_ratio = 0.829574241370
```

This means the first three components preserve substantially more variance than the first two components.

## Common mistakes to avoid

### 1. Forgetting feature scaling

This is the most important PCA mistake.

PCA is variance-based, so features with larger numeric ranges can dominate the components.

Always standardize numeric features unless there is a deliberate reason not to.

### 2. Interpreting PCA components as original features

Principal components are linear combinations of original features.

They are not the same as the original columns.

### 3. Assuming 2D PCA preserves all structure

A 2D projection is useful for visualization, but it discards information.

Always inspect explained variance ratio.

### 4. Treating PCA as supervised feature selection

PCA does not use the target.

It preserves high-variance directions, not necessarily the most predictive directions.

### 5. Assuming clusters in PCA space are ground truth

PCA visualizations can reveal structure, but they can also distort distances by discarding components.

Use PCA plots for exploration, not final proof.

### 6. Calling transformation methods before `fit`

The model must be fitted before calling:

```txt
transform
inverse_transform
mean
components
explained_variance
explained_variance_ratio
num_features
```

### 7. Using too many components for visualization

For human-readable 2D plots, use:

```txt
num_components = 2
```

More components may retain more information, but require different visualization strategies.

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating `PCA` are:

```txt
outputs/practical-exercises/unsupervised/metrics.csv
outputs/practical-exercises/unsupervised/projections.csv
outputs/practical-exercises/unsupervised/hyperparameter_sweep.csv
```

Related clustering output:

```txt
outputs/practical-exercises/unsupervised/clustering_assignments.csv
```

The main visualization notebooks are:

```txt
notebooks/practical-workflows/04_unsupervised_outputs.ipynb
notebooks/practical-workflows/05_hyperparameter_sweeps.ipynb
```

The main workflow implementations are:

```txt
include/ml/workflows/unsupervised_comparison.hpp
src/workflows/unsupervised_comparison.cpp
include/ml/workflows/hyperparameter_sweeps.hpp
src/workflows/hyperparameter_sweeps.cpp
```

The model implementation files are:

```txt
include/ml/unsupervised/pca.hpp
src/unsupervised/pca.cpp
```

The sweep interpretation doc is:

```txt
docs/practical/sweeps/unsupervised-pca-kmeans-sweep.md
```

## When to use this model

Use `PCA` when:

```txt
you need dimensionality reduction
you want a 2D visualization of numeric features
you want to inspect variance structure
you want a compact representation
you want a preprocessing step before clustering
you want to study explained variance
```

Avoid relying on it alone when:

```txt
features are not numeric
features are not scaled
you need supervised feature selection
low-variance directions may be predictive
you need directly interpretable original-feature rules
the 2D projection discards too much variance
the result will be used for high-stakes decisions without validation
```
