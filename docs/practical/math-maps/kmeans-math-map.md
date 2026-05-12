# KMeans Math Map

## Purpose

This document maps the public `KMeans` API and its important behavior to the mathematical process implemented by the model.

The goal is to make clear which parts of the class correspond to model mathematics and which parts are infrastructure.

Related files:

```txt
include/ml/unsupervised/kmeans.hpp
src/unsupervised/kmeans.cpp
include/ml/common/types.hpp
```

Related practical usage doc:

```txt
docs/practical/models/kmeans-usage.md
```

Related theory doc:

```txt
docs/theory/unsupervised-learning.md
```

## Model idea

`KMeans` is an unsupervised clustering algorithm.

It partitions samples into a fixed number of clusters by minimizing the squared distance between each sample and its assigned centroid.

Given:

```txt
X = data matrix
k = number of clusters
```

KMeans learns:

```txt
centroids = μ_1, μ_2, ..., μ_k
labels = cluster assignment for each sample
```

Each sample is assigned to the closest centroid:

```txt
label_i = argmin_c ||x_i - μ_c||^2
```

Each centroid is recomputed as the mean of the samples assigned to that cluster:

```txt
μ_c = mean({x_i : label_i = c})
```

The process alternates between assignment and update steps until convergence or until `max_iterations` is reached.

## Objective function

KMeans minimizes inertia, also called within-cluster sum of squares:

```txt
inertia = sum_i ||x_i - μ_label_i||^2
```

where:

```txt
x_i = sample i
μ_label_i = centroid assigned to sample i
```

Lower inertia means samples are closer to their assigned centroids.

However, lower inertia alone does not prove that clusters are semantically meaningful, because inertia usually decreases as the number of clusters increases.

## Public API to math mapping

## `KMeansOptions`

```cpp
struct KMeansOptions {
    std::size_t num_clusters;
    std::size_t max_iterations;
    double tolerance;
};
```

### Mathematical role

`KMeansOptions` controls the clustering optimization process.

Important fields:

```txt
num_clusters
max_iterations
tolerance
```

Math meaning:

```txt
num_clusters:
  number of centroids / clusters

max_iterations:
  maximum number of assignment-update loops

tolerance:
  convergence threshold for stopping
```

The value of `num_clusters` defines the number of cluster centers:

```txt
μ_1, μ_2, ..., μ_k
```

## `validate_kmeans_options`

```cpp
void validate_kmeans_options(
    const KMeansOptions& options,
    const std::string& context
);
```

### Mathematical role

Ensures the clustering configuration is meaningful.

Typical mathematical requirements:

```txt
num_clusters >= 1
max_iterations >= 1
tolerance >= 0
```

A model with zero clusters cannot assign samples.

A model with zero iterations cannot perform the iterative refinement process.

### Infrastructure role

It also provides:

```txt
consistent error messages
context-specific validation
early failure for invalid configuration
```

## Constructor

```cpp
KMeans();
explicit KMeans(KMeansOptions options);
```

### Mathematical role

The constructor does not perform clustering.

It only stores the clustering configuration:

```txt
number of clusters
maximum iterations
tolerance
```

Math implemented:

```txt
none directly
```

Infrastructure role:

```txt
configure clustering behavior
validate options
```

## `fit`

```cpp
void fit(const Matrix& X);
```

### Mathematical role

`fit` is the core KMeans training method.

It estimates:

```txt
cluster centroids
cluster assignments
```

through iterative optimization.

The algorithm follows this loop:

```txt
1. initialize centroids
2. assign each sample to the nearest centroid
3. recompute centroids from assigned samples
4. compute inertia
5. check convergence
6. repeat until convergence or max_iterations
```

Assignment step:

```txt
label_i = argmin_c ||x_i - μ_c||^2
```

Update step:

```txt
μ_c = (1 / n_c) * sum_{i: label_i = c} x_i
```

where:

```txt
n_c = number of samples assigned to cluster c
```

Objective:

```txt
inertia = sum_i ||x_i - μ_label_i||^2
```

### What `fit` does mathematically

`fit` implements:

```txt
1. choose initial centroids
2. compute nearest-centroid assignments
3. recompute centroids as cluster means
4. compute inertia
5. store inertia history
6. stop when centroid movement or objective change is below tolerance
7. store final labels and centroids
```

### What `fit` does as infrastructure

`fit` also handles:

```txt
option validation
input validation
empty dataset rejection
feature-count tracking
matrix allocation
iteration counting
fitted-state tracking
```

These are necessary for correctness but are not the clustering objective itself.

## `predict`

```cpp
Vector predict(const Matrix& X) const;
```

### Mathematical role

`predict` assigns each sample to the nearest fitted centroid.

For each sample `x_i`:

```txt
label_i = argmin_c ||x_i - μ_c||^2
```

Expected output:

```txt
labels.size() == X.rows()
```

The output values are cluster IDs:

```txt
0, 1, ..., num_clusters - 1
```

### What `predict` does mathematically

`predict` implements:

```txt
distance from sample to each centroid
nearest-centroid selection
cluster-label assignment
```

### What `predict` does as infrastructure

`predict` also handles:

```txt
fitted-state validation
input validation
feature-count validation
output vector allocation
looping over samples
```

## `fit_predict`

```cpp
Vector fit_predict(const Matrix& X);
```

### Mathematical role

`fit_predict` combines training and assignment.

Mathematically, it performs:

```txt
fit(X)
return labels
```

The returned labels are the assignments produced by the final fitted centroids.

### Infrastructure role

This is a convenience method.

It avoids manually calling:

```cpp
model.fit(X);
model.labels();
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
centroids
labels
inertia history
```

and can assign new samples.

## `options`

```cpp
const KMeansOptions& options() const;
```

### Mathematical role

Returns the clustering configuration.

This configuration defines:

```txt
number of centroids
maximum assignment-update iterations
convergence tolerance
```

## `centroids`

```cpp
const Matrix& centroids() const;
```

### Mathematical role

Returns the learned centroid matrix:

```txt
μ_1, μ_2, ..., μ_k
```

Expected shape:

```txt
centroids.rows() == num_clusters
centroids.cols() == number_of_features
```

Each row is a cluster center.

Centroid formula:

```txt
μ_c = mean({x_i : label_i = c})
```

### Interpretation

A centroid represents the average feature vector of samples assigned to that cluster.

For standardized features, centroid coordinates are expressed in standardized units.

### Infrastructure role

Also checks fitted state before exposing learned centroids.

## `labels`

```cpp
const Vector& labels() const;
```

### Mathematical role

Returns the cluster assignment for each training sample.

Expected shape:

```txt
labels.size() == number_of_training_samples
```

Each label is:

```txt
label_i = argmin_c ||x_i - μ_c||^2
```

### Interpretation

Cluster labels are arbitrary IDs.

Cluster `0` is not inherently better, smaller, or more important than cluster `1`.

### Infrastructure role

Also checks fitted state before exposing assignments.

## `inertia_history`

```cpp
const std::vector<double>& inertia_history() const;
```

### Mathematical role

Returns the objective value across KMeans iterations.

Each entry is:

```txt
sum_i ||x_i - μ_label_i||^2
```

for the current centroids and assignments.

This is useful for checking whether the algorithm is improving.

Expected behavior:

```txt
inertia should usually decrease or stay stable over iterations
```

### Infrastructure role

Stores diagnostic information for plotting and output export.

## `inertia`

```cpp
double inertia() const;
```

### Mathematical role

Returns the final KMeans objective value:

```txt
inertia = sum_i ||x_i - μ_label_i||^2
```

Lower inertia means more compact clusters under the current number of clusters and feature scaling.

### Infrastructure role

Also checks fitted state and exposes final diagnostic value.

## `num_iterations`

```cpp
std::size_t num_iterations() const;
```

### Mathematical role

Returns the number of assignment-update iterations completed.

This is related to convergence behavior.

It does not directly define the objective, but it tells how long the optimization process ran.

## Important internal math concepts

## Centroid initialization

KMeans requires initial centroids.

A simple implementation may choose initial samples or deterministic rows as centroids.

Initialization matters because KMeans can converge to different local minima.

The optimization objective is non-convex, so different starting centroids can produce different final clusters.

## Assignment step

Given current centroids, each sample is assigned to the nearest centroid:

```txt
label_i = argmin_c ||x_i - μ_c||^2
```

This step fixes centroids and optimizes labels.

## Update step

Given current labels, each centroid is updated to the mean of assigned samples:

```txt
μ_c = (1 / n_c) * sum_{i: label_i = c} x_i
```

This step fixes labels and optimizes centroids.

## Alternating minimization

KMeans alternates between:

```txt
assignment step
update step
```

Each step reduces or preserves the objective when implemented correctly.

This is why inertia should generally not increase across iterations.

## Inertia

Inertia measures cluster compactness:

```txt
inertia = sum_i ||x_i - μ_label_i||^2
```

Interpretation:

```txt
lower inertia:
  samples are closer to assigned centroids

higher inertia:
  samples are farther from assigned centroids
```

But inertia depends strongly on:

```txt
number of clusters
feature scaling
dataset size
feature dimensionality
```

## Number of clusters

The number of clusters `k` is chosen before fitting.

KMeans does not automatically discover the best `k`.

As `k` increases:

```txt
inertia usually decreases
clusters become more granular
interpretation can become harder
```

## Empty clusters

An empty cluster can occur if no samples are assigned to a centroid.

Mathematically, its mean is undefined:

```txt
μ_c = mean(empty set)
```

Implementations must handle this case.

Common strategies include:

```txt
keep the previous centroid
reinitialize the centroid
choose a far-away sample
```

The exact strategy is implementation-dependent.

## Feature scaling

KMeans uses squared Euclidean distances, so feature scale is part of the math.

If one feature has a much larger numeric range than others, it can dominate:

```txt
||x_i - μ_c||^2
```

This is why standardization is usually required.

## KMeans vs classification

KMeans does not use labels.

Cluster IDs are discovered from geometry, not from target classes.

Even if clusters visually resemble classes, they should not be treated as supervised labels unless validated separately.

## KMeans + PCA

In Phase 11, KMeans is combined with PCA for visualization.

Typical workflow:

```txt
standardized features
        ↓
PCA projection to 2D
        ↓
plot points using KMeans cluster labels
```

or:

```txt
standardized features
        ↓
PCA transform
        ↓
KMeans on PCA coordinates
        ↓
visualize clusters
```

PCA helps visualize cluster structure, but it may discard information.

## Method classification

| Method / Struct | Clustering math | Optimization math | Distance math | Diagnostics math | Infrastructure |
|---|---:|---:|---:|---:|---:|
| `KMeansOptions` | Partial | Yes | No | No | Yes |
| `validate_kmeans_options` | Partial | Partial | No | No | Yes |
| Constructor | No | No | No | No | Yes |
| `fit` | Yes | Yes | Yes | Yes | Yes |
| `predict` | Yes | No | Yes | No | Yes |
| `fit_predict` | Yes | Yes | Yes | Yes | Yes |
| `is_fitted` | No | No | No | No | Yes |
| `options` | Partial | Partial | No | No | Yes |
| `centroids` | Yes | No | No | No | Yes |
| `labels` | Yes | No | Yes | No | Yes |
| `inertia_history` | No | Diagnostic | Yes | Yes | Yes |
| `inertia` | No | Diagnostic | Yes | Yes | Yes |
| `num_iterations` | No | Diagnostic | No | Partial | Yes |

## Output files and math meaning

## `metrics.csv`

Relevant rows:

```txt
model = KMeans
metric = inertia
metric = iterations
```

Math meaning:

```txt
inertia:
  final within-cluster sum of squared distances

iterations:
  number of assignment-update iterations completed
```

For PCA + KMeans, rows may use:

```txt
model = PCA+KMeans
```

to indicate clustering in a PCA-derived representation.

## `clustering_assignments.csv`

Relevant columns:

```txt
row_id
method
cluster
label_reference
```

Math meaning:

```txt
row_id:
  sample index in the exported workflow

method:
  KMeans or PCA+KMeans

cluster:
  nearest-centroid assignment

label_reference:
  optional external label if available
```

The `cluster` column corresponds to:

```txt
argmin_c ||x_i - μ_c||^2
```

## `projections.csv`

Relevant columns:

```txt
component_1
component_2
label_reference
```

This file is usually produced by PCA, but KMeans cluster assignments can be visualized together with PCA projections.

Math meaning:

```txt
component_1, component_2:
  low-dimensional coordinates used for plotting

cluster labels:
  assigned by KMeans or PCA+KMeans workflow
```

## `hyperparameter_sweep.csv`

Relevant rows:

```txt
model = KMeans
param_name = num_clusters
metric = inertia
metric = iterations
```

Math meaning:

```txt
num_clusters:
  number of centroids used

inertia:
  compactness achieved by that number of clusters

iterations:
  optimization iterations required
```

The sweep helps show how increasing `k` changes cluster compactness.

## Why KMeans does not export supervised metrics

KMeans is unsupervised.

It does not know true labels during fitting.

Therefore, metrics such as:

```txt
accuracy
precision
recall
f1
```

are not part of the core KMeans workflow unless external labels are intentionally provided for evaluation.

## Practical interpretation

`KMeans` is useful when:

```txt
numeric samples may form centroid-like groups
features are standardized
clusters are used for exploration
```

If KMeans finds meaningful clusters, those clusters should show:

```txt
reasonable compactness
interpretable centroids
stable patterns under parameter changes
useful visualization structure
```

If KMeans behaves poorly, possible causes include:

```txt
wrong number of clusters
unscaled features
non-spherical cluster shapes
strong outliers
high-dimensional noisy features
random initialization issues
no real cluster structure
```

In the Phase 11 stock OHLCV workflow, KMeans is exploratory. Cluster IDs should not be interpreted as proven market regimes without additional validation.

## Summary

`KMeans` maps to the following mathematical pipeline:

```txt
data matrix X
chosen number of clusters k
        ↓
initialize k centroids
        ↓
repeat:
    assign samples to nearest centroid
    update centroids as cluster means
    compute inertia
until convergence or max_iterations
        ↓
final centroids
final cluster labels
        ↓
export metrics, cluster assignments, and visualization-ready outputs
```

The core math lives in:

```txt
fit
predict
fit_predict
centroids
labels
inertia
inertia_history
assignment step
centroid update step
```

The supporting infrastructure lives in:

```txt
options
validate_kmeans_options
is_fitted
num_iterations
input validation
feature-count tracking
output export workflows
```
