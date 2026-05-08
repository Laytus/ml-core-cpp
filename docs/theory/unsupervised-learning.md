# Unsupervised Learning

## 1. Purpose of This Document

Unsupervised learning studies structure in data without target labels.

In supervised learning, each sample has:

```txt
features X
target y
```

In unsupervised learning, we usually only have:

```txt
features X
```

The goal is not to predict a known target, but to discover useful structure.

Common unsupervised goals include:

```txt
group similar samples
find latent structure
compress data
reduce dimensionality
understand variance
visualize high-dimensional data
detect unusual patterns
```

Phase 8 focuses on two core unsupervised methods:

```txt
k-means:
    clustering by centroid assignment

PCA:
    dimensionality reduction through variance directions
```

These two methods are foundational before Deep Learning because they build intuition for representation learning.

---

## 2. k-means

### 1. Clustering Fundamentals

Clustering is the task of grouping samples based on similarity.

Given a dataset:

```txt
```
$$
X \in R^{n \times d}
$$

where:

```txt
n = number of samples
d = number of features
```

clustering tries to assign each sample to a group:

```txt
cluster_id ∈ {0, 1, ..., k - 1}
```

Unlike classification, these cluster IDs are not known beforehand.

They are discovered from the feature space.

---

### 2. Clustering vs Classification

Classification uses known labels.

Example:

```txt
sample -> known class label
```

Clustering does not use known labels.

Example:

```txt
sample -> discovered group
```

So the main difference is:

```txt
classification:
    supervised
    learns from labeled examples

clustering:
    unsupervised
    discovers groups from feature geometry
```

A clustering result may or may not correspond to meaningful real-world categories.

The algorithm can only use the structure visible in the features.

---

### 3. Similarity and Distance in Clustering

Most clustering algorithms rely on a notion of similarity.

For k-means, similarity is based on distance to centroids.

A point is assigned to the nearest centroid.

Usually, k-means uses squared Euclidean distance:

$$
\text{distance\_squared}(x, c) = ||x - c||^2
$$

where:

```txt
x = sample
c = centroid
```

This connects Phase 8 directly to Phase 7 distance metrics.

---

### 4. What Is a Centroid?

A centroid is the mean position of a group of points.

For a cluster containing samples:

```txt
x_1, x_2, ..., x_m
```

the centroid is:

$$
c = \frac{1}{m} * \sum_i{x_i}
$$

In vector form, the centroid has the same number of features as each sample.

Example in 2D:

```txt
points:
    [0, 0]
    [2, 0]
    [4, 0]

centroid:
    [2, 0]
```

The centroid represents the center of mass of the cluster.

---

### 5. k-Means Intuition

k-means tries to partition data into `k` clusters.

The algorithm alternates between two steps:

```txt
1. assign each point to the nearest centroid
2. update each centroid as the mean of assigned points
```

This repeats until the assignments stop changing or a maximum number of iterations is reached.

High-level algorithm:

```txt
initialize k centroids

repeat:
    assign each point to nearest centroid
    recompute each centroid as mean of assigned points
```

The algorithm is simple but powerful.

---

### 6. k-Means Objective

k-means minimizes within-cluster squared distance.

The objective is:

```txt
sum over clusters:
    sum over points in cluster:
        ||x_i - c_cluster||^2
```

More compactly:

$$
J = \sum_i ||x_i - c_{z_i}||^2
$$

where:

```txt
x_i = sample i
z_i = cluster assignment for sample i
c_{z_i} = centroid assigned to sample i
```

The goal is to make points close to their assigned centroid.

This objective is also called:

```txt
within-cluster sum of squares
```

or:

```txt
inertia
```

---

### 7. Assignment Step

During the assignment step, each sample is assigned to the nearest centroid.

For each sample:

$$
z_i = \argmin_j ||x_i - c_j||^2
$$

where:

```txt
j = candidate centroid index
```

Example:

```txt
x = [1, 1]

centroid 0 = [0, 0]
centroid 1 = [5, 5]

distance to centroid 0 is smaller
so x is assigned to cluster 0
```

This step reduces or preserves the k-means objective because each point chooses the best available centroid.

---

### 8. Centroid Update Step

After assignments are known, each centroid is updated as the mean of its assigned points.

For cluster `j`:

```txt
c_j = mean of all x_i assigned to cluster j
```

This is optimal because the mean minimizes squared error.

For fixed assignments, the best centroid is the average of its points.

This is why k-means alternates between:

```txt
best assignments for fixed centroids
best centroids for fixed assignments
```

---

### 9. Why k-Means Converges

Each k-means iteration does not increase the objective.

Assignment step:

```txt
each point chooses the closest centroid
objective cannot increase
```

Update step:

```txt
each centroid becomes the mean of its assigned points
objective cannot increase
```

Because the objective is non-negative, the algorithm eventually stops improving.

However, k-means may converge to a local minimum, not necessarily the global best solution.

---

### 10. Initialization Sensitivity

k-means depends strongly on initial centroids.

Different initial centroids can produce different final clusters.

Bad initialization can lead to poor clustering.

Example problem:

```txt
two centroids start too close together
one natural group may be split badly
another group may be ignored
```

For this project, the first implementation can use deterministic initialization.

Recommended first version:

```txt
use the first k samples as centroids
```

This is simple and reproducible.

Later, we can add:

```txt
random initialization
multiple restarts
k-means++ initialization
```

---

### 11. Empty Clusters

An empty cluster happens when no samples are assigned to a centroid.

This can occur after an assignment step.

Example:

```txt
cluster 0 receives 10 points
cluster 1 receives 0 points
cluster 2 receives 5 points
```

The implementation must decide how to handle this.

Possible strategies:

```txt
keep the centroid unchanged
reinitialize the centroid
throw an error
```

Recommended first strategy:

```txt
keep the centroid unchanged
```

This keeps the implementation simple and deterministic.

---

### 12. Choosing k

The number of clusters `k` must be chosen before running k-means.

Small `k`:

```txt
coarse grouping
may merge distinct structures
```

Large `k`:

```txt
finer grouping
may split natural structures
can overfit noise
```

There is no universally correct `k`.

Common ways to study `k` include:

```txt
inertia comparison
elbow method intuition
qualitative inspection
downstream usefulness
```

For Phase 8, experiments should compare different `k` values.

---

### 13. Feature Scaling and k-Means

k-means is sensitive to feature scale.

If one feature has much larger numeric range than another, it can dominate distances.

Example:

```txt
feature 1: 0 to 1
feature 2: 0 to 100000
```

The second feature will dominate squared Euclidean distance.

Therefore, k-means usually needs scaling:

```txt
standardization
min-max normalization
```

This connects to earlier preprocessing work.

---

## 3. PCA (Principal Component Analysis)

### 1. PCA Fundamentals

PCA means Principal Component Analysis.

PCA is an unsupervised dimensionality-reduction method.

Its goal is to find directions in feature space where the data varies the most.

These directions are called:

```txt
principal components
```

PCA can be used for:

```txt
dimensionality reduction
visualization
noise reduction
compression
understanding variance structure
```

---

### 2. Centering the Data

Before PCA, data should be centered.

For each feature, subtract its mean:

$$
X_{\text{centered}} = X - \text{column\_means}(X)
$$

Centering moves the data so each feature has mean zero.

This matters because PCA studies variance around the mean.

Without centering, the first component can be distorted by the data’s absolute position instead of its spread.

---

### 3. Covariance Matrix

The covariance matrix describes how features vary together.

For centered data:

$$
X_{\text{centered}} \in R^{n \times d}
$$

the sample covariance matrix is:

$$
C = \frac{1}{n - 1} \cdot X_{\text{centered}}^{T} X_{\text{centered}}
$$

Shape:

$$
C \in R^{d \times d}
$$

Diagonal entries are feature variances.

Off-diagonal entries are covariances between features.

Example:

```txt
C[j, j] = variance of feature j
C[j, k] = covariance between feature j and feature k
```

---

### 4. Covariance Intuition

Positive covariance means two features tend to increase together.

Negative covariance means one feature tends to increase while the other decreases.

Near-zero covariance means there is little linear relationship.

Examples:

```txt
positive covariance:
    height and weight often increase together

negative covariance:
    speed and travel time for fixed distance

near-zero covariance:
    unrelated measurements
```

PCA uses covariance structure to find the main axes of variation.

---

### 5. Principal Directions

A principal direction is a unit vector pointing along a direction of high variance.

The first principal component is the direction where projected data has maximum variance.

The second principal component is the next highest-variance direction, constrained to be orthogonal to the first.

So PCA finds an ordered set of directions:

```txt
component 1:
    maximum variance direction

component 2:
    maximum remaining variance direction orthogonal to component 1

component 3:
    maximum remaining variance direction orthogonal to previous components
```

---

### 6. Eigenvalues and Eigenvectors in PCA

PCA can be computed from the covariance matrix.

Given covariance matrix:

```txt
C
```

we compute eigenvectors and eigenvalues:

```txt
C v = lambda v
```

where:

```txt
v = eigenvector
lambda = eigenvalue
```

In PCA:

```txt
eigenvector = principal direction
eigenvalue = variance explained by that direction
```

Large eigenvalue means the corresponding direction explains a lot of variance.

Small eigenvalue means the direction explains little variance.

---

### 7. Explained Variance

Explained variance tells how much variance each principal component captures.

If eigenvalues are:

```txt
lambda_1, lambda_2, ..., lambda_d
```

then total variance is:

$$
\text{total} = \sum_j{\text{lambda}_j}
$$

Explained variance ratio for component `j` is:

```txt
lambda_j / total
```

Example:

```txt
eigenvalues:
    [8, 1, 1]

total:
    10

explained variance ratios:
    [0.8, 0.1, 0.1]
```

The first component explains 80% of the variance.

---

### 8. Dimensionality Reduction with PCA

To reduce dimensionality, keep only the first `m` principal components.

If:

$$
m < d
$$

then PCA maps:

$$
R^d \rightarrow R^m
$$

The transformed data is:

$$
Z = X_{\text{centered}} * \text{components}
$$

where:
$$
\text{components} \in R^{d \times m} \\
Z \in R^{n \times m}
$$


This gives a lower-dimensional representation of the original data.

---

### 9. Reconstruction

PCA can approximately reconstruct the original data from the reduced representation.

If:

$$
Z = X_{\text{centered}} * \text{components}
$$

then reconstruction is:

$$
X_{\text{reconstructed}} = Z * \text{components}^{T} + \text{mean}
$$

If all components are kept, reconstruction can be exact up to numerical precision.

If fewer components are kept, reconstruction loses information.

The reconstruction error measures how much information was lost.

---

## 4. PCA vs k-Means

k-means and PCA are both unsupervised, but they answer different questions.

k-means asks:

```txt
which group does this sample belong to?
```

PCA asks:

```txt
which directions explain the most variance?
```

k-means produces:

```txt
cluster assignments
centroids
```

PCA produces:

```txt
principal components
explained variance
lower-dimensional representations
```

Both rely heavily on feature geometry.

Both are affected by scaling.

---

## Phase 8 Implementation Scope

Phase 8 should implement:

```txt
KMeans
PCA
```

Recommended files:

```txt
include/ml/unsupervised/kmeans.hpp
src/unsupervised/kmeans.cpp

include/ml/unsupervised/pca.hpp
src/unsupervised/pca.cpp
```

Experiment folder:

```txt
experiments/phase-8-unsupervised/
outputs/phase-8-unsupervised/
```

Expected outputs:

```txt
kmeans_cluster_comparison.csv
kmeans_cluster_comparison.txt
pca_explained_variance.csv
pca_projection.csv
pca_reconstruction_summary.txt
```

---

## Phase 8 Completion Criteria

This phase is complete when:

```txt
k-means runs end-to-end with reusable fit/predict-style behavior
cluster assignment and centroid update loops are tested
PCA computes principal components using Eigen-supported operations
PCA exposes explained variance and transformed data
dimensionality-reduction experiments are exported
the relationship between covariance and PCA is documented clearly
```