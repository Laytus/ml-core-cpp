# Unsupervised Sweep – PCA and KMeans

## Setup

This sweep studies unsupervised behavior on the real feature dataset:

```txt
data/processed/stock_ohlcv_engineered.csv
```

The workflow uses the engineered OHLCV feature columns without a supervised target.

The sweep results were exported to:

```txt
outputs/practical-exercises/unsupervised/hyperparameter_sweep.csv
```

The unsupervised sweep covered:

```txt
PCA:
  num_components

KMeans:
  num_clusters
```

The main metrics used for interpretation were:

```txt
PCA:
  explained_variance_ratio_1
  explained_variance_ratio_2
  explained_variance_ratio_3
  cumulative_explained_variance_ratio

KMeans:
  inertia
  iterations
```

## PCA Sweep

### What changed

The PCA sweep varied the number of retained components:

```txt
num_components = 2
num_components = 3
```

The observed cumulative explained variance ratios were:

```txt
pca_components_2:
  cumulative_explained_variance_ratio = 0.629482245536

pca_components_3:
  cumulative_explained_variance_ratio = 0.829574241370
```

### What improved

Increasing the number of PCA components from 2 to 3 substantially increased the cumulative explained variance ratio.

The first two components explained approximately:

```txt
62.95% of the variance
```

The first three components explained approximately:

```txt
82.96% of the variance
```

This means the third component preserved a meaningful amount of additional information.

### What worsened

Using fewer components makes the representation easier to visualize, but it discards more information.

A 2D PCA projection is useful for plotting, but it does not preserve as much variance as the 3D representation.

The trade-off is:

```txt
2 components:
  easier to visualize
  less information retained

3 components:
  more information retained
  less convenient for simple 2D plots
```

### Conceptual takeaway

PCA is a dimensionality-reduction method that projects data onto orthogonal directions of maximum variance.

The number of components controls the balance between compression and information retention.

The main lesson is:

```txt
More PCA components retain more variance, but visualization becomes less simple.
A 2D PCA projection is useful for exploration, not necessarily for preserving all structure.
Cumulative explained variance helps quantify the compression trade-off.
```

## KMeans Sweep

### What changed

The KMeans sweep varied the number of clusters:

```txt
num_clusters = 2
num_clusters = 4
```

The observed inertia values were:

```txt
kmeans_k2:
  inertia = 3161.57834899

kmeans_k4:
  inertia = 2373.77557625
```

### What improved

Increasing the number of clusters from 2 to 4 reduced inertia.

This is expected because KMeans inertia measures the sum of squared distances from samples to their assigned cluster centroids. With more centroids, the model can usually fit the data more closely.

In this sweep:

```txt
k = 4
```

produced lower inertia than:

```txt
k = 2
```

### What worsened

Lower inertia does not automatically mean better or more meaningful clusters.

Increasing the number of clusters can make the model fit local structure more closely, but it can also create clusters that are harder to interpret.

The trade-off is:

```txt
fewer clusters:
  simpler interpretation
  higher inertia

more clusters:
  lower inertia
  potentially more detailed segmentation
  harder interpretation
```

### Conceptual takeaway

KMeans partitions data into groups by minimizing within-cluster squared distance.

The number of clusters controls the granularity of the partition.

The main lesson is:

```txt
KMeans inertia usually decreases as the number of clusters increases.
Inertia alone is not enough to choose the best k.
Cluster interpretability and downstream visualization should also be considered.
```

## PCA + KMeans Visualization Context

The broader unsupervised workflow also exports PCA projections and KMeans assignments:

```txt
outputs/practical-exercises/unsupervised/projections.csv
outputs/practical-exercises/unsupervised/clustering_assignments.csv
```

This allows notebook visualizations such as:

```txt
PCA component 1 vs PCA component 2
colored by KMeans cluster
```

The PCA sweep helps decide how much information the projection retains, while the KMeans sweep helps study different cluster granularities.

Together, they illustrate the common exploratory workflow:

```txt
standardize features
project high-dimensional data to 2D with PCA
cluster the data with KMeans
visualize clusters in PCA space
```

## Overall interpretation

The unsupervised sweep confirmed expected behavior:

```txt
PCA:
  adding components increased cumulative explained variance

KMeans:
  adding clusters reduced inertia
```

For the stock OHLCV engineered features, 2 PCA components retained a useful amount of variance for visualization, while 3 components preserved substantially more information.

For KMeans, using 4 clusters reduced inertia compared with 2 clusters, but the result should be interpreted as a finer segmentation rather than automatically a better one.

## Conceptual takeaway

Unsupervised learning requires more careful interpretation than supervised evaluation because there is no direct ground-truth target.

The main practical lesson is:

```txt
For PCA, use explained variance to understand information retention.
For KMeans, use inertia to understand compactness, but not as the only criterion.
For visualization, PCA + KMeans is useful for exploring structure, not proving semantic clusters.
```

## Files

Input sweep output:

```txt
outputs/practical-exercises/unsupervised/hyperparameter_sweep.csv
```

Related visualization-ready outputs:

```txt
outputs/practical-exercises/unsupervised/projections.csv
outputs/practical-exercises/unsupervised/clustering_assignments.csv
```

Generated by:

```txt
src/workflows/hyperparameter_sweeps.cpp
src/workflows/unsupervised_comparison.cpp
```

Summarized with:

```txt
scripts/summarize_hyperparameter_sweeps.py
```
