# KMeans Usage

## What the model does

`KMeans` is an unsupervised clustering model.

It partitions samples into a fixed number of clusters by minimizing the distance between each sample and its assigned cluster centroid.

Conceptually:

```txt
1. initialize k centroids
2. assign each sample to the nearest centroid
3. recompute each centroid as the mean of its assigned samples
4. repeat until convergence or max_iterations is reached
```

The optimization objective is inertia:

```txt
inertia = sum of squared distances from each sample to its assigned centroid
```

Lower inertia means samples are closer to their assigned centroids, but lower inertia alone does not prove that the clusters are meaningful.

In practical terms, `KMeans` is useful when you want:

```txt
unsupervised clustering
simple structure discovery
cluster assignments for numeric tabular data
a baseline clustering method
PCA + KMeans visualization workflows
```

## Supported task type

`KMeans` supports:

```txt
unsupervised clustering
```

It does not support:

```txt
supervised regression
binary classification
multiclass classification
probability prediction
decision scores
```

For dimensionality reduction, use:

```txt
PCA
```

For supervised classification, use one of the classifier models.

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
ml::Matrix X(5, 2);
X << 1.0, 2.0,
     1.2, 1.8,
     5.0, 8.0,
     6.0, 9.0,
     9.0, 1.0;
```

There is no target vector because KMeans is unsupervised.

For real dataset workflows, use the CSV loader to create the feature matrix from a processed CSV file.

## Target format

KMeans does not use a target.

In practical workflows, the processed dataset may still contain a supervised target column, but the unsupervised workflow should load only the selected feature columns.

## Preprocessing usually needed

KMeans is highly sensitive to feature scale because it is distance-based.

Recommended preprocessing:

```txt
standardize numeric features
reject missing values or preprocess them before loading
avoid non-numeric feature columns
remove extreme invalid values
consider dimensionality reduction for visualization
```

Feature scaling is very important. Without scaling, a feature with a larger numeric range can dominate the clustering result.

For practical workflows in this project, feature standardization is handled in the workflow layer before clustering.

## How to instantiate the model

Basic model:

```cpp
#include "ml/unsupervised/kmeans.hpp"

ml::KMeansOptions options;
options.num_clusters = 4;
options.max_iterations = 100;
options.tolerance = 1e-6;

ml::KMeans model(options);
```

Default constructor is also available:

```cpp
ml::KMeans model;
```

## Main options

`KMeansOptions` contains:

```txt
num_clusters
max_iterations
tolerance
```

### `num_clusters`

The number of clusters to find.

Interpretation:

```txt
smaller k:
  simpler segmentation
  higher inertia

larger k:
  finer segmentation
  lower inertia
  harder interpretation
```

### `max_iterations`

Maximum number of assignment/update iterations.

Interpretation:

```txt
larger max_iterations:
  more time to converge
  higher compute cost
```

### `tolerance`

Convergence threshold for centroid movement or objective improvement.

Interpretation:

```txt
larger tolerance:
  earlier stopping
  faster but potentially less refined

smaller tolerance:
  stricter convergence
  more iterations
```

## How to call `fit`

```cpp
model.fit(X);
```

Expected behavior:

```txt
- validates non-empty matrix input
- validates options
- initializes centroids
- repeatedly assigns samples to nearest centroids
- updates centroids
- stores final labels
- stores inertia history
- stores number of iterations
- marks the model as fitted
```

The input matrix must contain finite numeric values.

## How to call `predict`

```cpp
ml::Vector cluster_labels = model.predict(X_new);
```

Expected output:

```txt
cluster_labels.size() == X_new.rows()
```

Each value is the index of the nearest centroid.

Calling `predict` before `fit` should be rejected.

## How to call `fit_predict`

```cpp
ml::Vector cluster_labels = model.fit_predict(X);
```

This is a convenience method equivalent to:

```cpp
model.fit(X);
ml::Vector cluster_labels = model.labels();
```

Use `fit_predict` when you want to train the model and immediately obtain cluster assignments for the same dataset.

## How to inspect centroids

After fitting:

```cpp
const ml::Matrix& centroids = model.centroids();
```

Expected shape:

```txt
centroids.rows() == num_clusters
centroids.cols() == X.cols()
```

Each row is the centroid of one cluster.

Interpretation:

```txt
centroid k:
  average feature vector of samples assigned to cluster k
```

Centroids are easiest to interpret when features are standardized or when the feature units are known.

## How to inspect labels

After fitting:

```cpp
const ml::Vector& labels = model.labels();
```

Expected output:

```txt
labels.size() == X.rows()
labels contains cluster IDs from 0 to num_clusters - 1
```

Cluster IDs are arbitrary labels. Cluster `0` does not necessarily mean “smaller”, “better”, or “first” in a semantic sense.

## How to inspect inertia

After fitting:

```cpp
double final_inertia = model.inertia();
const std::vector<double>& history = model.inertia_history();
```

Interpretation:

```txt
inertia:
  sum of squared distances from samples to their assigned centroids

lower inertia:
  more compact clusters

inertia_history:
  objective value across KMeans iterations
```

Inertia should usually decrease or remain stable during training.

## How to inspect iteration count

After fitting:

```cpp
std::size_t iterations = model.num_iterations();
```

This tells how many iterations were run before stopping.

## How to evaluate clustering

KMeans is unsupervised, so there is no direct target-based accuracy unless external labels are available.

Common internal checks:

```txt
inertia
cluster counts
centroid inspection
PCA visualization
stability across k values
```

In this project, the main practical checks are:

```txt
inertia
iterations
cluster assignments
PCA + KMeans visualization
```

## How to read outputs

In Phase 11 practical workflows, `KMeans` writes results through the unsupervised comparison and hyperparameter sweep workflows.

Main output files:

```txt
outputs/practical-exercises/unsupervised/metrics.csv
outputs/practical-exercises/unsupervised/clustering_assignments.csv
outputs/practical-exercises/unsupervised/projections.csv
outputs/practical-exercises/unsupervised/hyperparameter_sweep.csv
```

### `metrics.csv`

Contains rows such as:

```txt
run_id,workflow,dataset,model,split,metric,value
kmeans_baseline,unsupervised,stock_ohlcv_engineered,KMeans,full,inertia,...
kmeans_baseline,unsupervised,stock_ohlcv_engineered,KMeans,full,iterations,...
```

For PCA + KMeans:

```txt
pca_kmeans_baseline,unsupervised,stock_ohlcv_engineered,PCA+KMeans,full,inertia,...
pca_kmeans_baseline,unsupervised,stock_ohlcv_engineered,PCA+KMeans,full,iterations,...
```

### `clustering_assignments.csv`

Contains:

```txt
run_id,row_id,workflow,dataset,method,split,cluster,label_reference
```

Important columns:

```txt
row_id:
  sample index in the exported workflow

method:
  KMeans or PCA+KMeans

cluster:
  assigned cluster ID

label_reference:
  optional reference label if available; empty for purely unsupervised workflows
```

### `projections.csv`

Contains PCA projection coordinates:

```txt
run_id,row_id,workflow,dataset,method,split,component_1,component_2,label_reference
```

This is useful for visualizing cluster assignments in two dimensions.

### `hyperparameter_sweep.csv`

Contains:

```txt
run_id,workflow,dataset,model,split,param_name,param_value,metric,value
```

For KMeans sweeps, the main parameter is:

```txt
num_clusters
```

Example:

```txt
kmeans_k4,unsupervised,stock_ohlcv_engineered,KMeans,full,num_clusters,4,inertia,...
```

## Practical workflow example

`KMeans` is used in the Phase 11 unsupervised workflow:

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
PCA 2D projection
KMeans cluster visualization
PCA + KMeans combined plot
cluster count comparison
num_clusters vs inertia
```

## KMeans vs PCA + KMeans

The Phase 11 unsupervised workflow compares:

```txt
KMeans:
  clustering in the standardized original feature space

PCA+KMeans:
  clustering after projecting to PCA components
```

Interpretation:

```txt
KMeans on original features:
  uses all selected standardized features
  may preserve more information
  harder to visualize directly

PCA+KMeans:
  clusters compressed PCA representation
  easier to visualize in 2D
  may discard information
```

The combined workflow is useful for exploration, not proof of true semantic clusters.

## Hyperparameter behavior

The main KMeans hyperparameter is:

```txt
num_clusters
```

Common behavior:

```txt
as num_clusters increases:
  inertia usually decreases
  clusters become more granular
  interpretation may become harder
```

In the Phase 11 stock OHLCV sweep:

```txt
kmeans_k2:
  inertia = 3161.57834899

kmeans_k4:
  inertia = 2373.77557625
```

The lower inertia for `k = 4` is expected because more centroids can fit the data more closely.

However, this does not automatically mean that `k = 4` is semantically better than `k = 2`.

## Common mistakes to avoid

### 1. Forgetting feature scaling

This is the most important KMeans mistake.

KMeans uses distances, so unscaled features can dominate the clustering result.

Always standardize numeric features unless there is a deliberate reason not to.

### 2. Treating lower inertia as always better

Inertia usually decreases as `num_clusters` increases.

This means inertia alone cannot determine the best number of clusters.

Use it together with:

```txt
interpretability
visualization
cluster counts
domain knowledge
stability checks
```

### 3. Interpreting cluster IDs as ordered labels

Cluster IDs are arbitrary.

Cluster `0` is not inherently smaller, better, or more important than cluster `1`.

### 4. Assuming clusters are ground-truth classes

KMeans finds geometric groupings, not necessarily real semantic categories.

For stock OHLCV data, clusters may reflect feature patterns but should not be treated as true market regimes without additional validation.

### 5. Using too many clusters on small data

Too many clusters can create very small groups that are hard to interpret.

### 6. Ignoring initialization sensitivity

KMeans can depend on centroid initialization.

If results vary strongly, multiple initializations or improved initialization strategies may be needed later.

### 7. Calling prediction methods before `fit`

The model must be fitted before calling:

```txt
predict
centroids
labels
inertia
inertia_history
num_iterations
```

## What experiment output demonstrates this model

The main Phase 11 output files demonstrating `KMeans` are:

```txt
outputs/practical-exercises/unsupervised/metrics.csv
outputs/practical-exercises/unsupervised/clustering_assignments.csv
outputs/practical-exercises/unsupervised/projections.csv
outputs/practical-exercises/unsupervised/hyperparameter_sweep.csv
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
include/ml/unsupervised/kmeans.hpp
src/unsupervised/kmeans.cpp
```

The sweep interpretation doc is:

```txt
docs/practical/sweeps/unsupervised-pca-kmeans-sweep.md
```

## When to use this model

Use `KMeans` when:

```txt
the task is unsupervised clustering
features are numeric
features can be standardized
you want a simple clustering baseline
you want cluster assignments for exploratory analysis
you want to compare different numbers of clusters
you want to combine clustering with PCA visualization
```

Avoid relying on it alone when:

```txt
features are not numeric
features are not scaled
clusters are not roughly centroid-shaped
you need guaranteed semantic categories
you need automatic selection of the best number of clusters
the dataset has severe outliers
the clustering result is used for high-stakes decisions without validation
```
