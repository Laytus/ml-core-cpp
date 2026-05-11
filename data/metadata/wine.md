# Dataset: UCI Wine

## Source

- Source name: UCI Machine Learning Repository – Wine recognition data
- Source URL: https://archive.ics.uci.edu/ml/machine-learning-databases/wine/
- Citation / reference: Forina, M. et al.; Stefan Aeberhard. See `wine.names` for original source notes and past usage references.
- License / usage note: For educational project use. Keep the original `wine.names` file with the raw data for attribution/context.

## Task Type

- multiclass classification
- unsupervised / visualization fallback

## Raw Data Location

```txt
data/raw/wine/
```

Expected raw files:

```txt
data/raw/wine/wine.data
data/raw/wine/wine.names
```

## Processed Data Location

```txt
data/processed/wine.csv
```

## Raw Columns

The raw `wine.data` file has no header.

The first column is the class identifier.

The remaining 13 columns are continuous chemical measurements:

```txt
class
alcohol
malic_acid
ash
alcalinity_of_ash
magnesium
total_phenols
flavanoids
nonflavanoid_phenols
proanthocyanins
color_intensity
hue
od280_od315_of_diluted_wines
proline
```

## Target Column

```txt
class
```

## Feature Columns

```txt
alcohol
malic_acid
ash
alcalinity_of_ash
magnesium
total_phenols
flavanoids
nonflavanoid_phenols
proanthocyanins
color_intensity
hue
od280_od315_of_diluted_wines
proline
```

## Label Encoding

The raw file uses class labels:

```txt
1
2
3
```

The processed CSV should encode classes as zero-based labels:

```txt
0 = original class 1
1 = original class 2
2 = original class 3
```

This zero-based encoding is preferred for C++ multiclass model workflows.

## Preprocessing Notes

Processing should:

- read `data/raw/wine/wine.data`
- assign explicit column headers
- convert the class labels from `1,2,3` to `0,1,2`
- preserve all 13 numeric feature columns
- write one processed CSV with a header row
- reject rows with missing or invalid values

Scaling should not be baked into the processed CSV.

Standardization should be applied inside experiment workflows when needed, especially for:

- `KNNClassifier`
- `SoftmaxRegression`
- `PCA`
- `KMeans`

Tree-based models are less sensitive to feature scaling, but using a consistent preprocessing workflow may still simplify comparisons.

## Intended Phase 11 Workflows

- multiclass classification comparison
- PCA projection fallback
- KMeans clustering fallback
- PCA + KMeans combined workflow fallback
- Python/Jupyter visualization of multiclass and unsupervised outputs

## Intended Models

Multiclass classification:

- `SoftmaxRegression`
- `KNNClassifier`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GaussianNaiveBayes`

Unsupervised fallback:

- `PCA`
- `KMeans`
- PCA + KMeans combined workflow

## Dataset Size and Class Balance

The raw `wine.names` file reports:

```txt
Number of instances:
  class 1: 59
  class 2: 71
  class 3: 48

Number of attributes:
  13

Missing attribute values:
  none
```

## Notes

Wine is kept as a stable classical tabular dataset so Phase 11 does not depend entirely on custom stock feature engineering.

It is small enough for fast C++ experiments and useful for validating multiclass classification, PCA, KMeans, and visualization outputs.

The dataset documentation recommends standardizing variables for classifiers that are not scale invariant.
