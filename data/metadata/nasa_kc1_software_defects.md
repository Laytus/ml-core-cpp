# Dataset: NASA KC1 Software Defect Prediction

## Source

- Source name: PROMISE Software Engineering Repository / NASA Metrics Data Program KC1 dataset
- Source URL: http://promise.site.uottawa.ca/SERepository/
- Citation / reference: NASA Metrics Data Program dataset donated by Tim Menzies. The ARFF header includes the original dataset notes, past usage references, and attribute descriptions.
- License / usage note: For educational project use. Follow the acknowledgment guidelines from the PROMISE repository if publishing material based on this dataset.

## Task Type

- binary classification

## Raw Data Location

```txt
data/raw/nasa_kc1_software_defects/kc1.arff
```

## Processed Data Location

```txt
data/processed/nasa_kc1_software_defects.csv
```

## Raw Columns

The raw ARFF file defines the following attributes:

```txt
loc
v(g)
ev(g)
iv(g)
n
v
l
d
i
e
b
t
lOCode
lOComment
lOBlank
locCodeAndComment
uniq_Op
uniq_Opnd
total_Op
total_Opnd
branchCount
defects
```

## Target Column

```txt
defects
```

## Feature Columns

```txt
loc
v(g)
ev(g)
iv(g)
n
v
l
d
i
e
b
t
lOCode
lOComment
lOBlank
locCodeAndComment
uniq_Op
uniq_Opnd
total_Op
total_Opnd
branchCount
```

## Label Encoding

The raw ARFF file uses:

```txt
false
true
```

The processed CSV should encode the target as:

```txt
0 = false = non-defective
1 = true = defective
```

## Preprocessing Notes

Processing should:

- parse the ARFF file from `data/raw/nasa_kc1_software_defects/kc1.arff`
- preserve all numeric feature columns
- convert the `defects` target from `{false,true}` to `{0,1}`
- write a numeric CSV with one header row
- reject rows with missing or invalid values
- avoid scaling inside the processed CSV

Scaling should be applied inside experiment workflows using the training split only.

Some feature names include characters such as parentheses, for example `v(g)`. If these names become inconvenient in C++ CSV handling or exported result files, the processed dataset may use sanitized names.

Recommended sanitized processed names:

```txt
loc
vg
evg
ivg
n
v
l
d
i
e
b
t
lo_code
lo_comment
lo_blank
loc_code_and_comment
uniq_op
uniq_opnd
total_op
total_opnd
branch_count
defects
```

If sanitized names are used, the conversion must be documented in the processing script or experiment notes.

## Intended Phase 11 Workflows

- binary classification comparison
- classification metrics export
- confusion matrix export
- optional probability / score export where supported
- Python/Jupyter visualization of classification results

## Intended Models

- `LogisticRegression`
- `LinearSVM`
- `GaussianNaiveBayes`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `TinyMLPBinaryClassifier`
- optional educational baseline: `Perceptron`

## Dataset Size and Class Balance

The ARFF header reports:

```txt
Number of instances: 2109
Number of attributes: 22
Missing attributes: none
Class distribution:
  defective: 326 = 15.45%
  non-defective: 1783 = 84.54%
```

This dataset is imbalanced.

Accuracy alone should not be treated as sufficient.

Classification reports should include at least:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

## Notes

This dataset connects Phase 11 to software engineering and code-quality prediction.

The features are static code metrics, including McCabe complexity, Halstead metrics, lines-of-code measurements, and branch count.

The target represents whether a module has reported defects.

The dataset is useful for practical binary classification, but results should be interpreted carefully because static code metrics are imperfect probabilistic indicators, not definitive proof of software quality.
