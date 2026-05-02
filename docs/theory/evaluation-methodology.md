# Evaluation Methodology

## 1. Supervised Dataset Representation

In supervised learning, a dataset is represented by:

- a feature matrix `X`
- a target vector `y`

ML Core uses the convention:

```txt
X: m x n matrix
y: m-dimensional vector
```

where:

- m is the number of samples
- n is the number of features
- each row of X is one sample
- each column of X is one feature
- y(i) is the target corresponding to row X.row(i)

A supervised dataset is valid only if:

```txt
X.rows() == y.size()
X.rows() > 0
X.cols() > 0
```

This abstraction is necessary because later utilities such as train/test splitting, cross-validation, preprocessing pipelines, and evaluation harnesses must operate on the same representation.

The dataset abstraction should not own modeling logic. It should only represent data and enforce shape consistency.

---

## 2. Train/Test Split

A train/test split separates a supervised dataset into two parts:

```txt
training set:
    used to fit model parameters

test set:
    used to estimate performance on unseen data
```

For a supervised dataset:

```txt
X: m x n feature matrix
y: m-dimensional target vector
```

the split must preserve row alignment:

```txt
X.row(i) corresponds to y(i)
```

This means that if row i of $X$ is assigned to the training set, then $y(i)$ must also be assigned to the training set.

A valid split produces:

```txt
X_train: m_train x n
y_train: m_train

X_test: m_test x n
y_test: m_test
```

where:

```txt
m_train + m_test = m
```

The feature count must remain unchanged:

```txt
X_train.cols() == X.cols()
X_test.cols() == X.cols()
```

### Why shuffling matters

If data is ordered, taking the first rows for training and the last rows for testing can produce a biased split.

Examples:

```txt
data sorted by target value
data sorted by time
data sorted by class
```

For general i.i.d. datasets, shuffling before splitting is usually appropriate.

For time-series data, random shuffling may be wrong because it can leak future information into training. Time-aware splitting will be handled later when needed.

### Random seed

A train/test split should support a random seed.

The seed makes a random split reproducible:

```txt
same dataset + same seed = same split
```

This is important for debugging, experiments, and comparing models fairly.

---

## 3. K-Fold Cross-Validation

A single train/validation split can give an unstable estimate of model performance.

If the validation set happens to be unusually easy or unusually hard, the evaluation result may be misleading. K-fold cross-validation reduces this dependency on one specific split.

K-fold cross-validation divides a supervised dataset into `k` folds.

For each iteration:

```txt
one fold      = validation set
remaining k-1 folds = training set
```

Each fold is used as the validation set exactly once.

For a dataset with:

```txt
X: m x n feature matrix
y: m-dimensional target vector
```

k-fold cross-validation produces `k` train/validation splits.

Example with `k = 3`:

```txt
iteration 1:
    validation = fold 1
    train      = folds 2 + 3

iteration 2:
    validation = fold 2
    train      = folds 1 + 3

iteration 3:
    validation = fold 3
    train      = folds 1 + 2
```

The final model performance is usually summarized by averaging the metric values across folds.

For example:

```txt
mean_validation_mse = average validation MSE across all folds
```

### Row alignment

As with train/test splitting, k-fold splitting must preserve `X/y` alignment.

If row `i` of `X` goes into a validation fold, then `y(i)` must go into the same validation fold.

The same applies to training folds.

```txt
X.row(i) must always stay paired with y(i)
```

### Fold sizes

If the number of samples is not perfectly divisible by `k`, fold sizes may differ slightly.

ML Core uses the standard rule:

```txt
base_fold_size = m / k
remainder      = m % k

first remainder folds get one extra sample
```

Example:

```txt
m = 10
k = 3

fold sizes:
4, 3, 3
```

This keeps the folds as balanced as possible.

### Shuffling and reproducibility

For general i.i.d. datasets, shuffling before creating folds is usually appropriate.

A random seed should be used so that the same dataset and same seed produce the same folds.

```txt
same dataset + same k + same seed = same folds
```

This matters for reproducible experiments and fair model comparison.

For ordered data such as time series, random shuffling may be inappropriate because it can leak future information into earlier training folds. Time-aware cross-validation is a separate methodology and should not be mixed with ordinary random k-fold cross-validation.

### Relationship with the test set

K-fold cross-validation is usually applied only to the training portion of the data.

The test set should remain separate and untouched until final evaluation.

A disciplined workflow is:

```txt
1. split full data into train/test
2. run k-fold cross-validation on the training set
3. choose model or hyperparameters using cross-validation results
4. train final model on the full training set
5. evaluate once on the test set
```

The test set should not be used to choose hyperparameters or compare many model variants. Otherwise, it stops being an unbiased final evaluation set.

### ML Core implementation goal

The ML Core k-fold utility should produce a collection of train/validation splits:

```txt
fold 1:
    train dataset
    validation dataset

fold 2:
    train dataset
    validation dataset

...

fold k:
    train dataset
    validation dataset
```

Each split must satisfy:

```txt
train.X.rows() + validation.X.rows() == original.X.rows()
train.X.cols() == original.X.cols()
validation.X.cols() == original.X.cols()
train.X.rows() == train.y.size()
validation.X.rows() == validation.y.size()
```

The utility should support:

```txt
k
shuffle / no shuffle
random seed
```

This will allow later model phases to evaluate algorithms more reliably than with a single validation split.

---

## 4. Preprocessing Pipeline Rules

Preprocessing transforms raw feature data before it is passed to a model.

Common preprocessing steps include:

```txt
standardization
min-max normalization
missing-value handling
feature encoding
feature selection
dimensionality reduction
```

In ML Core, we already have basic feature scaling utilities:

```cpp
standardize_columns(X);
normalize_min_max_columns(X);
```

These functions are useful for direct transformations, but they are not enough for a correct evaluation pipeline.

The key rule is:

```txt
Preprocessing statistics must be learned from the training data only.
```

---

### Why this matters

Some preprocessing operations learn information from the data.

For example, standardization uses:

```txt
column means
column standard deviations
```

Min-max normalization uses:

```txt
column minimum values
column maximum values
```

If these values are computed using the full dataset before splitting, then information from the validation or test sets leaks into the training process.

This is called data leakage.

---

### Incorrect workflow

This workflow is incorrect:

```txt
1. compute means/stds using the full dataset
2. standardize the full dataset
3. split into train/validation/test
4. train and evaluate model
```

The problem is that the training data has indirectly used information from validation and test samples.

Even if the model never sees validation or test labels, it has still benefited from their feature distribution.

This can make evaluation results overly optimistic.

---

### Correct workflow

The correct workflow is:

```txt
1. split data into train/validation/test
2. fit preprocessing statistics on the training set only
3. transform the training set using training statistics
4. transform validation/test sets using the same training statistics
5. train the model on transformed training data
6. tune decisions using validation data
7. evaluate once on test data
```

For standardization:

```txt
fit on train:
    train_means
    train_standard_deviations

transform train:
    (X_train - train_means) / train_standard_deviations

transform validation:
    (X_validation - train_means) / train_standard_deviations

transform test:
    (X_test - train_means) / train_standard_deviations
```

The validation and test sets are transformed using statistics learned from the training set, not their own statistics.

---

### Fit vs transform

A preprocessing pipeline should distinguish between two operations:

```txt
fit:
    learn preprocessing parameters from data

transform:
    apply already-learned preprocessing parameters to data
```

For example, a standard scaler should do:

```txt
fit(X_train):
    compute column means
    compute column standard deviations

transform(X):
    subtract stored means
    divide by stored standard deviations
```

This distinction prevents leakage and makes preprocessing reusable across train, validation, and test datasets.

---

### Fit-transform

A convenience operation is:

```txt
fit_transform(X_train)
```

This means:

```txt
1. fit preprocessing parameters on X_train
2. immediately transform X_train using those parameters
```

This is valid only for the training set.

For validation and test sets, use only:

```txt
transform(X_validation)
transform(X_test)
```

not:

```txt
fit_transform(X_validation)
fit_transform(X_test)
```

because fitting on validation or test data would leak information.

---

### Constant columns

Some feature columns may be constant in the training set.

For standardization, this means:

```txt
standard_deviation = 0
```

For min-max normalization, this means:

```txt
range = max - min = 0
```

ML Core uses this rule:

```txt
If a training feature has zero standard deviation or zero range, replace the scale value with 1.0.
```

This avoids division by zero.

After centering, a constant training column becomes all zeros. Validation or test values are still transformed relative to the training statistics.

---

### Feature count consistency

A fitted preprocessing object must only transform matrices with the same number of features it was fitted on.

If a scaler is fitted on:

```txt
X_train: m x n
```

then any transformed matrix must satisfy:

```txt
X.cols() == n
```

Otherwise, the stored preprocessing statistics do not match the input features.

---

### ML Core implementation goal

For Phase 2, ML Core should introduce fitted preprocessing objects such as:

```cpp
StandardScaler
MinMaxScaler
```

These objects should support:

```txt
fit on training data
transform any compatible dataset
fit_transform for training data
stored preprocessing parameters
shape validation
zero-scale handling
```

The goal is not to build a full pipeline framework yet.

The goal is to make leakage-safe preprocessing possible before implementing full ML models.

---

## 5. Data Leakage

Data leakage happens when information that should not be available during training influences the model, preprocessing, feature engineering, model selection, or evaluation process.

Leakage is dangerous because it can make validation or test results look better than they really are.

A model affected by leakage may appear to perform well during experimentation but fail when used on genuinely unseen data.

---

### Core idea

A clean evaluation workflow must simulate the real future use case.

At training time, the model should only have access to information that would realistically be available before making predictions.

The validation and test sets should act like unseen data.

This means:

```txt
training data:
    can be used to fit models and preprocessing

validation data:
    can be used to compare model choices and tune hyperparameters

test data:
    should be used only once for final evaluation
```

If information from validation or test data influences training decisions, the evaluation becomes biased.

---

### Leakage through preprocessing

A common leakage mistake is fitting preprocessing on the full dataset before splitting.

Incorrect:

```txt
1. compute means/stds using all data
2. standardize all data
3. split into train/validation/test
4. train and evaluate
```

This leaks validation and test feature distribution into the training pipeline.

Correct:

```txt
1. split data into train/validation/test
2. fit scaler on training data only
3. transform train using training statistics
4. transform validation using training statistics
5. transform test using training statistics
```

In ML Core, the correct workflow is:

```cpp
ml::StandardScaler scaler;

ml::Matrix X_train_scaled = scaler.fit_transform(split.train.X);
ml::Matrix X_validation_scaled = scaler.transform(split.validation.X);
ml::Matrix X_test_scaled = scaler.transform(split.test.X);
```

The same idea applies to min-max normalization:

```cpp
ml::MinMaxScaler scaler;

ml::Matrix X_train_scaled = scaler.fit_transform(split.train.X);
ml::Matrix X_validation_scaled = scaler.transform(split.validation.X);
ml::Matrix X_test_scaled = scaler.transform(split.test.X);
```

The validation and test sets must not call `fit_transform`.

---

### Leakage through test-set tuning

Another common leakage mistake is repeatedly using the test set to choose models.

Incorrect:

```txt
1. train model A
2. evaluate on test set
3. train model B
4. evaluate on test set
5. choose the model with the best test score
```

This uses the test set as a validation set.

After enough comparisons, the selected model may be indirectly overfitted to the test set.

Correct:

```txt
1. train multiple models
2. compare them on validation data
3. choose model/hyperparameters using validation results
4. evaluate the final selected model once on the test set
```

The test set is for final estimation, not experimentation.

---

### Leakage through feature engineering

Feature engineering can also leak information.

Examples:

```txt
using future values to create current features
using the target variable to create input features
selecting features based on the full dataset before splitting
encoding categories using statistics from validation/test data
computing aggregate features using rows that would not be available at prediction time
```

A feature is valid only if it would be available at the moment the prediction is supposed to be made.

For example, in a financial prediction setting, a feature computed using tomorrow's price cannot be used to predict today's decision.

---

### Leakage through duplicated or related samples

Leakage can occur when highly similar or duplicated samples appear in both train and validation/test sets.

Example:

```txt
same user appears in train and test
same document appears in train and test
nearly identical rows appear across splits
multiple measurements from the same object are split independently
```

In these cases, the model may appear to generalize, but it is actually recognizing repeated or highly related samples.

When samples are grouped, splitting may need to happen by group rather than by row.

Examples:

```txt
split by user
split by patient
split by company
split by document
split by time period
```

ML Core's current split utilities operate at row level. Group-aware splitting is intentionally out of scope for now.

---

### Leakage in time-ordered data

Time-ordered data requires special care.

For time series or financial data, random shuffling can leak future information into the training set.

Incorrect:

```txt
randomly shuffle all rows
train on future samples
test on earlier samples
```

Correct:

```txt
train on earlier samples
validate/test on later samples
```

For time-ordered data, the split should respect chronological order.

ML Core's current random split and k-fold utilities are appropriate for general i.i.d. datasets, not for time-aware evaluation.

Time-aware splitting will be introduced later if needed.

---

### Leakage through cross-validation

K-fold cross-validation should usually be applied only to the training portion of the dataset.

Incorrect:

```txt
1. run k-fold cross-validation on the full dataset
2. choose hyperparameters
3. report cross-validation score as final performance
```

This does not provide a separate final test estimate.

Correct:

```txt
1. split full dataset into train/test
2. run k-fold cross-validation only on the training set
3. choose hyperparameters using cross-validation results
4. train final model on the full training set
5. evaluate once on the test set
```

This keeps the test set untouched until the end.

---

### Leakage through metric reporting

Leakage can also happen in how results are reported.

Examples:

```txt
reporting the best validation score after many attempts without acknowledging model selection
choosing the metric after seeing results
discarding bad experimental runs
using test results to decide preprocessing choices
```

A clean evaluation report should clearly state:

```txt
what data was used for training
what data was used for validation/model selection
what data was used for final testing
what preprocessing was fitted on which split
which metric was chosen before comparison
```

---

### Practical anti-leakage checklist

Before trusting an evaluation result, ask:

```txt
Was the data split before fitting preprocessing?

Were validation/test statistics kept out of training?

Was the test set used only once at the end?

Were hyperparameters chosen using validation data, not test data?

Were duplicate or related samples handled correctly?

For time-ordered data, was chronological order respected?

Were features available at prediction time?

Was cross-validation applied only to the training portion?

Was the metric chosen before comparing models?
```

If the answer to any of these is no, the reported performance may be optimistic.

---

### ML Core rule

For ML Core, the default evaluation discipline is:

```txt
1. split first
2. fit preprocessing on training data only
3. transform validation/test using training preprocessing parameters
4. train models only on training data
5. choose configurations using validation or cross-validation
6. evaluate final model once on the test set
7. compare against a baseline
```

This rule will guide later phases when implementing linear models, logistic regression, trees, and model comparison workflows.

---

## 6. Baseline Evaluation Flow

A baseline is a simple reference predictor used to judge whether a trained model is actually useful.

A Machine Learning model should not only produce a metric value. It should improve over a reasonable simple alternative.

The core rule is:

```txt
A trained model is useful only if it beats a baseline under the same evaluation protocol.
```

---

### Why baselines matter

Without a baseline, a metric can be hard to interpret.

For example, suppose a regression model obtains:

```txt
MSE = 25.0
```

This number alone does not tell us whether the model is good.

If a simple mean predictor obtains:

```txt
MSE = 100.0
```

then the trained model is clearly better.

But if the mean predictor obtains:

```txt
MSE = 20.0
```

then the trained model is worse than a trivial baseline.

Baselines prevent us from overvaluing complex models that do not actually improve prediction quality.

---

### Baseline predictor

A baseline predictor is intentionally simple.

For regression, a common baseline is the mean target predictor.

It is fitted using the training targets:

```txt
mean_target = mean(y_train)
```

Then it predicts the same value for every sample:

```txt
prediction_i = mean_target
```

Example:

```txt
y_train = [2, 4, 6]

mean_target = 4

baseline predictions for 3 samples:
[4, 4, 4]
```

This baseline ignores all input features.

That is the point: if a trained model cannot beat a predictor that ignores `X`, the trained model is not learning useful feature-target structure.

---

### Trained predictor

A trained predictor is the model being evaluated.

In later phases, this could be:

```txt
linear regression
logistic regression
decision tree
k-nearest neighbors
ensemble model
```

For the current evaluation layer, we do not need to implement the trained model yet.

We only need to support comparing:

```txt
baseline predictions
trained model predictions
targets
```

This keeps the evaluation flow independent from any specific model class.

---

### Metric comparison

Both baseline and trained model predictions must be evaluated using the same metric on the same target vector.

For regression, ML Core uses the following metrics:

```txt
MSE
RMSE
MAE
R²
```

#### Mean Squared Error — MSE

Mean Squared Error is the average squared prediction error:

$$
MSE = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2
$$

where:

```txt
y_i      = true target for sample i
ŷ_i      = prediction for sample i
m        = number of samples
```

Interpretation:

```txt
lower is better
0 means perfect predictions
large errors are penalized strongly because errors are squared
units are squared target units
```

MSE is useful when large errors should be punished more heavily than small errors.

#### Root Mean Squared Error — RMSE

Root Mean Squared Error is the square root of MSE:

$$
RMSE = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2}
$$

or:

$$
RMSE = \sqrt{MSE}
$$

Interpretation:

```txt
lower is better
0 means perfect predictions
expressed in the same units as the target variable
still penalizes large errors strongly because it comes from squared error
```

RMSE is often easier to interpret than MSE because it has the same unit as the target.

#### Mean Absolute Error — MAE

Mean Absolute Error is the average absolute prediction error:

$$
MAE = \frac{1}{m}\sum_{i=1}^{m}|\hat{y}_i - y_i|
$$

Interpretation:

```txt
lower is better
0 means perfect predictions
expressed in the same units as the target variable
treats errors linearly instead of squaring them
less sensitive to large outliers than MSE/RMSE
```

MAE is useful when we want the average error magnitude in the original target units.

#### Coefficient of Determination — R²

R² compares the model's squared error against the squared error of a mean-target baseline.

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

where:

$$
SS_{res} = \sum_{i=1}^{m}(\hat{y}_i - y_i)^2
$$

and:

$$
SS_{tot} = \sum_{i=1}^{m}(y_i - \bar{y})^2
$$

with:

$$
\bar{y} = \frac{1}{m}\sum_{i=1}^{m}y_i
$$

Interpretation:

```txt
higher is better
R² = 1 means perfect predictions
R² = 0 means same squared-error performance as predicting the target mean
R² < 0 means worse than predicting the target mean
```

R² is useful for understanding how much better the model is than a simple mean baseline, but it is undefined when the target vector has zero variance.

```txt
baseline_metrics = evaluate(baseline_predictions, y)
model_metrics    = evaluate(model_predictions, y)
```

Then compare:

```txt
model MSE  < baseline MSE
model RMSE < baseline RMSE
model MAE  < baseline MAE
model R²   > baseline R²
```

For error metrics such as MSE, RMSE, and MAE, lower is better.

For R², higher is better.

---

### Correct baseline workflow

A leakage-safe baseline comparison should follow this workflow:

```txt
1. split data into train/validation/test
2. fit preprocessing on training data only, if needed
3. fit baseline predictor using y_train only
4. train model using X_train and y_train
5. generate baseline predictions on validation/test data
6. generate model predictions on the same validation/test data
7. compute metrics for both predictors
8. compare model metrics against baseline metrics
```

The baseline and the trained model must be evaluated on the same split.

For example, if model predictions are evaluated on `y_test`, then baseline predictions must also be evaluated on `y_test`.

---

### Validation vs test comparison

During experimentation, baseline comparison can be done on the validation set:

```txt
baseline vs model on validation data
```

This helps decide whether the model is worth keeping or adjusting.

Final reporting should compare both baseline and trained model on the test set:

```txt
baseline vs final model on test data
```

The test set should still be used only once after model selection is complete.

---

### Example regression comparison

Suppose:

```txt
y_test = [2, 4, 6]
```

A baseline predicts the training mean:

```txt
baseline_predictions = [4, 4, 4]
```

A trained model predicts:

```txt
model_predictions = [2, 5, 5]
```

Baseline residuals:

```txt
baseline_predictions - y_test = [2, 0, -2]
```

Model residuals:

```txt
model_predictions - y_test = [0, 1, -1]
```

Baseline MSE:

```txt
(2² + 0² + (-2)²) / 3 = 8 / 3
```

Model MSE:

```txt
(0² + 1² + (-1)²) / 3 = 2 / 3
```

The trained model improves over the baseline because:

```txt
model MSE < baseline MSE
```

---

### ML Core implementation goal

For Phase 2, ML Core should introduce a minimal baseline evaluation flow:

```txt
baseline predictor
regression metrics
evaluation result object
baseline-vs-model comparison
```

The goal is not to build a full experiment tracking system.

The goal is to make future model phases answer this question clearly:

```txt
Does the trained model beat a simple baseline under the same evaluation protocol?
```

The initial implementation should support regression first.

Classification baselines and classification metrics will be introduced later when logistic regression is implemented.

---

## 7. Reusable Evaluation Harness

A reusable evaluation harness is a small structure that standardizes how model predictions are evaluated.

The goal is to avoid rewriting the same evaluation logic for every model implemented later.

Instead of each model phase manually deciding how to compare predictions, ML Core should expose a common evaluation flow:

```txt
targets
baseline predictions
model predictions
        ↓
regression metrics
baseline-vs-model comparison
structured report
```

---

### Why an evaluation harness is useful

Without a shared harness, each model implementation could evaluate results slightly differently.

That creates problems:

```txt
different metrics across models
inconsistent baseline comparison
unclear model names
duplicated evaluation code
harder debugging
harder experiment comparison
```

A reusable harness makes evaluation consistent.

Later models should focus on producing predictions. The harness should handle metric computation and baseline comparison.

---

### What the harness should receive

For regression, the harness needs:

```txt
targets
baseline_predictions
model_predictions
baseline_name
model_name
```

The predictions must all refer to the same evaluation split.

For example, if `targets` is `y_test`, then both `baseline_predictions` and `model_predictions` must also be predictions for the test set.

The required shape rules are:

```txt
baseline_predictions.size() == targets.size()
model_predictions.size() == targets.size()
targets.size() > 0
```

The names should be non-empty so that reports are interpretable.

---

### What the harness should return

The harness should return a structured report containing:

```txt
baseline name
model name
baseline metrics
model metrics
comparison helpers
```

For regression, the metric report should include:

```txt
MSE
RMSE
MAE
R²
```

The comparison helpers answer questions such as:

```txt
Did the model beat the baseline on MSE?
Did the model beat the baseline on RMSE?
Did the model beat the baseline on MAE?
Did the model beat the baseline on R²?
```

For error metrics:

```txt
lower is better
```

So the model beats the baseline when:

```txt
model_metric < baseline_metric
```

For R²:

```txt
higher is better
```

So the model beats the baseline when:

```txt
model_r2 > baseline_r2
```

---

### The harness should not train models

The evaluation harness should not own model training.

It should not know how linear regression, logistic regression, trees, or any future model is fitted.

Its responsibility is only:

```txt
take predictions
compute metrics
compare against baseline
return a structured report
```

This keeps the design modular.

A future model should plug into the harness like this:

```cpp
ml::Vector model_predictions = model.predict(X_test);

ml::RegressionEvaluationInput input{
    y_test,
    baseline_predictions,
    model_predictions,
    "LinearRegression",
    "MeanRegressor"
};

ml::RegressionEvaluationReport report =
    ml::run_regression_evaluation(input);
```

The model is responsible for producing predictions.

The harness is responsible for evaluating those predictions.

---

### Example evaluation flow

A typical regression evaluation flow is:

```txt
1. split dataset into train/validation/test
2. fit preprocessing on training data only
3. transform train/validation/test using training preprocessing parameters
4. fit baseline predictor on y_train
5. train model on X_train and y_train
6. generate baseline predictions on validation or test split
7. generate model predictions on the same split
8. run evaluation harness
9. inspect whether model beats baseline
```

For example:

```cpp
ml::MeanRegressor baseline;
baseline.fit(split.train.y);

ml::Vector baseline_predictions =
    baseline.predict(split.test.num_samples());

ml::Vector model_predictions =
    model.predict(split.test.X);

ml::RegressionEvaluationInput input{
    split.test.y,
    baseline_predictions,
    model_predictions,
    "LinearRegression",
    "MeanRegressor"
};

ml::RegressionEvaluationReport report =
    ml::run_regression_evaluation(input);
```

Then the result can be inspected:

```cpp
report.comparison.baseline.mse;
report.comparison.model.mse;

report.model_beats_baseline_mse();
report.model_beats_baseline_r2();
```

---

### Relationship with previous Phase 2 components

The reusable evaluation harness depends on the components already introduced in this phase:

```txt
SupervisedDataset
train/test and train/validation/test splits
k-fold splitting
leakage-safe preprocessing
baseline predictor
regression metrics
baseline comparison
```

It does not replace those components.

It organizes the final comparison step so later model phases can reuse the same evaluation pattern.

---

### ML Core implementation goal

For Phase 2, ML Core should introduce a minimal regression evaluation harness with:

```txt
RegressionEvaluationInput
RegressionEvaluationReport
run_regression_evaluation(...)
comparison helper methods
```

The harness should be independent from concrete model classes.

It should support the first serious model phase:

```txt
Phase 3 – Linear Models
```

and later be extended for classification when logistic regression is implemented.

---

## 8. Metrics and Experiment Summary Export

Terminal output is useful while developing, but it is not enough for serious experiments.

When comparing models, we need experiment results that can be saved, inspected, committed when small, and compared later.

A reproducible ML workflow should make it clear:

```txt
which experiment was run
which dataset was used
which split was evaluated
which baseline was used
which model was evaluated
which metrics were produced
whether the model beat the baseline
```

---

### Why exporting summaries matters

If results only appear in the terminal, they are easy to lose.

This creates problems:

```txt
hard to compare runs
hard to inspect previous results
hard to review changes after commits
hard to know whether a model improved over time
hard to document experiment outcomes
```

Exported summaries create a durable record of the evaluation.

They also make later phases easier because every model can produce results in the same format.

---

### What an experiment summary should contain

A minimal regression experiment summary should include:

```txt
experiment_name
dataset_name
split_name
baseline_name
model_name
baseline metrics
model metrics
baseline-vs-model comparison flags
```

For regression, the metric fields should include:

```txt
baseline_mse
model_mse

baseline_rmse
model_rmse

baseline_mae
model_mae

baseline_r2
model_r2
```

The comparison fields should include:

```txt
beats_mse
beats_rmse
beats_mae
beats_r2
```

These boolean fields answer whether the model improved over the baseline for each metric.

---

### CSV export

CSV is useful for structured summaries.

A CSV summary should contain one row per experiment result.

Example columns:

```csv
experiment_name,dataset_name,split_name,baseline_name,model_name,baseline_mse,model_mse,baseline_rmse,model_rmse,baseline_mae,model_mae,baseline_r2,model_r2,beats_mse,beats_rmse,beats_mae,beats_r2
```

CSV is useful because it can be opened by:

```txt
spreadsheet tools
Python scripts
data analysis notebooks
simple text editors
```

In ML Core, CSV summaries should stay small and focused.

They should capture the final metrics and comparison flags, not large prediction vectors or full datasets.

---

### TXT export

TXT is useful for human-readable summaries.

A TXT summary should be easy to inspect directly.

Example structure:

```txt
Experiment: ...
Dataset: ...
Split: ...

Baseline: ...
Model: ...

Baseline metrics:
  MSE:
  RMSE:
  MAE:
  R2:

Model metrics:
  MSE:
  RMSE:
  MAE:
  R2:

Comparison:
  Beats baseline on MSE:
  Beats baseline on RMSE:
  Beats baseline on MAE:
  Beats baseline on R2:
```

TXT summaries are useful for quick review and experiment notes.

They should be readable without requiring another tool.

---

### What should not be exported here

This Phase 2 export layer should stay minimal.

It should not export:

```txt
full datasets
large prediction arrays
large intermediate matrices
large generated artifacts
plots
model checkpoints
binary files
```

Those may be added later if needed, but they should not be part of the core metrics summary layer.

The goal is not to build a full experiment tracking platform.

The goal is to provide small, reusable, version-friendly summaries.

---

### Suggested output location

Experiment summaries should go under:

```txt
outputs/
```

For Phase 2 evaluation methodology experiments:

```txt
outputs/phase-2-evaluation-methodology/
```

Example files:

```txt
outputs/phase-2-evaluation-methodology/regression_summary.csv
outputs/phase-2-evaluation-methodology/regression_summary.txt
```

Small representative summaries can be committed if they help document the project.

Large or disposable generated outputs should not be committed.

---

### Relationship with the evaluation harness

The export layer should build on the reusable evaluation harness.

The evaluation harness produces a structured report:

```txt
RegressionEvaluationReport
```

The export layer wraps that report with experiment metadata:

```txt
experiment_name
dataset_name
split_name
```

Together, they form:

```txt
RegressionExperimentSummary
```

Conceptually:

```txt
targets + predictions
        ↓
run_regression_evaluation(...)
        ↓
RegressionEvaluationReport
        ↓
RegressionExperimentSummary
        ↓
CSV / TXT export
```

This keeps responsibilities separated:

```txt
metrics:
    compute metric values

evaluation harness:
    compare baseline and model

experiment summary:
    attach metadata and export results
```

---

### ML Core implementation goal

For Phase 2, ML Core should introduce:

```txt
RegressionExperimentSummary
export_regression_summary_csv(...)
export_regression_summary_txt(...)
```

The export functions should validate:

```txt
experiment_name is not empty
dataset_name is not empty
split_name is not empty
baseline_name is not empty
model_name is not empty
output_path is not empty
output file can be opened
```

This gives later model phases a standard way to save evaluation outputs without designing export logic again.

---

## 9. Proper Evaluation Discipline

Proper evaluation discipline is the set of rules that keeps model evaluation honest, reproducible, and comparable across experiments.

The goal is to make sure that a model's reported performance reflects its ability to generalize to unseen data, not its ability to exploit accidental leakage, repeated test-set usage, or inconsistent evaluation choices.

This section summarizes the evaluation rules that all later ML Core phases should follow.

---

### Core evaluation rule

The central rule is:

```txt
Models must be compared using the same data split, same preprocessing discipline, same metrics, and same baseline.
```

If any of those elements differ, the comparison may not be meaningful.

For example, comparing two models is not fair if:

```txt
model A uses one train/test split
model B uses a different train/test split

model A is evaluated after leakage-safe preprocessing
model B is evaluated after preprocessing fitted on the full dataset

model A is compared using MSE
model B is compared using MAE

model A is compared against a baseline
model B is reported without a baseline
```

Evaluation discipline exists to avoid these inconsistencies.

---

### Standard ML Core evaluation workflow

The default ML Core regression workflow is:

```txt
1. Start with a supervised dataset:
       X, y

2. Split the data:
       train / validation / test
   or:
       train / test + k-fold cross-validation on train

3. Fit preprocessing on training data only.

4. Transform train, validation, and test using training preprocessing parameters.

5. Fit the baseline predictor using training targets only.

6. Train the model using training data only.

7. Use validation data or cross-validation for model choices.

8. Generate baseline predictions and model predictions on the same evaluation split.

9. Evaluate both predictors using the same metrics.

10. Compare model metrics against baseline metrics.

11. Export experiment summaries.

12. Use the test set once for final reporting.
```

This workflow is the default unless a later phase explicitly defines a different evaluation setting.

---

### Train, validation, and test responsibilities

Each split has a different role.

```txt
training set:
    fit model parameters
    fit preprocessing parameters
    fit baseline predictor

validation set:
    compare model choices
    tune hyperparameters
    inspect whether the model is worth keeping

test set:
    final evaluation only
    should not guide model design decisions
```

The test set must remain untouched until the final evaluation.

If the test set is used repeatedly to choose a model, it becomes part of the model-selection process and stops being a clean estimate of generalization.

---

### Cross-validation discipline

K-fold cross-validation should normally be applied only to the training portion of the data.

Correct workflow:

```txt
1. split full dataset into train/test
2. run k-fold cross-validation on train only
3. choose model configuration using cross-validation results
4. train final model on the full training set
5. evaluate once on the test set
```

Incorrect workflow:

```txt
1. run k-fold cross-validation on the full dataset
2. choose model configuration
3. report the cross-validation result as final test performance
```

Cross-validation helps estimate model behavior during development, but it does not replace an untouched final test set.

---

### Preprocessing discipline

Preprocessing must follow the fit/transform separation.

Correct:

```txt
fit scaler on X_train
transform X_train using training statistics
transform X_validation using training statistics
transform X_test using training statistics
```

Incorrect:

```txt
fit scaler on full X
split transformed data afterward
```

Incorrect:

```txt
fit_transform X_validation
fit_transform X_test
```

The validation and test sets must never be used to learn preprocessing statistics.

In ML Core, this is why fitted preprocessing objects exist:

```cpp
ml::StandardScaler scaler;

ml::Matrix X_train_scaled = scaler.fit_transform(split.train.X);
ml::Matrix X_validation_scaled = scaler.transform(split.validation.X);
ml::Matrix X_test_scaled = scaler.transform(split.test.X);
```

---

### Baseline discipline

Every trained model should be compared against a simple baseline.

For regression, the initial baseline is:

```txt
MeanRegressor
```

The baseline must be fitted on the same training target vector used by the model:

```txt
baseline.fit(y_train)
```

The baseline and trained model must be evaluated on the same target vector:

```txt
baseline_predictions evaluated against y_eval
model_predictions evaluated against y_eval
```

A trained model is only useful if it improves over the baseline under the same evaluation protocol.

---

### Metric discipline

Metrics must be chosen before comparing models.

For regression, ML Core currently uses:

```txt
MSE
RMSE
MAE
R²
```

The interpretation is:

```txt
MSE:
    lower is better

RMSE:
    lower is better

MAE:
    lower is better

R²:
    higher is better
```

Metric comparisons must always use the same prediction vector length and the same target vector.

A model should not be judged by changing metrics after seeing the results.

---

### Reproducibility discipline

Randomized operations should expose a seed.

This applies to:

```txt
train/test split
train/validation/test split
k-fold cross-validation
future randomized models or optimizers
```

A reproducible experiment should make it possible to rerun:

```txt
same dataset
same split strategy
same seed
same preprocessing
same model configuration
same metrics
```

and obtain the same evaluation result, except where a later algorithm explicitly introduces controlled randomness.

---

### Export discipline

Important experiment results should be exported in a small, structured format.

Terminal output is not enough.

ML Core uses experiment summaries to record:

```txt
experiment_name
dataset_name
split_name
baseline_name
model_name
baseline metrics
model metrics
baseline-vs-model comparison flags
```

Small CSV or TXT summaries can be kept as representative artifacts when useful.

Large generated outputs, disposable test files, large datasets, and bulky intermediate artifacts should not be committed by default.

---

### Anti-leakage discipline

Before trusting an evaluation result, check:

```txt
Was the data split before preprocessing was fitted?

Were validation/test data excluded from fitting preprocessing?

Was the test set used only once at the end?

Were hyperparameters chosen using validation or cross-validation, not test data?

Were duplicate or related samples handled correctly?

For time-ordered data, was chronological order respected?

Were all features available at prediction time?

Was the model compared against a baseline?

Were metrics chosen before comparison?

Was the experiment summary exported?
```

If any answer is no, the evaluation may be biased.

---

### What later phases must do

Later model phases should not redesign evaluation from scratch.

A future regression model should plug into the Phase 2 evaluation flow like this:

```txt
1. produce predictions
2. compare predictions against a baseline
3. compute standard regression metrics
4. export an experiment summary
```

The model implementation should focus on learning and prediction.

The evaluation system should handle:

```txt
metrics
baseline comparison
report structure
summary export
```

This separation keeps ML Core modular and makes model comparisons consistent across phases.

---

### Phase 2 completion rule

Phase 2 is complete when ML Core has:

```txt
dataset abstraction
train/test and train/validation/test splitting
k-fold cross-validation
leakage-safe preprocessing
explicit data leakage rules
baseline evaluation flow
reusable evaluation harness
metrics and summary export
proper evaluation discipline
```

After this, Phase 3 can implement linear models without redefining how datasets, splits, preprocessing, metrics, baselines, or evaluation reports work.