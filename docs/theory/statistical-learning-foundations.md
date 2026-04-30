# Statistical Learning Foundations

This document summarizes the statistical learning concepts needed for ML Core before implementing full models and evaluation pipelines.

The goal is not to cover all probability theory. The goal is to define the concepts needed to reason about datasets, generalization, model evaluation, and model behavior.

---

## 1. Why Statistical Learning Matters

Machine Learning is not only about fitting a function to data.

The real goal is to learn patterns that generalize to unseen data.

A model that performs well on the data it saw during training may still perform poorly on new data.

Statistical learning provides the language for this problem:

```txt
data distribution
sample
expectation
variance
covariance
generalization
bias
variance
overfitting
underfitting
```

These concepts are required before serious evaluation and model comparison.

---

## 2. Random Variables and Samples

A random variable represents a quantity whose value depends on chance.

In ML, each feature can be thought of as a random variable.

For example:

```txt
house_size
income
temperature
asset_return
```

A dataset is a finite sample from some underlying data-generating process.

In practice, we observe:

$$
x_1, x_2, \dots, x_m
$$

but we usually care about the broader distribution that produced those observations.

This distinction matters because a model should perform well beyond the exact samples it has already seen.

---

## 3. Expectation

Expectation is the theoretical average value of a random variable.

For a discrete random variable $X$:

$$
\mathbb{E}[X] = \sum_x x \, P(X = x)
$$

For a dataset sample, the empirical mean approximates the expectation:

$$
\bar{x} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

In ML Core, the empirical mean is implemented as:

```cpp
mean(values);
```

For feature matrices, column-wise means approximate the expected value of each feature:

```cpp
column_means(X);
```

---

## 4. Variance

Variance measures how much values spread around their mean.

The population variance is:

$$
\sigma^2 = \mathbb{E}[(X - \mathbb{E}[X])^2]
$$

The empirical population-style variance is:

$$
\sigma^2 = \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu)^2
$$

The sample variance is:

$$
s^2 = \frac{1}{m - 1} \sum_{i=1}^{m}(x_i - \bar{x})^2
$$

Variance matters because:

```txt
high variance feature:
    values are widely spread

low variance feature:
    values barely change

zero variance feature:
    values are constant and carry no sample variation
```

In ML Core, variance is implemented through:

```cpp
variance_population(values);
variance_sample(values);
column_variance_population(X);
column_variance_sample(X);
```

---

## 5. Standard Deviation

The standard deviation is the square root of variance:

$$
\sigma = \sqrt{\sigma^2}
$$

It is useful because it is expressed in the same units as the original data.

Standard deviation is used directly in standardization:

$$
z = \frac{x - \mu}{\sigma}
$$

In ML Core, this is implemented through:

```cpp
standard_deviation_population(values);
standard_deviation_sample(values);
column_standard_deviation_population(X);
column_standard_deviation_sample(X);
standardize_columns(X);
```

---

## 6. Covariance

Covariance measures how two variables vary together.

For two variables $X$ and $Y$:

$$
\operatorname{Cov}(X, Y) =
\mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

Empirically:

$$
\operatorname{Cov}(x, y) =
\frac{1}{m} \sum_{i=1}^{m}(x_i - \bar{x})(y_i - \bar{y})
$$

Interpretation:

```txt
positive covariance:
    variables tend to increase together

negative covariance:
    one variable tends to increase when the other decreases

near-zero covariance:
    no strong linear co-movement
```

Covariance is important for:

```txt
feature relationships
PCA
Gaussian models
portfolio/finance data
correlation analysis
```

ML Core has not implemented covariance yet. It is introduced here conceptually and will become more important in later phases.

---

## 7. Distributions Intuition

A distribution describes how values are likely to appear.

Examples:

```txt
normal-like data
skewed data
heavy-tailed data
binary labels
categorical classes
```

In ML, distributions matter because training and test data are assumed to come from related data-generating processes.

If the training distribution differs strongly from the test distribution, model performance can degrade.

This is called distribution shift.

Examples:

```txt
training on old market data, testing on a new market regime
training on one user population, deploying on another
training on clean data, deploying on noisy data
```

For ML Core, distribution intuition matters for:

```txt
train/test splitting
model evaluation
generalization
probabilistic ML
unsupervised learning
```

---

## 8. Train / Validation / Test Roles

A serious ML workflow separates data into different roles.

### Training Set

The training set is used to fit model parameters.

Example:

```txt
learn weights
learn bias
fit preprocessing parameters
fit model structure
```

### Validation Set

The validation set is used to make modeling decisions.

Examples:

```txt
choose hyperparameters
compare learning rates
choose regularization strength
choose model depth
select threshold
```

The validation set should not be used to directly fit model parameters.

### Test Set

The test set is used only for final evaluation.

It estimates how the selected model might perform on unseen data.

The test set should not influence modeling decisions.

If the test set is used repeatedly to choose models, it stops being a real test set.

---

## 9. Data Leakage

Data leakage happens when information from validation or test data influences training.

Examples:

```txt
standardizing using statistics from the full dataset before splitting
selecting features using test labels
tuning hyperparameters on the test set
duplicated samples across train and test
time-series split that trains on future data
```

Leakage produces overly optimistic evaluation results.

For preprocessing, the correct workflow is:

```txt
1. split data
2. fit preprocessing on training data only
3. apply learned preprocessing to validation/test data
4. train model on training data
5. evaluate on validation/test data
```

This is why Phase 2 will introduce explicit data pipeline and evaluation methodology rules.

---

## 10. Underfitting

Underfitting happens when a model is too simple to capture the structure in the data.

Symptoms:

```txt
high training error
high validation/test error
model predictions are systematically poor
```

Common causes:

```txt
model too simple
not enough features
excessive regularization
not enough training
bad optimization
```

Example:

```txt
using a straight line for a strongly non-linear relationship
```

Underfitting is usually a high-bias problem.

---

## 11. Overfitting

Overfitting happens when a model learns the training data too specifically, including noise or accidental patterns.

Symptoms:

```txt
low training error
high validation/test error
large gap between training and validation performance
```

Common causes:

```txt
model too complex
too many parameters
too little data
weak regularization
data leakage
training too long in some models
```

Example:

```txt
a very deep decision tree that memorizes the training data
```

Overfitting is usually a high-variance problem.

---

## 12. Bias

Bias is error caused by simplifying assumptions in the model.

High bias means the model is not flexible enough to capture the true relationship.

Examples:

```txt
linear model used for a non-linear pattern
too much regularization
missing important features
```

High-bias models tend to underfit.

In practical terms:

```txt
high bias:
    train error high
    validation error high
```

---

## 13. Variance in the Statistical Learning Sense

Variance, in statistical learning, refers to how sensitive a model is to changes in the training data.

High variance means that small changes in the dataset can produce very different learned models.

Examples:

```txt
deep decision trees
high-degree polynomial models
large unregularized models
```

High-variance models tend to overfit.

In practical terms:

```txt
high variance:
    train error low
    validation error much higher
```

This is different from feature variance, although the ideas are related through the general notion of variability.

---

## 14. Bias-Variance Tradeoff

The bias-variance tradeoff describes the tension between:

```txt
simple models:
    higher bias, lower variance

complex models:
    lower bias, higher variance
```

The goal is not to minimize only training error.

The goal is to find a model that generalizes well.

A useful mental model:

```txt
underfitting:
    model too simple
    high bias

overfitting:
    model too sensitive
    high variance

good fit:
    enough flexibility to learn structure
    enough constraint to avoid memorization
```

---

## 15. Generalization

Generalization is the ability of a model to perform well on unseen data.

A model generalizes well when:

```txt
training performance is good
validation/test performance is also good
the gap between train and validation/test is acceptable
```

A model does not generalize well when:

```txt
training performance is much better than validation/test performance
```

This is why evaluation methodology is not optional. It is central to Machine Learning.

---

## 16. What This Means for ML Core

These concepts directly motivate the next phases.

Phase 2 will implement:

```txt
dataset abstractions
train/validation/test splitting
evaluation discipline
leakage prevention
baseline-vs-model comparison
```

Later phases will use these ideas to analyze:

```txt
linear regression behavior
regularization
classification thresholds
optimization behavior
decision tree depth
PCA variance
probabilistic assumptions
```

---

## 17. Summary

The essential ideas are:

```txt
expectation:
    theoretical average value

empirical mean:
    sample approximation of expectation

variance:
    spread around the mean

covariance:
    joint variation between two variables

distribution:
    pattern of possible values

train set:
    fits model parameters

validation set:
    guides model selection

test set:
    estimates final generalization

underfitting:
    model too simple, high bias

overfitting:
    model too sensitive, high variance

bias:
    error from simplifying assumptions

variance:
    sensitivity to training data

generalization:
    performance on unseen data
```

These concepts define the statistical language needed to evaluate ML models seriously.