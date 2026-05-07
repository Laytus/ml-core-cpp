# Distance and Kernel Thinking

## 1. Purpose of This Document

This document covers the geometric side of classical Machine Learning.

The goal is to understand how models can make predictions from:

```txt
distances
neighborhoods
similarity
margins
feature-space transformations
kernels
```

This phase connects directly to:

```txt
k-NN:
    prediction by nearby samples

SVM:
    prediction by margin-based separation

kernels:
    similarity in transformed feature spaces
```

Phase 7 will implement distance metrics, multivariate k-NN, and simple reusable kernel functions.

SVM and full kernel SVM solvers remain theory + small demo level for this phase.

---

## 2. Multivariate k-NN

k-Nearest Neighbors is one of the simplest supervised learning algorithms.

Unlike linear regression, logistic regression, decision trees, or neural networks, k-NN does not learn explicit parameters during training.

Instead, it stores the training data.

Prediction happens by comparing a new sample to the stored training samples.

High-level algorithm:

```txt
fit:
    store X_train and y_train

predict one sample:
    compute distance from x to every training row
    select the k nearest rows
    aggregate their labels
```

For classification:

```txt
prediction = majority class among k nearest neighbors
```

For regression:

```txt
prediction = average target among k nearest neighbors
```

In Phase 7, the implementation target is:

```txt
KNNClassifier
```

---

## 3. Multivariate Samples

In a real dataset, each sample usually has multiple features.

A sample is a vector:

```txt
x = [x_1, x_2, ..., x_d]
```

where:

```txt
d = number of features
```

A dataset is a matrix:

```txt
X ∈ R^(n × d)
```

where:

```txt
n = number of samples
d = number of features
```

For k-NN, each row of `X` represents a point in `d`-dimensional space.

Prediction depends on which training points are closest to the query point.

---

## 4. Distance Metrics

A distance metric measures how far apart two samples are.

Given two vectors:

```txt
a = [a_1, a_2, ..., a_d]
b = [b_1, b_2, ..., b_d]
```

a distance function returns a non-negative value:

```txt
distance(a, b) >= 0
```

Smaller distance means greater similarity.

In k-NN, the choice of distance metric directly changes which neighbors are selected.

---

## 5. Euclidean Distance

Euclidean distance is the standard straight-line distance.

```txt
euclidean(a, b) =
    sqrt(sum_j (a_j - b_j)^2)
```

In two dimensions, this is the usual geometric distance between points.

Example:

```txt
a = [0, 0]
b = [3, 4]

distance = sqrt(3^2 + 4^2) = 5
```

Euclidean distance is sensitive to large coordinate differences because differences are squared.

This means feature scaling matters a lot.

If one feature has a much larger scale than others, it can dominate the distance.

---

## 6. Squared Euclidean Distance

Squared Euclidean distance removes the square root:

```txt
squared_euclidean(a, b) =
    sum_j (a_j - b_j)^2
```

This preserves the same nearest-neighbor ordering as Euclidean distance because square root is monotonic.

That means:

```txt
if euclidean(a, b) < euclidean(a, c)
then squared_euclidean(a, b) < squared_euclidean(a, c)
```

For k-NN neighbor selection, squared Euclidean distance is often enough.

It is also cheaper to compute because it avoids `sqrt`.

However, if the actual numeric distance is needed, Euclidean distance is more interpretable.

---

## 7. Manhattan Distance

Manhattan distance is the sum of absolute coordinate differences.

```txt
manhattan(a, b) =
    sum_j |a_j - b_j|
```

Example:

```txt
a = [0, 0]
b = [3, 4]

distance = |3| + |4| = 7
```

It is called Manhattan distance because it resembles movement along city blocks.

Compared with Euclidean distance, Manhattan distance can behave differently when features are sparse or when coordinate-wise differences matter independently.

---

## 8. Distance Metric Choice

Different distance metrics can produce different neighbors.

Example:

```txt
query:
    [0, 0]

point A:
    [3, 0]

point B:
    [2, 2]
```

Euclidean:

```txt
distance(query, A) = 3
distance(query, B) = sqrt(8) ≈ 2.83

B is closer
```

Manhattan:

```txt
distance(query, A) = 3
distance(query, B) = 4

A is closer
```

So a k-NN classifier can make different predictions depending on the distance metric.

This is why Phase 7 experiments should compare:

```txt
Euclidean vs Manhattan
```

---

## 9. Feature Scaling and k-NN

k-NN is highly sensitive to feature scale.

Suppose a dataset has:

```txt
age:
    18 to 80

income:
    20,000 to 200,000
```

The income feature can dominate Euclidean distance simply because it has larger numeric values.

That means k-NN often requires preprocessing:

```txt
standardization
min-max normalization
```

This connects to earlier preprocessing pipeline work.

For fair distance comparisons, features should usually be scaled before k-NN.

---

## 10. k-NN Voting

For classification, k-NN predicts by voting among the nearest labels.

Example:

```txt
k = 5

nearest labels:
    [1, 1, 0, 1, 0]

vote counts:
    class 0 -> 2
    class 1 -> 3

prediction:
    class 1
```

If there is a tie, the implementation should use deterministic tie-breaking.

Recommended rule:

```txt
choose the smallest class label among tied classes
```

Example:

```txt
nearest labels:
    [0, 1]

tie:
    class 0 -> 1
    class 1 -> 1

prediction:
    class 0
```

This keeps tests reproducible.

---

## 11. Choosing k

The value of `k` controls the smoothness of the classifier.

Small `k`:

```txt
more flexible
sensitive to noise
can overfit
```

Large `k`:

```txt
smoother decision boundary
less sensitive to individual samples
can underfit
```

Special case:

```txt
k = 1
```

The model predicts the label of the single nearest training sample.

This can fit training data very closely, but it may be unstable.

A larger `k` makes the prediction depend on a local neighborhood instead of a single point.

---

## 12. k-NN as a Non-Parametric Model

k-NN is called non-parametric because it does not learn a fixed-size parameter vector.

Instead, model complexity grows with the dataset.

Training is cheap:

```txt
store the data
```

Prediction is expensive:

```txt
compute distances to many training samples
```

For `n` training samples and `d` features, naive prediction for one sample costs roughly:

```txt
O(n * d)
```

This is acceptable for a learning project but can be expensive for large datasets.

---

## 13. Curse of Dimensionality

The curse of dimensionality refers to problems that appear when the number of dimensions grows.

Distance-based methods are especially affected.

In high dimensions, points tend to become far from each other.

The idea of “nearest” can become less meaningful.

---

## 14. Why High Dimensions Hurt Distances

In low dimensions, neighborhoods are often meaningful.

In two dimensions:

```txt
nearby points are visually close
```

In high dimensions, several things happen:

```txt
distances grow
points become sparse
nearest and farthest distances can become less distinguishable
local neighborhoods become less reliable
```

This makes k-NN harder to use effectively.

---

## 15. Sparsity of Space

As dimensionality increases, the volume of the feature space grows very quickly.

Example:

```txt
1D unit interval:
    length = 1

2D unit square:
    area = 1

3D unit cube:
    volume = 1
```

The total volume remains numerically 1 if using `[0, 1]^d`, but the number of regions needed to cover it grows exponentially.

If each dimension is split into 10 bins:

```txt
1D:
    10 cells

2D:
    100 cells

3D:
    1000 cells

d dimensions:
    10^d cells
```

So data becomes sparse very quickly.

A fixed number of samples covers less and less of the space.

---

## 16. Distance Concentration

In high dimensions, distances can concentrate.

This means the difference between the nearest and farthest neighbors may become small relative to the absolute distances.

Conceptually:

```txt
nearest distance ≈ farthest distance
```

If all points are similarly far away, nearest-neighbor methods lose useful contrast.

This does not mean k-NN never works in high dimensions.

It means distance metrics become more fragile and require careful feature engineering, scaling, or dimensionality reduction.

---

## 17. Curse of Dimensionality and Feature Quality

The curse of dimensionality is not only about the raw number of features.

It is also about whether the features are informative.

A 100-dimensional dataset with meaningful structure can still work.

A 20-dimensional dataset with noisy irrelevant features can fail.

Irrelevant features add distance noise.

If a feature does not help distinguish classes but contributes to distance, it can make nearest neighbors worse.

This is why feature selection and dimensionality reduction matter.

---

## 18. Margins and SVM Intuition

Support Vector Machines are margin-based classifiers.

The basic idea is to find a separating hyperplane that leaves the largest possible margin between classes.

For binary classification, a linear classifier has the form:

```txt
f(x) = w^T x + b
```

Prediction:

```txt
if f(x) >= 0:
    predict +1
else:
    predict -1
```

The decision boundary is:

```txt
w^T x + b = 0
```

This is a line in 2D, a plane in 3D, and a hyperplane in higher dimensions.

---

## 19. What Is a Margin?

The margin is the distance between the decision boundary and the closest training points.

SVM tries to maximize this margin.

The closest points are called support vectors.

They are the samples that most directly determine the boundary.

Intuition:

```txt
small margin:
    boundary is close to training samples
    model may be sensitive

large margin:
    boundary has more separation
    model may generalize better
```

---

## 20. Hard Margin SVM

A hard-margin SVM assumes the data is perfectly linearly separable.

It tries to find a boundary such that all samples are correctly classified with a margin.

This is only possible when the classes can be separated exactly by a hyperplane.

In real datasets, this is often too strict.

Noise or overlapping classes can make hard-margin separation impossible.

---

## 21. Soft Margin SVM

Soft-margin SVM allows some margin violations.

This means some samples can be:

```txt
inside the margin
or even misclassified
```

A regularization parameter controls the trade-off:

```txt
large penalty for violations:
    stricter margin
    less tolerance for errors

small penalty for violations:
    more tolerance
    wider or more flexible margin behavior
```

Soft margin makes SVM more practical for real data.

---

## 22. Hinge Loss Intuition

Linear SVM can be understood through hinge loss.

For labels:

```txt
y_i ∈ {-1, +1}
```

and score:

```txt
f(x_i) = w^T x_i + b
```

the hinge loss is:

```txt
max(0, 1 - y_i * f(x_i))
```

If:

```txt
y_i * f(x_i) >= 1
```

the sample is correctly classified with enough margin, so loss is zero.

If:

```txt
y_i * f(x_i) < 1
```

the sample violates the margin and receives positive loss.

A serious linear SVM implementation would optimize:

```txt
regularization + hinge loss
```

This is feasible, but it is deferred until after the Phase 7 core is complete.

---

## 23. Primal Linear SVM Implementation Notes

The Phase 7 extension implements a focused `LinearSVM` using the primal hinge-loss formulation.

This is different from a full kernel SVM.

A primal linear SVM learns directly:

```txt
w = weight vector
b = bias/intercept
```

and predicts with a linear score:

```txt
score(x) = w^T x + b
```

This is compatible with the rest of ML Core because earlier phases already implemented:

```txt
matrix/vector operations
binary classification utilities
classification metrics
gradient-based optimization intuition
training loss histories
```

The goal is to implement a serious but manageable linear margin classifier, not a full constrained dual SVM solver.

---

### 23.1 Binary label convention

The public API should accept binary labels in the same convention used by the rest of the project:

```txt
0, 1
```

Internally, SVM training should map those labels to:

```txt
0 -> -1
1 -> +1
```

This is useful because the hinge-loss formula is naturally written with labels:

```txt
y_i ∈ {-1, +1}
```

The mapping is:

```txt
y_svm = 2 * y_binary - 1
```

Examples:

```txt
y_binary = 0  ->  y_svm = -1
y_binary = 1  ->  y_svm = +1
```

Validation should reject:

```txt
non-integer labels
negative labels
labels other than 0 or 1
empty target vectors
mismatched X/y row counts
```

---

### 23.2 Linear score and prediction

The model score is:

```txt
score_i = w^T x_i + b
```

Prediction is based on the sign of the score:

```txt
if score_i >= 0:
    predict class 1
else:
    predict class 0
```

The implementation should expose both:

```cpp
Vector decision_function(const Matrix& X) const;

Vector predict(const Matrix& X) const;
```

`decision_function` returns raw signed scores.

`predict` converts those scores into class labels.

This is important because the score contains margin information, while the class prediction only contains the final side of the boundary.

---

### 23.3 Margin condition

For SVM labels:

```txt
y_i ∈ {-1, +1}
```

and score:

```txt
score_i = w^T x_i + b
```

we define the signed margin quantity:

```txt
y_i * score_i
```

The key margin condition is:

```txt
y_i * score_i >= 1
```

Interpretation:

```txt
y_i * score_i >= 1:
    correctly classified with enough margin

0 < y_i * score_i < 1:
    correctly classified but inside the margin

y_i * score_i <= 0:
    misclassified or exactly on the wrong side
```

SVM training only applies the hinge-loss gradient contribution when the margin condition is violated.

---

### 23.4 Hinge loss

For one sample, hinge loss is:

```txt
max(0, 1 - y_i * score_i)
```

If the sample satisfies the margin:

```txt
y_i * score_i >= 1
```

then:

```txt
hinge_loss = 0
```

If the sample violates the margin:

```txt
y_i * score_i < 1
```

then:

```txt
hinge_loss = 1 - y_i * score_i
```

The average hinge loss over the dataset is:

```txt
mean_hinge_loss = (1 / n) * sum_i max(0, 1 - y_i * score_i)
```

---

### 23.5 L2 regularization

A linear SVM should use L2 regularization on the weights.

The regularization term is:

```txt
0.5 * lambda * ||w||^2
```

where:

```txt
lambda >= 0
```

The bias term `b` should not be regularized in the first implementation.

The total training objective is:

```txt
loss = mean_hinge_loss + 0.5 * lambda * ||w||^2
```

Interpretation:

```txt
hinge loss:
    penalizes margin violations

L2 regularization:
    discourages very large weights
    helps control the margin geometry
```

A larger `lambda` creates stronger regularization.

A smaller `lambda` allows the model to fit the training data more aggressively.

---

### 23.6 Subgradient update intuition

Hinge loss is not differentiable exactly at:

```txt
y_i * score_i = 1
```

So training uses a subgradient.

For a sample that satisfies the margin:

```txt
y_i * score_i >= 1
```

there is no hinge-loss contribution.

Only L2 regularization affects the weights:

```txt
grad_w = lambda * w
grad_b = 0
```

For a sample that violates the margin:

```txt
y_i * score_i < 1
```

the gradients are:

```txt
grad_w = lambda * w - y_i * x_i
grad_b = -y_i
```

This is the core learning rule for primal linear SVM with stochastic gradient descent.

---

### 23.7 SGD training loop

The first implementation should use a simple deterministic SGD-style training loop.

High-level algorithm:

```txt
initialize w = zeros
initialize b = 0

for epoch in 1..max_epochs:
    for each training sample in deterministic order:
        score = w^T x_i + b
        margin = y_i * score

        if margin >= 1:
            grad_w = lambda * w
            grad_b = 0
        else:
            grad_w = lambda * w - y_i * x_i
            grad_b = -y_i

        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

    compute full training loss
    append loss to training_loss_history
```

This loop is simple, deterministic, and enough for the first `LinearSVM` model.

Random shuffling can be added later, but the first implementation should prioritize reproducibility.

---

### 23.8 Batch-style alternative

A batch-style implementation is also possible.

For each epoch:

```txt
compute gradients over all samples
average gradients
apply one update
```

This can be easier to reason about mathematically, but SGD is closer to the standard practical intuition for linear SVMs.

For ML Core Phase 7, either is acceptable, but the recommended first implementation is:

```txt
deterministic SGD without shuffling
```

because it is simple and exposes margin-violation logic clearly.

---

### 23.9 LinearSVM options

Recommended first options:

```cpp
struct LinearSVMOptions {
    double learning_rate{0.01};
    std::size_t max_epochs{100};
    double l2_lambda{0.01};
};
```

Validation:

```txt
learning_rate must be finite and positive
max_epochs must be at least 1
l2_lambda must be finite and non-negative
```

The first version does not need random seed support if training order is deterministic.

If shuffling is added later, a `random_seed` option can be introduced.

---

### 23.10 LinearSVM API

Recommended public API:

```cpp
class LinearSVM {
public:
    LinearSVM();

    explicit LinearSVM(LinearSVMOptions options);

    void fit(const Matrix& X, const Vector& y);

    Vector decision_function(const Matrix& X) const;

    Vector predict(const Matrix& X) const;

    bool is_fitted() const;

    const LinearSVMOptions& options() const;

    const Vector& weights() const;

    double bias() const;

    const std::vector<double>& training_loss_history() const;
};
```

Expected behavior:

```txt
fit:
    validate X and y
    map labels from {0, 1} to {-1, +1}
    train with deterministic SGD
    store weights, bias, and loss history

decision_function:
    return raw linear scores

predict:
    return 0/1 class labels based on score sign
```

---

### 23.11 Training loss history

After each epoch, the model should compute and store the full objective:

```txt
loss = mean_hinge_loss + 0.5 * lambda * ||w||^2
```

Expected behavior:

```txt
training_loss_history.size() == max_epochs
```

On simple separable datasets, the loss should generally decrease.

The test dataset should be chosen so this behavior is stable.

---

### 23.12 Relationship to Logistic Regression

Linear SVM and Logistic Regression can both learn linear decision boundaries.

The difference is the loss function.

Logistic Regression uses probabilistic cross-entropy-style loss.

Linear SVM uses hinge loss and focuses directly on margin violations.

Comparison:

```txt
Logistic Regression:
    probabilistic interpretation
    smooth loss
    outputs probabilities

Linear SVM:
    margin-based interpretation
    hinge loss
    outputs decision scores
```

This makes Logistic Regression and LinearSVM useful to compare experimentally.

---

### 23.13 Phase 7 LinearSVM completion criteria

The `LinearSVM` extension is complete when:

```txt
LinearSVMOptions is implemented and validated
LinearSVM fits simple binary data
LinearSVM exposes decision_function and predict
training maps labels from {0, 1} to {-1, +1}
hinge-loss margin logic is used
L2 regularization is included
training loss history is stored
predict-before-fit and invalid inputs are rejected
experiments compare k-NN and LinearSVM behavior
full kernel SVM remains explicitly deferred
```

---

## 24. Why Full Kernel SVM Is Deferred

A full kernel SVM is more complex than a linear SVM.

It usually requires the dual optimization problem and specialized solvers such as SMO.

A complete implementation would involve:

```txt
kernel matrix
alpha coefficients
support vectors
box constraints
KKT conditions
dual objective
numerical tolerance
SMO-style updates
```

This is too large for Phase 7.

Phase 7 will instead implement reusable kernel functions and small kernel demos.

---

## 25. Kernels and Feature-Space Lifting

A kernel is a similarity function.

It computes the dot product between transformed versions of two samples without explicitly constructing the transformed features.

Conceptually:

```txt
K(a, b) = phi(a)^T phi(b)
```

where:

```txt
phi(x) = transformed feature representation
```

The kernel trick is useful because `phi(x)` can be high-dimensional or even infinite-dimensional.

The kernel lets us compute similarity in that transformed space directly.

---

## 26. Linear Kernel

The linear kernel is simply the dot product:

```txt
K(a, b) = a^T b
```

It corresponds to no nonlinear feature lifting.

It measures alignment between vectors.

For normalized vectors, larger dot product means stronger directional similarity.

---

## 27. Polynomial Kernel

The polynomial kernel has the form:

```txt
K(a, b) = (gamma * a^T b + coef0)^degree
```

A simpler version can use:

```txt
K(a, b) = (a^T b + coef0)^degree
```

It allows interactions between features.

For example, a degree-2 polynomial feature map can represent squared terms and pairwise interactions.

This can make non-linear patterns linearly separable in the transformed space.

---

## 28. RBF Kernel

The radial basis function kernel is:

```txt
K(a, b) = exp(-gamma * ||a - b||^2)
```

where:

```txt
gamma > 0
```

The RBF kernel measures local similarity.

If two points are close:

```txt
||a - b||^2 is small
K(a, b) is close to 1
```

If two points are far:

```txt
||a - b||^2 is large
K(a, b) approaches 0
```

RBF kernels are powerful because they can model complex nonlinear structure.

---

## 29. Kernel Similarity Interpretation

Kernel values are not always distances.

They are similarities.

For RBF:

```txt
larger value = more similar
smaller value = less similar
```

For Euclidean distance:

```txt
smaller value = more similar
larger value = less similar
```

So distances and kernels often move in opposite directions:

```txt
near points:
    small distance
    high RBF similarity

far points:
    large distance
    low RBF similarity
```

This is useful for demos.

---

## 30. Phase 7 Implementation Scope

Phase 7 will implement:

```txt
distance metrics:
    euclidean_distance
    squared_euclidean_distance
    manhattan_distance

k-NN:
    KNNClassifier

kernels:
    linear_kernel
    polynomial_kernel
    rbf_kernel
```

Phase 7 also implements a focused primal `LinearSVM` extension after the core distance/k-NN/kernel work.

Phase 7 will not implement a full kernel SVM solver.

---

## 31. Expected Implementation Design

Recommended files:

```txt
include/ml/distance/distance_metrics.hpp
src/distance/distance_metrics.cpp

include/ml/distance/knn_classifier.hpp
src/distance/knn_classifier.cpp

include/ml/distance/kernels.hpp
src/distance/kernels.cpp
include/ml/linear_models/linear_svm.hpp
src/linear_models/linear_svm.cpp
```

Experiments:

```txt
experiments/phase-7-distance-kernel/
outputs/phase-7-distance-kernel/
```

Expected outputs:

```txt
knn_metric_comparison.csv
knn_metric_comparison.txt
kernel_similarity_demo.csv
svm_margin_intuition.txt
linear_svm_comparison.csv
linear_svm_comparison.txt
linear_svm_margin_behavior.csv
```

---

## 32. Completion Criteria

This theory section is complete when the following ideas are clear:

```txt
k-NN predicts from local neighborhoods
distance metric choice changes neighbor selection
feature scaling matters for distance methods
high dimensions weaken naive distance intuition
SVMs are margin-based classifiers
soft margin allows violations
kernels compute similarity in transformed feature spaces
LinearSVM uses primal hinge-loss optimization
LinearSVM exposes decision scores and class predictions
LinearSVM is compared with the k-NN workflow
simple kernel functions are useful without implementing full kernel SVM
```