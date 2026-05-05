# Trees and Ensembles

This document covers the theory needed for Phase 6 of ML Core.

Phase 6 introduces non-linear tabular models through decision trees and ensemble intuition.

The implementation target is a real simple Decision Tree, not only isolated impurity utilities.

The goal is to understand and implement:

```txt
split quality
Gini impurity
entropy
weighted child impurity
recursive tree construction
stopping rules
pruning intuition
random forest intuition
gradient boosting intuition
```

---

## 1. Split Quality, Gini, Entropy, and Weighted Child Impurity

A decision tree learns by repeatedly splitting a dataset into smaller regions.

At each internal node, the tree chooses a rule such as:

```txt
feature_j <= threshold
```

This rule divides the current samples into two child groups:

```txt
left child:
    samples where feature_j <= threshold

right child:
    samples where feature_j > threshold
```

The purpose of a split is to make the child nodes more “pure” than the parent node.

For classification, a pure node contains samples from only one class.

For regression, a pure node contains targets with low variation.

In this phase, the first implementation target should focus on classification trees.

---

### Node impurity

Impurity measures how mixed the class labels are inside a node.

Suppose a node contains samples from `K` classes.

Let:

```txt
p_k = proportion of samples in the node that belong to class k
```

A node is pure when one class has probability `1.0` and all others have probability `0.0`.

A node is highly impure when classes are mixed.

---

### Gini impurity

Gini impurity is defined as:

$$
Gini = 1 - \sum_k(p_{k}^{2})
$$

Interpretation:

```txt
Gini = 0:
    the node is pure

higher Gini:
    the node is more mixed
```

Example:

```txt
Node labels:
    [0, 0, 0, 0]

Class proportions:
    p0 = 1.0
    p1 = 0.0

Gini:
    1 - (1.0^2 + 0.0^2) = 0
```

This node is perfectly pure.

Another example:

```txt
Node labels:
    [0, 0, 1, 1]

Class proportions:
    p0 = 0.5
    p1 = 0.5

Gini:
    1 - (0.5^2 + 0.5^2)
    = 1 - 0.5
    = 0.5
```

This node is mixed.

For binary classification, Gini impurity is maximal when both classes are equally represented.

---

### Entropy

Entropy is another impurity measure.

It is defined as:

$$
Entropy = - \sum_k p_k * \log_{2}{(p_k)}
$$

Terms where $p_k = 0$ are treated as contributing `0`.

Interpretation:

```txt
Entropy = 0:
    the node is pure

higher entropy:
    the node is more uncertain / mixed
```

Example:

```txt
Node labels:
    [0, 0, 0, 0]

Entropy:
    -(1.0 * log2(1.0)) = 0
```

Another example:

```txt
Node labels:
    [0, 0, 1, 1]

Entropy:
    -(0.5 * log2(0.5) + 0.5 * log2(0.5))
    = 1
```

For binary classification, entropy is maximal when both classes have probability `0.5`.

---

### Gini vs entropy

Gini and entropy usually produce similar split choices.

Important differences:

```txt
Gini:
    simpler to compute
    often slightly faster
    common default in CART-style trees

Entropy:
    connected to information theory
    used in information gain
    can be slightly more sensitive to class probability changes
```

For ML Core, Gini should be the first implementation target because it is simple and enough to build a serious first tree.

Entropy can be implemented as an additional split criterion if useful.

---

### Parent impurity

Before evaluating a split, compute the impurity of the current node.

Example:

```txt
parent_labels = [0, 0, 1, 1, 1, 1]
```

The parent impurity measures how mixed the node is before splitting.

A split is useful only if the child nodes are less impure on average than the parent.

---

### Weighted child impurity

After a split, the dataset is divided into left and right children.

Each child has its own impurity.

However, the two children may contain different numbers of samples.

Therefore, the split quality must use a weighted average:

```txt
weighted_child_impurity =
    (n_left / n_total) * impurity_left
  + (n_right / n_total) * impurity_right
```

where:

```txt
n_left = number of samples in the left child
n_right = number of samples in the right child
n_total = n_left + n_right
```

A split is good when this weighted child impurity is small.

---

### Impurity reduction

Impurity reduction measures how much a split improves the node:

```txt
impurity_reduction =
    parent_impurity - weighted_child_impurity
```

A larger reduction means a better split.

If impurity reduction is zero or negative, the split does not improve the node.

In practice, the tree should reject splits that do not produce enough improvement.

This is usually controlled by a parameter such as:

```txt
min_impurity_decrease
```

---

### Example split evaluation

Suppose a parent node has labels:

```txt
[0, 0, 0, 1, 1, 1]
```

Parent Gini:

```txt
p0 = 3/6 = 0.5
p1 = 3/6 = 0.5

Gini_parent = 1 - (0.5^2 + 0.5^2)
            = 0.5
```

A candidate split produces:

```txt
left labels:
    [0, 0, 0]

right labels:
    [1, 1, 1]
```

Left Gini:

```txt
0
```

Right Gini:

```txt
0
```

Weighted child impurity:

```txt
(3/6) * 0 + (3/6) * 0 = 0
```

Impurity reduction:

```txt
0.5 - 0 = 0.5
```

This is an excellent split.

---

### Bad split example

Same parent:

```txt
[0, 0, 0, 1, 1, 1]
```

Candidate split:

```txt
left labels:
    [0, 1, 1]

right labels:
    [0, 0, 1]
```

Both children are still mixed.

The weighted child impurity remains high, so the impurity reduction is small.

This split should not be preferred.

---

### Candidate thresholds

For numerical features, a tree must decide which threshold to test.

For one feature column:

```txt
x_j = [1.0, 2.0, 4.0, 8.0]
```

possible thresholds can be placed between sorted unique values:

```txt
1.5, 3.0, 6.0
```

Each threshold defines a split:

```txt
x_j <= threshold
x_j > threshold
```

The tree evaluates all candidate thresholds and chooses the one with the best impurity reduction.

---

### Split quality in ML Core

The first implementation should expose reusable split scoring functions.

Minimum useful components:

```txt
gini_impurity(y)
entropy(y)
weighted_child_impurity(left_y, right_y)
impurity_reduction(parent_y, left_y, right_y)
evaluate_candidate_threshold(X, y, feature_index, threshold)
find_best_split(X, y)
```

The tree builder should then use these utilities instead of embedding all split logic directly in recursive construction.

This keeps the design modular and testable.

---

## 2. Recursive Tree Construction and Stopping Rules

A decision tree is built recursively.

Each node receives a subset of the training data.

At that node, the algorithm decides whether to:

```txt
stop and create a leaf
```

or:

```txt
find the best split and create child nodes
```

---

### Tree node types

A decision tree contains two main node types.

#### Internal node

An internal node contains a split rule:

```txt
feature_index
threshold
left child
right child
```

Prediction follows the rule:

```txt
if x[feature_index] <= threshold:
    go left
else:
    go right
```

#### Leaf node

A leaf node contains a prediction.

For classification, the prediction is usually:

```txt
majority class among training samples in the leaf
```

The leaf may also store class probabilities:

```txt
class_probability_k = count(class k in leaf) / samples_in_leaf
```

For the first ML Core tree, majority-class prediction is enough.

---

### Recursive construction

The high-level recursive algorithm is:

```txt
build_node(X, y, depth):

    if stopping condition is met:
        return leaf node

    best_split = find_best_split(X, y)

    if no valid split exists:
        return leaf node

    split X and y into left and right subsets

    left_child = build_node(X_left, y_left, depth + 1)
    right_child = build_node(X_right, y_right, depth + 1)

    return internal node(best_split, left_child, right_child)
```

This process continues until every branch reaches a stopping condition.

---

### Why recursion fits trees naturally

A tree is a recursive structure.

Each child of a decision node is itself another decision tree.

This makes recursive construction natural:

```txt
a tree is either:
    a leaf
or:
    a split plus two subtrees
```

This same idea should be reflected in the C++ implementation.

For example:

```txt
TreeNode:
    is_leaf
    prediction
    feature_index
    threshold
    left child
    right child
```

---

### Stopping rules

Stopping rules prevent the tree from growing forever.

They also control overfitting.

Common stopping rules include:

```txt
maximum depth reached
node is pure
too few samples in node
split creates child nodes that are too small
impurity improvement is too small
no valid split exists
```

---

### Maximum depth

`max_depth` limits how many levels the tree can grow.

Example:

```txt
max_depth = 1:
    decision stump

max_depth = 3:
    small tree

max_depth = unlimited:
    tree may grow until leaves are pure
```

A deeper tree can model more complex patterns.

However, deeper trees are more likely to overfit.

---

### Pure node stopping

If all samples in a node have the same class, there is nothing left to split.

Example:

```txt
y = [1, 1, 1, 1]
```

The node is pure.

The tree should create a leaf predicting class `1`.

---

### Minimum samples per split

`min_samples_split` prevents splitting tiny nodes.

Example:

```txt
min_samples_split = 4
```

If a node has fewer than 4 samples, it cannot be split.

This helps reduce overfitting and avoids meaningless splits.

---

### Minimum samples per leaf

`min_samples_leaf` prevents candidate splits that create very small leaves.

Example:

```txt
min_samples_leaf = 2
```

A candidate split is invalid if either child has fewer than 2 samples.

This avoids rules that isolate one sample just to improve training accuracy.

---

### Minimum impurity decrease

`min_impurity_decrease` requires a split to improve impurity by at least some amount.

Example:

```txt
min_impurity_decrease = 1e-7
```

If the best split improves impurity by less than this threshold, the node becomes a leaf.

This avoids adding splits that technically improve the metric but do not meaningfully improve the model.

---

### No valid split

Sometimes no valid split exists.

This can happen when:

```txt
all feature values are identical
all candidate splits violate min_samples_leaf
all splits produce no impurity improvement
```

In this case, the node should become a leaf.

---

### Leaf prediction

For classification, the leaf prediction should be the majority class.

Example:

```txt
leaf labels = [0, 1, 1, 1, 2]

class counts:
    0 -> 1
    1 -> 3
    2 -> 1

prediction:
    1
```

If there is a tie, the implementation should use a deterministic rule.

For example:

```txt
choose the smallest class index among tied classes
```

Determinism matters for tests and reproducibility.

---

### Overfitting and tree depth

Decision trees can overfit easily.

A fully grown tree can keep splitting until leaves are pure.

This may produce excellent training performance but poor validation/test performance.

A shallow tree may underfit, while a very deep tree may overfit.

Therefore, Phase 6 experiments should compare behavior under settings such as:

```txt
max_depth = 1
max_depth = 2
max_depth = 4
max_depth = unlimited or large
```

and inspect training/test behavior.

---

### Recursive tree construction in ML Core

The first implementation should focus on a classification tree with numerical features.

The recommended minimal design is:

```txt
DecisionTreeClassifier:
    user-facing model

TreeNode:
    internal node / leaf representation

split_scoring:
    impurity and split scoring utilities

tree_builder:
    recursive construction logic
```

The public model should expose:

```txt
fit(X, y)
predict(X)
```

The internal builder should handle:

```txt
find best split
apply stopping rules
create leaf
create internal node
recurse
```

---

## 3. Pruning Intuition

Pruning is the process of reducing tree complexity after or during training.

The goal is to improve generalization.

A tree that grows too deep can memorize training data.

Pruning removes or prevents branches that are unlikely to help on unseen data.

---

### Why pruning matters

Decision trees are high-variance models.

This means small changes in the dataset can produce very different tree structures.

A deep tree can create highly specific rules such as:

```txt
if feature_3 <= 7.41 and feature_1 > 2.13 and feature_5 <= 0.02:
    predict class 1
```

Such rules may fit training samples well but fail to generalize.

Pruning controls this behavior.

---

### Pre-pruning

Pre-pruning means stopping the tree early during construction.

This is also called early stopping for trees.

Examples:

```txt
max_depth
min_samples_split
min_samples_leaf
min_impurity_decrease
```

These rules prevent unnecessary branches from being created.

Pre-pruning is simple and should be the main approach in ML Core Phase 6.

---

### Post-pruning

Post-pruning means growing a larger tree first, then removing branches afterward.

A common idea is:

```txt
1. grow a large tree
2. evaluate whether replacing a subtree with a leaf improves validation behavior
3. prune the subtree if the simpler version is better
```

Post-pruning can be powerful but requires more machinery.

It usually needs:

```txt
validation data
subtree scoring
complexity penalty
tree traversal utilities
```

For Phase 6, post-pruning should remain conceptual unless it becomes strategically necessary later.

---

### Cost-complexity intuition

A tree can be evaluated using both error and complexity.

A simplified objective is:

```txt
score = training_error + alpha * number_of_leaves
```

where:

```txt
alpha = complexity penalty
```

If `alpha` is small, complex trees are allowed.

If `alpha` is large, simpler trees are preferred.

This is the intuition behind cost-complexity pruning.

The exact algorithm does not need to be implemented in this phase, but the concept matters.

---

### Pruning and bias-variance tradeoff

Pruning changes the bias-variance balance.

A very shallow tree:

```txt
higher bias
lower variance
may underfit
```

A very deep tree:

```txt
lower bias
higher variance
may overfit
```

A well-pruned tree tries to find a useful compromise.

This connects directly to the statistical learning foundations from Phase 1.

---

### Practical Phase 6 decision

In ML Core Phase 6:

```txt
implement:
    pre-pruning / stopping rules

document conceptually:
    post-pruning
    cost-complexity pruning
```

This gives the project a real usable tree without turning Phase 6 into a full tree-library implementation.

---

## 4. Ensembles: Random Forest and Gradient Boosting Intuition

Decision trees are useful by themselves, but many of the strongest tabular ML methods are ensembles of trees.

An ensemble combines multiple models to produce a stronger predictor.

The main tree ensemble families are:

```txt
random forests
gradient boosting
```

Both use decision trees, but they combine them in very different ways.

---

## 4.1 Random Forest Intuition

A random forest is an ensemble of decision trees trained with randomness.

Each tree is trained on a slightly different view of the data.

The final prediction aggregates predictions from many trees.

For classification:

```txt
forest prediction = majority vote among trees
```

For regression:

```txt
forest prediction = average prediction across trees
```

---

### Bagging

Random forests are based on bagging.

Bagging means bootstrap aggregating.

A bootstrap sample is created by sampling from the training set with replacement.

Example:

```txt
original samples:
    [A, B, C, D, E]

bootstrap sample:
    [A, C, C, E, B]
```

Some samples may appear multiple times.

Some samples may not appear at all.

Each tree receives a different bootstrap sample, so each tree is different.

---

### Feature randomness

Random forests add another source of randomness.

At each split, a tree considers only a random subset of features.

This prevents all trees from choosing the same strongest feature early.

As a result, the trees become less correlated with each other.

Less correlated trees produce a stronger ensemble when averaged or voted.

---

### Why random forests work

A single deep decision tree has high variance.

It may overfit to small details in the training data.

A random forest reduces variance by averaging many different trees.

The intuition is:

```txt
individual trees:
    noisy and high variance

forest average/vote:
    more stable and better generalization
```

Random forests usually work well on tabular data with little preprocessing.

They are strong baselines.

---

### Random forest tradeoffs

Advantages:

```txt
strong tabular baseline
handles nonlinear feature interactions
less overfitting than a single deep tree
works with little feature scaling
can estimate feature importance
```

Disadvantages:

```txt
larger memory footprint
less interpretable than a single tree
slower prediction than one tree
not as flexible as boosting in many competitions
```

For ML Core, random forests should be understood conceptually.

A full implementation can be deferred unless needed later.

---

## 4.2 Gradient Boosting Intuition

Gradient boosting is another tree ensemble method.

Instead of training trees independently, boosting trains trees sequentially.

Each new tree tries to correct the errors of the current ensemble.

The model is built additively:

```txt
prediction = tree_1 + tree_2 + tree_3 + ... + tree_M
```

For classification, the additive model usually works in logit or score space.

For regression, it often works directly on residual-like quantities.

---

### Boosting as error correction

A simple regression intuition:

```txt
1. start with a simple prediction
2. compute residuals
3. train a small tree to predict residuals
4. add this tree to the model
5. repeat
```

Each new tree focuses on what the previous ensemble still gets wrong.

This is different from random forests, where trees are trained mostly independently.

---

### Gradient boosting and loss functions

Gradient boosting generalizes residual correction by using gradients of a loss function.

At each step, the algorithm computes the direction in which predictions should change to reduce the loss.

Then it trains a tree to approximate that direction.

This is why it is called gradient boosting.

The “gradient” is with respect to model predictions, not directly with respect to tree parameters.

---

### Learning rate in boosting

Boosting usually uses a learning rate, sometimes called shrinkage.

Each new tree is scaled before being added:

```txt
ensemble = ensemble + learning_rate * new_tree
```

A smaller learning rate usually requires more trees but can improve generalization.

This resembles the optimization ideas from Phase 5.

---

### Why gradient boosting works well

Gradient boosting is powerful because it builds a strong model from many weak learners.

Each tree can be shallow.

The ensemble becomes expressive through many correction steps.

Modern gradient boosting libraries are among the strongest methods for structured tabular data.

Examples include:

```txt
XGBoost
LightGBM
CatBoost
```

ML Core does not need to reimplement these systems.

The important goal is to understand the idea.

---

### Random forest vs gradient boosting

Random forests and gradient boosting both use many trees, but their logic differs.

| Method | Training style | Main idea | Main strength |
|---|---|---|---|
| Random Forest | Parallel / independent trees | Average many noisy trees | Variance reduction |
| Gradient Boosting | Sequential trees | Correct previous errors | Bias reduction and strong predictive power |

Random forests reduce instability by averaging.

Gradient boosting builds a strong predictor step by step.

---

### Phase 6 ensemble scope

For Phase 6, the implementation priority is:

```txt
implement:
    one simple Decision Tree properly

document conceptually:
    random forest
    gradient boosting

optional:
    small comparison note against external libraries or saved outputs
```

This keeps the phase focused.

The goal is not to build a production ensemble library.

The goal is to understand trees deeply enough that ensemble methods make sense.

---

## 5. Phase 6 Tree-Building Workflow

Phase 6 should not be implemented as a collection of isolated tree utilities.

The goal is to build a real `DecisionTreeClassifier` workflow where each utility exists because it supports the full training and prediction process.

The implementation should follow this architecture:

```txt
DecisionTreeClassifier:
    public model API
    owns fit/predict behavior
    stores the learned tree root

TreeNode:
    internal representation of one node
    can be either a leaf node or an internal split node

split_scoring:
    impurity functions
    weighted child impurity
    impurity reduction
    candidate split evaluation

tree_builder:
    recursive tree construction
    stopping rules
    leaf creation
    internal node creation

prediction traversal:
    walks from root to leaf for each input sample
    returns the leaf prediction
```

The central design idea is:

```txt
split-scoring utilities are not the final product
they are support functions used by the recursive tree builder
```

A phase that only implements Gini, entropy, and split scoring would be incomplete.

Phase 6 is complete only when the project contains a tree that can:

```txt
fit(X, y)
build nodes recursively
apply stopping rules
store split decisions
traverse the tree
predict classes for new samples
```

---

### Public model: DecisionTreeClassifier

The public model should be the main user-facing class.

Recommended public API:

```cpp
DecisionTreeClassifier();

explicit DecisionTreeClassifier(DecisionTreeOptions options);

void fit(const Matrix& X, const Vector& y);

Vector predict(const Matrix& X) const;

bool is_fitted() const;

const DecisionTreeOptions& options() const;
```

The model should hide recursive construction details from the user.

The user should only need:

```cpp
DecisionTreeClassifier tree(options);
tree.fit(X_train, y_train);
Vector predictions = tree.predict(X_test);
```

This keeps the API consistent with the previous model classes:

```txt
LinearRegression
LogisticRegression
SoftmaxRegression
```

---

### Options: DecisionTreeOptions

Tree behavior should be controlled by an options struct.

Recommended initial fields:

```cpp
struct DecisionTreeOptions {
    std::size_t max_depth{3};
    std::size_t min_samples_split{2};
    std::size_t min_samples_leaf{1};
    double min_impurity_decrease{0.0};
};
```

These options define the pre-pruning behavior.

They should be validated before training.

Invalid configurations should throw `std::invalid_argument`.

Examples:

```txt
max_depth = 0:
    invalid or interpreted carefully as stump-only behavior

min_samples_split < 2:
    invalid

min_samples_leaf < 1:
    invalid

min_impurity_decrease < 0:
    invalid
```

For Phase 6, keep the options simple and deterministic.

Do not add random feature selection yet. That belongs naturally to Random Forest later.

---

### Internal node representation: TreeNode

The tree should be represented as nodes.

A node should be either:

```txt
leaf node:
    stores prediction

internal node:
    stores split rule and children
```

Recommended conceptual structure:

```cpp
struct TreeNode {
    bool is_leaf{true};

    double prediction{0.0};

    Eigen::Index feature_index{-1};
    double threshold{0.0};

    double impurity{0.0};
    double impurity_decrease{0.0};
    std::size_t num_samples{0};

    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
};
```

For a leaf:

```txt
is_leaf = true
prediction is used
left/right are null
```

For an internal node:

```txt
is_leaf = false
feature_index and threshold are used
left/right are non-null
```

This representation makes recursive construction and prediction traversal straightforward.

---

### Split candidate representation

Split search should return structured information.

Recommended structure:

```cpp
struct SplitCandidate {
    bool valid{false};

    Eigen::Index feature_index{-1};
    double threshold{0.0};

    double parent_impurity{0.0};
    double left_impurity{0.0};
    double right_impurity{0.0};
    double weighted_child_impurity{0.0};
    double impurity_decrease{0.0};

    std::size_t left_count{0};
    std::size_t right_count{0};
};
```

This avoids returning loose values such as only:

```txt
feature_index
threshold
score
```

A structured result makes tests clearer and helps debugging.

It also allows exported experiments to report split quality later.

---

### Split-scoring layer

The split-scoring layer should provide pure functions.

Recommended functions:

```cpp
double gini_impurity(const Vector& y);

double entropy(const Vector& y);

double weighted_child_impurity(
    double left_impurity,
    double right_impurity,
    std::size_t left_count,
    std::size_t right_count
);

double impurity_reduction(
    double parent_impurity,
    double weighted_child_impurity
);
```

These functions should not know about the full tree.

They should only compute impurity and split-quality values.

This makes them easy to test independently.

---

### Candidate threshold evaluation

After basic impurity functions, implement candidate split evaluation.

Recommended function:

```cpp
SplitCandidate evaluate_candidate_threshold(
    const Matrix& X,
    const Vector& y,
    Eigen::Index feature_index,
    double threshold,
    const DecisionTreeOptions& options
);
```

This function should:

```txt
1. split y into left/right groups using X[:, feature_index] <= threshold
2. reject the split if either child is empty
3. reject the split if either child violates min_samples_leaf
4. compute parent impurity
5. compute left/right impurity
6. compute weighted child impurity
7. compute impurity decrease
8. reject the split if impurity decrease < min_impurity_decrease
9. return a valid SplitCandidate otherwise
```

This function should not recursively build the tree.

It only evaluates one possible split.

---

### Best split selection

The best-split search should evaluate all valid candidate thresholds.

Recommended function:

```cpp
SplitCandidate find_best_split(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options
);
```

It should:

```txt
for each feature:
    collect sorted unique feature values
    create midpoint thresholds between consecutive unique values
    evaluate each threshold
    keep the valid split with the largest impurity decrease
```

If no valid split exists, return:

```txt
SplitCandidate.valid = false
```

Tie-breaking should be deterministic.

Recommended tie-breaking order:

```txt
1. larger impurity decrease wins
2. smaller feature index wins
3. smaller threshold wins
```

Determinism matters because tests should not depend on accidental iteration behavior.

---

### Recursive builder

The recursive builder should be the core of `fit`.

Conceptually:

```txt
build_node(X, y, depth):

    create node metadata:
        impurity
        num_samples
        majority-class prediction

    if stopping condition is met:
        return leaf

    best_split = find_best_split(X, y, options)

    if best_split is invalid:
        return leaf

    split X and y into left/right subsets

    node becomes internal:
        feature_index = best_split.feature_index
        threshold = best_split.threshold
        impurity_decrease = best_split.impurity_decrease

    node.left = build_node(X_left, y_left, depth + 1)
    node.right = build_node(X_right, y_right, depth + 1)

    return node
```

The stopping rules should be applied before attempting a split when possible.

Core stopping rules:

```txt
max_depth reached
node is pure
num_samples < min_samples_split
no valid split exists
```

Candidate-level stopping rules:

```txt
min_samples_leaf
min_impurity_decrease
```

---

### Prediction traversal

Prediction should not recompute splits.

After training, prediction should only traverse the stored tree.

For each sample:

```txt
start at root

while node is not leaf:
    if sample[feature_index] <= threshold:
        go left
    else:
        go right

return node.prediction
```

This should be implemented as a small internal helper, for example:

```cpp
double predict_one(
    const Vector& sample,
    const TreeNode& node
) const;
```

The public `predict(X)` should call this for each row.

---

### Validation rules

Phase 6 should follow the validation discipline used in previous phases.

The tree should reject:

```txt
empty X
empty y
mismatched X/y sample counts
non-binary or invalid class labels if the first version assumes class indices
invalid options
predict before fit
feature count mismatch at predict time
```

For the first tree implementation, class labels should be numeric class indices:

```txt
0, 1, 2, ...
```

This is consistent with the existing classification utilities.

---

### Minimal implementation order

The recommended implementation order is:

```txt
1. Define DecisionTreeOptions
2. Define SplitCandidate
3. Implement gini_impurity
4. Implement entropy
5. Implement weighted_child_impurity
6. Implement impurity_reduction
7. Implement evaluate_candidate_threshold
8. Implement find_best_split
9. Define TreeNode
10. Implement recursive build_node
11. Implement DecisionTreeClassifier::fit
12. Implement prediction traversal
13. Implement DecisionTreeClassifier::predict
14. Add depth/stopping-rule experiments
```

This order avoids building the public model before the split logic is tested.

It also avoids stopping at isolated utilities without completing the model.

---

### Phase 6 scope boundary

Phase 6 should implement a strong single-tree baseline.

It should not implement Random Forest or Gradient Boosting immediately unless the single-tree implementation is complete and stable.

However, Random Forest and Gradient Boosting remain part of the project roadmap.

After Phase 6, the project should explicitly decide between:

```txt
Option A:
    implement Random Forest and Gradient Boosting immediately

Option B:
    create a dedicated future phase:
        Phase 6B – Tree Ensembles
```

The important point is that ensembles should reuse the tree infrastructure from this phase.

Random Forest should reuse:

```txt
DecisionTreeClassifier
bootstrap sampling
feature subsampling added later
prediction aggregation
```

Gradient Boosting should reuse or extend:

```txt
tree-building logic
shallow trees
residual/gradient targets
additive prediction structure
```

Therefore, the best immediate strategy is:

```txt
build one clean tree first
then build ensembles on top of it
```

---

## Phase 6 Implementation Direction

The theory in this document leads to the following implementation goals:

```txt
1. Implement impurity scoring utilities.
2. Implement weighted child impurity and impurity reduction.
3. Implement candidate-threshold generation/evaluation.
4. Implement best split selection.
5. Implement TreeNode representation.
6. Implement recursive tree construction.
7. Implement stopping rules.
8. Implement tree prediction traversal.
9. Add experiments comparing tree behavior under different depth and stopping settings.
10. Keep random forest and gradient boosting conceptual unless deeper implementation becomes strategically necessary.
```

Phase 6 is complete only when the project contains a real simple Decision Tree implementation.

A few split-scoring utilities alone are not enough.

The target is a tree that can fit data, grow recursively, stop according to explicit rules, and produce predictions through traversal.