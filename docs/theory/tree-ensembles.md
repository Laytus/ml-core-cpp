# Tree Ensembles and Advanced Tree Features

This document covers the theory for Phase 6B of ML Core.

Phase 6 implemented a real `DecisionTreeClassifier` with:

```txt
split scoring
candidate threshold evaluation
best split selection
recursive tree growth
stopping rules
prediction traversal
tree behavior experiments
```

Phase 6B extends that foundation.

The goal is to implement advanced tree controls and then build ensemble methods on top of the existing tree infrastructure.

This document starts with the advanced single-tree controls:

```txt
max_leaf_nodes
max_features
class_weight
sample_weight
missing-value handling
cost-complexity pruning
```

These features matter because Random Forest and Gradient Boosting should reuse a strong, configurable tree learner instead of duplicating tree-building logic.

---

## 1. Why Advanced Tree Controls Matter

A basic decision tree can already model non-linear decision boundaries.

However, basic trees are high-variance models.

A tree can easily grow too deep, isolate individual samples, and overfit the training set.

Advanced tree controls improve three things:

```txt
1. regularization
2. robustness
3. ensemble readiness
```

Regularization means controlling model complexity.

Robustness means handling imperfect or imbalanced data.

Ensemble readiness means exposing the controls needed by Random Forest and Gradient Boosting.

In Phase 6, the tree already supports:

```txt
max_depth
min_samples_split
min_samples_leaf
min_impurity_decrease
```

Phase 6B extends this with:

```txt
max_leaf_nodes
max_features
class_weight
sample_weight
missing-value handling
cost-complexity pruning
```

---

## 2. `max_leaf_nodes`

### 2.1 Definition

`max_leaf_nodes` limits the total number of leaf nodes in the tree.

A leaf node is a terminal prediction node.

Example:

```txt
max_leaf_nodes = 4
```

means the final tree can have at most 4 leaves.

This is another way of controlling tree complexity.

---

### 2.2 Difference between `max_depth` and `max_leaf_nodes`

`max_depth` limits how deep the tree can grow.

`max_leaf_nodes` limits how many terminal regions the tree can create.

They are related, but not equivalent.

A shallow tree can still have multiple leaves:

```txt
depth 0:
    1 node, 1 leaf

depth 1:
    root split -> 2 leaves

depth 2:
    up to 4 leaves

depth 3:
    up to 8 leaves
```

But `max_leaf_nodes` controls the final number of prediction regions directly.

Example:

```txt
max_depth = 10
max_leaf_nodes = 3
```

This allows deep paths if needed, but only up to 3 final leaves.

---

### 2.3 Why `max_leaf_nodes` helps

A decision tree partitions feature space into regions.

Each leaf is one region.

More leaves means the tree can represent more complex rules.

Fewer leaves means a simpler model.

So `max_leaf_nodes` directly controls:

```txt
number of terminal decision regions
model complexity
overfitting risk
interpretability
```

A tree with too many leaves may memorize training samples.

A tree with too few leaves may underfit.

---

### 2.4 Local recursive growth vs best-first growth

There are two common ways to respect `max_leaf_nodes`.

#### Recursive depth-first growth

The current Phase 6 tree grows recursively:

```txt
build left subtree
build right subtree
```

This approach naturally respects `max_depth`, but `max_leaf_nodes` is harder because the tree does not globally know which split is best across all current leaves.

A simple implementation can track a leaf budget during recursion.

However, this may not always create the globally best tree under a leaf limit.

#### Best-first growth

A best-first tree grows the leaf that gives the largest impurity decrease at each step.

High-level algorithm:

```txt
start with one root leaf

while number_of_leaves < max_leaf_nodes:
    find the current leaf with the best valid split
    split that leaf
```

This is more aligned with `max_leaf_nodes`.

It is also more complex than simple recursion.

---

### 2.5 Recommended project approach

For ML Core, Phase 6B will use Option B: a best-first builder.

```txt
goal:
    implement max_leaf_nodes more faithfully
```

The current Phase 6 recursive builder is still useful and should remain available for the basic `DecisionTreeClassifier` workflow.

However, `max_leaf_nodes` is more naturally expressed as a best-first growth process:

```txt
1. start with one root leaf
2. evaluate the best split available for every current leaf
3. split the leaf with the largest impurity decrease
4. repeat until the leaf budget is exhausted or no valid split exists
```

This avoids pretending that a local recursive leaf budget is equivalent to a global leaf limit.

The implementation goal is to add best-first growth without breaking the existing recursive tree path. If the change would require too much modification to the current `tree_builder`, Phase 6B should introduce a new builder file instead, for example:

```txt
include/ml/trees/best_first_tree_builder.hpp
src/trees/best_first_tree_builder.cpp
```

The basic Phase 6 `DecisionTreeClassifier` should continue to work normally.

The advanced Phase 6B path should extend the tree system without making the original implementation harder to use.

---

### 2.6 Expected behavior

If `max_leaf_nodes` is not set:

```txt
tree grows according to existing stopping rules
```

If `max_leaf_nodes` is set:

```txt
tree must not exceed that many leaves
```

Invalid values:

```txt
max_leaf_nodes = 0:
    invalid

max_leaf_nodes = 1:
    valid, tree is a stump leaf with no splits

max_leaf_nodes >= 2:
    valid, tree can split until leaf budget is exhausted
```

---

## 3. `max_features`

### 3.1 Definition

`max_features` limits how many features are considered when searching for the best split at each node.

Without `max_features`, the tree evaluates all features:

```txt
for each feature:
    evaluate candidate thresholds
```

With `max_features`, each node only considers a subset of features.

Example:

```txt
num_features = 10
max_features = 3
```

At each split, the tree samples 3 features and only searches thresholds for those features.

---

### 3.2 Why `max_features` matters

For a single Decision Tree, `max_features` can act as regularization.

It prevents the tree from always choosing the globally strongest feature.

For Random Forest, `max_features` is essential.

Random Forest relies on making individual trees different from each other.

Two main randomness sources are:

```txt
bootstrap sampling of rows
feature subsampling at each split
```

Feature subsampling reduces correlation between trees.

If all trees always see all features, they may repeatedly choose the same strong splits and become too similar.

---

### 3.3 Common `max_features` choices

For classification, common choices include:

```txt
sqrt(num_features)
log2(num_features)
fixed integer
all features
```

For regression, a common choice is:

```txt
num_features / 3
```

For ML Core, we can keep the first implementation simple:

```txt
max_features = 0:
    invalid if used as a count

max_features unset:
    use all features

max_features = k:
    sample k features per split

max_features > num_features:
    invalid or clamp to num_features
```

Better design:

```txt
use std::optional<std::size_t> max_features
```

where:

```txt
nullopt:
    use all features

value:
    use that many randomly selected features
```

---

### 3.4 Determinism and random seed

Because `max_features` introduces randomness, the tree needs deterministic random control.

The tree options should include:

```txt
random_seed
shuffle / randomize
```

or an equivalent RNG strategy.

For tests, two trees trained with the same seed should produce the same predictions.

Different seeds may produce different trees.

---

### 3.5 Expected implementation behavior

At each node:

```txt
1. choose candidate feature subset
2. find best split only among those features
3. continue recursively
```

This means `find_best_split` should eventually support either:

```txt
all features
```

or:

```txt
specific candidate feature indices
```

Recommended extension:

```cpp
SplitCandidate find_best_split(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options,
    const std::vector<Eigen::Index>& feature_indices
);
```

or an internal helper equivalent.

---

## 4. `class_weight`

### 4.1 Definition

`class_weight` changes the importance of each class during training.

It is used mainly for imbalanced classification.

Example:

```txt
class 0: 950 samples
class 1: 50 samples
```

Without weighting, the tree may prefer splits that mostly improve performance on class 0 because class 0 dominates the dataset.

With class weights:

```txt
class 0 weight = 1.0
class 1 weight = 19.0
```

errors or impurities involving class 1 become more important.

---

### 4.2 Why class imbalance matters

Accuracy can be misleading on imbalanced datasets.

Example:

```txt
95% class 0
5% class 1
```

A model that always predicts class 0 gets:

```txt
accuracy = 95%
```

but it completely fails to detect class 1.

For trees, imbalance affects split selection because impurity is based on class proportions.

If minority classes are too rare, many splits may ignore them.

---

### 4.3 Weighted class proportions

Standard Gini uses:

```txt
p_k = count_k / total_count
```

Weighted Gini uses weighted counts:

```txt
weighted_count_k = sum of sample weights for samples in class k
p_k = weighted_count_k / total_weight
```

With class weights, each sample gets a weight based on its class:

```txt
sample_weight_i = class_weight[y_i]
```

Then impurity is computed using weighted proportions.

---

### 4.4 Example

Suppose a node contains:

```txt
class 0: 9 samples
class 1: 1 sample
```

Unweighted proportions:

```txt
p0 = 0.9
p1 = 0.1
```

If class 1 has weight 9:

```txt
weighted_count_0 = 9 * 1 = 9
weighted_count_1 = 1 * 9 = 9
```

Weighted proportions:

```txt
p0 = 0.5
p1 = 0.5
```

Now the node is treated as highly mixed, even though class 1 has fewer raw samples.

This encourages the tree to find splits that separate the minority class.

---

### 4.5 `balanced` class weights

For ML Core Phase 6B, the first implementation should support balanced class weights.

Balanced mode is useful because it gives the project a principled default for imbalanced classification without requiring the user to manually choose class weights.

The intended behavior is:

```txt
class_weight = balanced:
    class_weight_k = n_samples / (n_classes * count_k)
```

Manual class weights can be added later if needed.

Balanced class weights should be implemented through reusable helpers so they can be combined with `sample_weight` later.

---

### 4.6 Expected implementation behavior

Class weights should affect:

```txt
impurity computation
weighted child impurity
impurity reduction
leaf class probabilities if implemented
possibly leaf prediction tie-breaking
```

They should not change:

```txt
the raw input labels
the number of samples
the model API for prediction
```

For first implementation, the prediction can still be the weighted-majority class.

---

## 5. `sample_weight`

### 5.1 Definition

`sample_weight` assigns an individual importance weight to each training example.

Example:

```txt
sample 0 weight = 1.0
sample 1 weight = 0.5
sample 2 weight = 3.0
```

This is more general than `class_weight`.

`class_weight` produces weights based on labels.

`sample_weight` gives direct per-sample weights.

---

### 5.2 Relationship between `class_weight` and `sample_weight`

If both exist, they usually combine multiplicatively:

```txt
effective_weight_i = sample_weight_i * class_weight[y_i]
```

So each sample’s final importance depends on:

```txt
its individual sample weight
its class-level weight
```

---

### 5.3 Why sample weights are useful

Sample weights are used for:

```txt
imbalanced datasets
cost-sensitive learning
downweighting noisy examples
boosting algorithms
importance sampling
```

Gradient Boosting and AdaBoost-style methods often rely heavily on sample or residual weighting concepts.

Even if the first Gradient Boosting implementation does not use full sample weighting, adding this concept prepares the tree code for future extensions.

---

### 5.4 Weighted impurity with sample weights

For a node, compute total weight per class:

```txt
weighted_count_k = sum_i weight_i for samples where y_i = k
```

Total node weight:

```txt
total_weight = sum_k weighted_count_k
```

Then:

```txt
p_k = weighted_count_k / total_weight
```

Gini becomes:

```txt
Gini = 1 - sum_k(p_k^2)
```

Entropy becomes:

```txt
Entropy = -sum_k p_k log2(p_k)
```

The formulas are the same.

Only the definition of `p_k` changes.

---

### 5.5 Weighted child impurity

Unweighted child impurity uses sample counts:

```txt
(n_left / n_total) * impurity_left
+ (n_right / n_total) * impurity_right
```

Weighted child impurity should use total sample weights:

```txt
(weight_left / weight_total) * impurity_left
+ (weight_right / weight_total) * impurity_right
```

where:

```txt
weight_left = sum sample weights in left child
weight_right = sum sample weights in right child
```

This is essential if sample weights are meaningful.

---

### 5.6 Expected implementation behavior

If no `sample_weight` is provided:

```txt
all samples have weight 1.0
```

If `sample_weight` is provided:

```txt
sample_weight.size() must equal number of rows in X
all weights must be finite
all weights must be non-negative
total weight must be positive
```

Zero weights can be allowed.

A sample with zero weight is effectively ignored for impurity purposes, though it may still travel through the tree.

---

## 6. Missing-Value Handling

### 6.1 Why missing values matter

Real datasets often contain missing values.

In C++ numerical matrices, missing values are usually represented as:

```txt
NaN
```

A basic decision tree implementation often rejects non-finite values.

That is acceptable for a first tree, but advanced tree systems need an explicit strategy.

Missing-value handling must be deliberate.

Silent behavior is dangerous.

---

### 6.2 Common strategies

There are several ways to handle missing values.

#### Strategy A: reject missing values

This is the simplest and safest behavior.

```txt
if X contains NaN:
    throw invalid_argument
```

This is what a strict first implementation should do.

Advantages:

```txt
simple
safe
easy to test
no hidden assumptions
```

Disadvantages:

```txt
requires preprocessing before training
cannot directly handle missing data
```

---

#### Strategy B: impute missing values before training

Missing values can be filled before fitting the tree.

Common imputation strategies:

```txt
mean imputation
median imputation
mode imputation
constant value
```

This can be done outside the tree.

Advantages:

```txt
keeps tree logic simpler
consistent with preprocessing pipeline discipline
```

Disadvantages:

```txt
imputation choice affects model behavior
must avoid data leakage
```

Important rule:

```txt
fit imputation statistics only on training data
apply them to validation/test data
```

---

#### Strategy C: learn missing direction

Some tree algorithms decide whether missing values should go left or right for each split.

For each candidate split, evaluate options:

```txt
missing values go left
missing values go right
```

Choose the direction that gives the best impurity reduction.

At prediction time, missing values follow the learned missing direction.

This is more powerful but more complex.

The split must store:

```txt
missing_go_left = true/false
```

---

#### Strategy D: surrogate splits

A surrogate split is an alternative split used when the main split feature is missing.

Example:

```txt
main split:
    feature_3 <= 2.5

surrogate split:
    feature_1 <= 10.0
```

If `feature_3` is missing, use `feature_1`.

This is powerful but much more complex.

Not recommended for ML Core unless specifically needed.

---

### 6.3 Recommended project strategy

For Phase 6B:

```txt
start with explicit rejection of missing values
then optionally implement learned missing direction
```

The first implementation should make missing-value behavior explicit and safe:

```txt
training with NaN:
    reject with std::invalid_argument

prediction with NaN:
    reject with std::invalid_argument
```

This keeps the current tree behavior predictable while preparing the architecture for future missing-value support.

If time allows, the next step is learned missing direction.

If implemented, learned missing direction should affect:

```txt
split evaluation
best split selection
TreeNode storage
prediction traversal
tests
experiments
```

The basic Phase 6 tree should remain usable without missing-value support. Missing-value handling should be added as an explicit advanced behavior, not as a silent default.

---

### 6.4 Prediction with missing values

If the tree supports missing values, prediction must know what to do when a sample has `NaN` in the split feature.

With learned missing direction:

```txt
if sample[feature_index] is NaN:
    go to learned missing child
else if sample[feature_index] <= threshold:
    go left
else:
    go right
```

This requires each internal node to store:

```txt
missing_go_left
```

Without this stored behavior, missing-value handling at prediction is undefined.

---

## 7. Post-Pruning and Cost-Complexity Pruning

Phase 6 already implemented pre-pruning through stopping rules.

Pre-pruning stops the tree from growing too much during construction:

```txt
max_depth
min_samples_split
min_samples_leaf
min_impurity_decrease
max_leaf_nodes
```

Post-pruning is different.

Post-pruning first grows a larger tree, then removes branches that do not justify their complexity.

The goal is to improve generalization by simplifying the tree after observing its learned structure.

---

### 7.1 Why post-pruning exists

Decision trees are high-variance models.

A deep tree can create very specific rules that fit training samples extremely well but fail on new data.

Example:

```txt
if x_2 <= 3.41 and x_0 > 8.12 and x_5 <= 0.03:
    predict class 1
```

This rule may be correct for a small group of training samples but too specific to generalize.

Pre-pruning prevents some of this by limiting tree growth.

Post-pruning takes a different approach:

```txt
1. allow the tree to grow
2. inspect the learned subtrees
3. remove subtrees whose predictive benefit is too small
```

This can be more flexible than stopping early because the pruning decision is made after seeing the subtree.

---

### 7.2 Subtree pruning intuition

Consider an internal node with a full subtree below it.

There are two choices:

```txt
Option A:
    keep the subtree

Option B:
    replace the subtree with a single leaf
```

If the subtree only slightly improves training quality but adds many leaves, it may not be worth keeping.

Replacing the subtree with a leaf makes the model simpler.

The replacement leaf predicts the majority class of the samples reaching that node.

For classification, the replacement leaf should use the same prediction rule as normal leaves:

```txt
unweighted tree:
    majority class

weighted tree:
    weighted majority class
```

---

### 7.3 Tree risk

To decide whether to prune, we need a notion of tree risk.

For classification, a simple risk is training misclassification error at the leaves.

For a leaf:

```txt
leaf_error = number of samples in the leaf that are not in the majority class
```

For a subtree:

```txt
subtree_error = sum of leaf_error over all leaves in that subtree
```

If using sample weights:

```txt
leaf_error = total weight of samples in the leaf
             - weight of the majority class in the leaf
```

This lets pruning work with both unweighted and weighted trees.

---

### 7.4 Complexity penalty

A larger tree usually fits training data better.

So if we only minimize training error, the tree may stay too large.

Cost-complexity pruning adds a penalty for the number of leaves.

The objective is:

```txt
R_alpha(T) = R(T) + alpha * |T_leaves|
```

where:

```txt
T = tree or subtree
R(T) = empirical risk of the tree/subtree
|T_leaves| = number of leaf nodes
alpha = complexity penalty
```

Interpretation:

```txt
alpha = 0:
    no penalty for complexity
    larger trees are preferred if they reduce training error

larger alpha:
    stronger penalty for extra leaves
    smaller trees become more attractive
```

---

### 7.5 Comparing a subtree with a leaf

For each internal node, compare two scores.

Keep the subtree:

```txt
score_subtree =
    risk_subtree + alpha * number_of_leaves_subtree
```

Replace it with a leaf:

```txt
score_leaf =
    risk_leaf + alpha * 1
```

If:

```txt
score_leaf <= score_subtree
```

then the subtree can be pruned.

That means the simpler leaf is at least as good as the full subtree under the penalized objective.

---

### 7.6 Bottom-up pruning

A simple post-pruning algorithm works bottom-up.

High-level algorithm:

```txt
prune(node):

    if node is leaf:
        return leaf statistics

    prune(left child)
    prune(right child)

    compute score of keeping this subtree
    compute score of replacing this subtree with one leaf

    if leaf score <= subtree score:
        replace this internal node with a leaf

    return updated subtree statistics
```

Bottom-up pruning matters because child subtrees should be simplified before deciding whether the parent subtree is still worth keeping.

---

### 7.7 Effective alpha intuition

Full cost-complexity pruning usually computes a sequence of trees.

Each internal subtree has an effective alpha:

```txt
effective_alpha =
    (risk_leaf - risk_subtree) / (num_leaves_subtree - 1)
```

This value estimates how much complexity penalty is needed before replacing that subtree with a leaf becomes worthwhile.

Subtrees with small effective alpha are weak: they add complexity without much risk reduction.

A full pruning path repeatedly removes the weakest subtrees:

```txt
T0 = full tree
T1 = slightly pruned tree
T2 = more pruned tree
...
Tn = root-only tree
```

Then validation data can choose the best tree.

---

### 7.8 Simple pruning vs full pruning path

There are two possible implementation scopes.

#### Simple pruning with one `ccp_alpha`

This project can start with:

```txt
input:
    one fitted tree
    one ccp_alpha value

process:
    walk bottom-up
    prune any subtree where leaf_score <= subtree_score

output:
    one pruned tree
```

This is useful, testable, and much smaller than a full pruning-path implementation.

It requires:

```txt
leaf counting
subtree risk computation
node-to-leaf replacement
stored node prediction
stored node sample statistics or enough data to recompute risk
```

#### Full pruning path

A complete implementation would generate many candidate trees:

```txt
input:
    full tree

process:
    compute effective alpha for every subtree
    prune weakest subtree
    repeat until root-only tree

output:
    sequence of alpha values and corresponding trees
```

Then the best tree is selected using validation performance.

This is more complete but much larger.

It requires:

```txt
tree copying
alpha path generation
validation scoring
multiple tree states
more extensive experiments
```

---

### 7.9 Recommended Phase 6B scope

For Phase 6B, the recommended scope is:

```txt
implement if feasible:
    simple post-pruning with one ccp_alpha

defer:
    full cost-complexity pruning path
```

The reason is that Phase 6B’s main goal is still:

```txt
RandomForestClassifier
GradientBoosting model
advanced tree controls
```

A full pruning-path implementation could easily delay the ensemble work.

The simple version gives enough practical understanding without turning pruning into the whole phase.

---

### 7.10 Required data for pruning

To prune correctly, an internal node must know enough about the samples reaching it.

At minimum, pruning needs:

```txt
number of samples reaching the node
majority prediction at the node
risk if this node becomes a leaf
risk of the full subtree
number of leaves in the subtree
```

Current Phase 6 nodes already store:

```txt
prediction
impurity
num_samples
left child
right child
```

For better pruning, Phase 6B may need to store or compute:

```txt
class counts at the node
weighted class counts if sample weights are used
leaf risk
subtree risk
leaf count
```

If storing this directly in `TreeNode` makes the basic tree too messy, pruning should live in a separate module:

```txt
include/ml/trees/pruning.hpp
src/trees/pruning.cpp
```

This keeps the base tree clean.

---

### 7.11 Interaction with other Phase 6B features

Cost-complexity pruning interacts with several features.

#### With `max_leaf_nodes`

Both control tree size.

```txt
max_leaf_nodes:
    limits complexity during growth

cost-complexity pruning:
    reduces complexity after growth
```

They can coexist, but initial experiments should isolate them.

#### With `sample_weight`

Risk should use weighted errors when sample weights are provided.

This is important because class weighting and sample weighting change which mistakes are expensive.

#### With Random Forest

Random Forest usually controls overfitting through bagging and feature subsampling.

Pruning is less central there.

A first `RandomForestClassifier` does not need pruning.

#### With Gradient Boosting

Boosting usually uses shallow trees.

Complexity is often controlled with:

```txt
max_depth
max_leaf_nodes
learning_rate
number_of_estimators
```

Cost-complexity pruning is not required for a first boosting implementation.

---

### 7.12 Implementation decision

The Phase 6B decision is:

```txt
simple post-pruning:
    optional medium-priority feature

full pruning path:
    deferred unless the project explicitly needs it later
```

If simple pruning is implemented, it should be done after:

```txt
max_leaf_nodes
max_features
class_weight
sample_weight
```

and before or after Random Forest depending on implementation cost.

The implementation must not break the existing Phase 6 tree path.

The basic `DecisionTreeClassifier` should continue to work normally when pruning is disabled.

Recommended API idea:

```cpp
struct DecisionTreeOptions {
    // existing fields

    double ccp_alpha{0.0};
};
```

Expected behavior:

```txt
ccp_alpha = 0.0:
    no pruning

ccp_alpha > 0.0:
    apply simple post-pruning after the tree is grown
```

Validation:

```txt
ccp_alpha must be finite
ccp_alpha must be non-negative
```

This keeps pruning explicit, optional, and compatible with the current tree implementation.

---

## 8. Implementation Design Implications

The advanced controls should not be bolted on randomly.

They imply changes to the tree architecture.

The design rule for Phase 6B is:

```txt
extend the current tree system without breaking the Phase 6 DecisionTreeClassifier
```

When an advanced feature only requires a small extension, update the existing functions with optional parameters.

Example:

```txt
weighted Gini:
    add optional sample weights
    keep existing unweighted behavior working
```

When an advanced feature would require major changes to the current recursive builder, create a separate implementation path.

Example:

```txt
max_leaf_nodes best-first growth:
    prefer a new best-first builder file if the recursive builder would become messy
```

The basic `DecisionTreeClassifier` must remain usable exactly as before.

Phase 6B should add advanced capabilities without turning the Phase 6 implementation into a dependency problem.

---

### 8.1 Tree options

`DecisionTreeOptions` may need fields such as:

```cpp
std::optional<std::size_t> max_leaf_nodes;
std::optional<std::size_t> max_features;

unsigned int random_seed{42};
bool randomize_features{false};

std::map<int, double> class_weight;
bool use_balanced_class_weight{false};

MissingValueStrategy missing_value_strategy{MissingValueStrategy::Reject};
double ccp_alpha{0.0};
```

The exact design can evolve, but the options should remain explicit.

---

### 8.2 Split scoring

Split scoring should preserve the existing unweighted API and add weighted behavior carefully.

The preferred style is either optional parameters or separate weighted helpers, depending on which keeps the code clearer.

Split scoring may need weighted versions:

```cpp
double weighted_gini_impurity(
    const Vector& y,
    const Vector& sample_weight
);

double weighted_entropy(
    const Vector& y,
    const Vector& sample_weight
);
```

And child impurity should use total child weights, not only child counts.

The existing unweighted calls should continue to behave exactly as they did in Phase 6.

For example:

```cpp
gini_impurity(y)
```

should remain valid.

Weighted behavior can be exposed as:

```cpp
gini_impurity(y, sample_weight)
```

or:

```cpp
weighted_gini_impurity(y, sample_weight)
```

The implementation should choose the version that keeps tests and call sites clearest.

---

### 8.3 Split candidates

`SplitCandidate` may need additional fields:

```cpp
double left_weight;
double right_weight;
bool missing_go_left;
std::vector<Eigen::Index> feature_indices_considered;
```

Only add fields when implementation requires them.

---

### 8.4 Tree nodes

`TreeNode` may need to store:

```cpp
bool missing_go_left;
double weighted_num_samples;
std::size_t leaf_count;
```

For pruning, nodes may also need:

```cpp
double subtree_risk;
double leaf_risk;
```

But avoid adding too much too early.

---

### 8.5 Ensemble reuse

Random Forest should reuse:

```txt
DecisionTreeClassifier
DecisionTreeOptions
max_features
bootstrap sampling
majority voting
```

Gradient Boosting may reuse:

```txt
tree-building logic
shallow trees
regression-style trees
residual targets
learning-rate-controlled additive predictions
```

This may require implementing a regression tree later.

For that reason, Gradient Boosting may start with:

```txt
GradientBoostingRegressor
```

before classification.

---

## 9. Recommended Implementation Order

The recommended Phase 6B order is:

```txt
1. Write this theory document.
2. Add advanced tree option fields carefully.
3. Add tests for option validation.
4. Add explicit missing-value rejection for train/predict paths.
5. Add balanced class-weight helpers.
6. Add optional weighted impurity / weighted split-scoring support without breaking unweighted calls.
7. Implement max_features, because Random Forest needs it.
8. Implement best-first builder support for max_leaf_nodes.
9. Implement bootstrap sampling utilities.
10. Implement RandomForestClassifier.
11. Add Random Forest experiments.
12. Decide whether simple cost-complexity post-pruning fits in this phase.
13. Implement simple pruning only if it does not block ensembles.
14. Implement Gradient Boosting only after deciding whether a regression tree is needed.
```

The exact order may change, but the key principle is:

```txt
do not break the clean Phase 6 tree while adding advanced features
```

---

## 10. Phase 6B Scope Decision

Phase 6B is intentionally ambitious.

It contains two types of work:

```txt
core ensemble work:
    Random Forest
    Gradient Boosting

advanced tree infrastructure:
    max_leaf_nodes
    max_features
    class_weight
    sample_weight
    missing-value handling
    cost-complexity pruning
```

The minimum successful Phase 6B should include:

```txt
RandomForestClassifier
at least one Gradient Boosting model
max_features support
deterministic ensemble behavior
ensemble comparison experiments
```

The advanced features can be prioritized as follows:

```txt
High priority:
    max_features
    bootstrap sampling
    deterministic random seed support
    balanced class_weight
    explicit missing-value rejection

Medium priority:
    max_leaf_nodes with best-first growth
    sample_weight
    simple cost-complexity post-pruning

Lower priority / optional:
    learned missing-value direction
    full cost-complexity pruning path
```

This means some advanced features may be honestly deferred if they would overload the phase.

However, every feature must be either:

```txt
implemented and tested
```

or:

```txt
explicitly documented as deferred with a reason
```

The main rule is:

```txt
Random Forest and Gradient Boosting must be implemented as reusable project models,
not left as theory-only notes.
```