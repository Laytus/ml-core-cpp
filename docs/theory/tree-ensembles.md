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

## 9. Random Forest Implementation Details

Random Forest is an ensemble method built from many decision trees.

A single decision tree is powerful but unstable. Small changes in the training data can produce a very different tree.

Random Forest reduces this instability by training many different trees and aggregating their predictions.

For classification, the final prediction is usually the majority vote across trees.

```txt
tree_1 predicts class 0
tree_2 predicts class 1
tree_3 predicts class 1

Random Forest prediction:
    class 1
```

The main idea is:

```txt
many high-variance trees
+ diversity between trees
+ voting / averaging
= lower-variance ensemble
```

---

### 9.1 Why Random Forest works

A fully grown decision tree has low bias but high variance.

This means:

```txt
low bias:
    it can fit complex patterns

high variance:
    it changes a lot when the training data changes
```

Random Forest reduces variance by averaging or voting across many trees.

The ensemble works best when individual trees are:

```txt
strong enough to learn useful patterns
different enough from each other
```

If every tree is identical, the forest gives no benefit.

The two main mechanisms used to make trees different are:

```txt
bootstrap sampling
feature subsampling
```

---

### 9.2 Bagging

Random Forest is based on bagging.

Bagging means:

```txt
bootstrap aggregating
```

The process is:

```txt
1. create many bootstrap samples from the training data
2. train one tree on each bootstrap sample
3. aggregate predictions from all trees
```

Each tree sees a slightly different dataset.

This makes the trees different and reduces variance when their predictions are combined.

---

### 9.3 Bootstrap samples

A bootstrap sample is created by sampling training rows **with replacement**.

If the original dataset has `n` samples, each bootstrap sample usually also has `n` rows.

Example:

```txt
original rows:
    [0, 1, 2, 3, 4]

bootstrap sample:
    [0, 2, 2, 4, 1]
```

Some rows may appear multiple times.

Some rows may not appear at all.

This is intentional.

The rows that do not appear in a given tree’s bootstrap sample are called **out-of-bag** rows for that tree.

---

### 9.4 Bootstrap sample properties

Because sampling is done with replacement, each tree receives a noisy version of the original dataset.

For large datasets, each bootstrap sample contains roughly:

```txt
about 63.2% unique original samples
```

and leaves roughly:

```txt
about 36.8% out-of-bag samples
```

This is an approximation, but it explains why out-of-bag evaluation is possible.

The exact percentages are not required for implementation.

What matters is:

```txt
bootstrap sample:
    rows used to train a tree

out-of-bag rows:
    rows not used to train that tree
```

---

### 9.5 Feature subsampling

Random Forest also uses feature subsampling.

At each split, instead of evaluating all features, each tree considers only a random subset of features.

This is controlled by:

```txt
max_features
```

Example:

```txt
num_features = 10
max_features = 3
```

At each node, the tree randomly chooses 3 features and searches for the best split only among those 3.

This prevents all trees from repeatedly choosing the same strongest feature.

Feature subsampling reduces correlation between trees.

Lower correlation makes voting more useful.

---

### 9.6 Why bootstrap sampling alone is not enough

If the dataset has one very strong feature, many bootstrapped trees may still choose the same first split.

This can make the trees too similar.

Feature subsampling solves this by sometimes hiding the strongest feature from a split.

Then different trees discover different useful patterns.

The forest becomes more diverse.

This is why Random Forest uses both:

```txt
row randomness:
    bootstrap samples

feature randomness:
    max_features at split time
```

---

### 9.7 Ensemble voting

For classification, each tree outputs a predicted class.

The forest combines these predictions through majority voting.

Example:

```txt
tree predictions:
    [0, 1, 1, 2, 1]

vote counts:
    class 0 -> 1
    class 1 -> 3
    class 2 -> 1

forest prediction:
    class 1
```

If there is a tie, the implementation should use a deterministic rule.

Recommended tie-breaking:

```txt
choose the smallest class index among tied classes
```

This matches the deterministic behavior already used for tree leaf majority class.

---

### 9.8 Prediction probabilities

A Random Forest should also estimate class probabilities.

For each sample:

```txt
probability(class k) =
    number of trees predicting class k / number of trees
```

Example:

```txt
tree predictions:
    [0, 1, 1, 1]

probability(class 0) = 1 / 4 = 0.25
probability(class 1) = 3 / 4 = 0.75
```

In ML Core, the first `RandomForestClassifier` implementation should include both:

```cpp
Vector predict(const Matrix& X) const;

Matrix predict_proba(const Matrix& X) const;
```

`predict` returns the majority-vote class.

`predict_proba` returns one row per sample and one column per class.

Each row should sum to 1.0 because the probabilities are computed from the fraction of trees voting for each class.

The predicted class should match the class with the highest predicted probability.

Tie-breaking should remain deterministic. If two classes have the same probability, the smallest class index should win.

---

### 9.9 Out-of-bag intuition

Out-of-bag evaluation uses the rows that were not selected in a tree’s bootstrap sample.

For one tree:

```txt
bootstrap sample:
    rows used to train this tree

out-of-bag rows:
    rows not used to train this tree
```

A row can be evaluated by the subset of trees where that row was out-of-bag.

High-level process:

```txt
for each training row i:
    collect predictions from trees where row i was out-of-bag
    aggregate those predictions
    compare against y_i
```

This gives an internal validation-like estimate without creating a separate validation set.

---

### 9.10 Why out-of-bag evaluation is useful

Out-of-bag evaluation is useful because it gives an estimate of generalization performance while still allowing each tree to train on a full-size bootstrap sample.

It can be used to compare settings such as:

```txt
number of trees
max_features
max_depth
max_leaf_nodes
bootstrap enabled/disabled
```

However, it adds implementation complexity.

The first Random Forest implementation can defer out-of-bag scoring while still storing enough bootstrap information to add it later.

---

### 9.11 Random Forest options

The Random Forest model should have its own options struct.

Recommended first version:

```cpp
struct RandomForestOptions {
    std::size_t n_estimators{100};
    bool bootstrap{true};
    std::optional<std::size_t> max_features{std::nullopt};
    unsigned int random_seed{42};

    DecisionTreeOptions tree_options{};
};
```

Interpretation:

```txt
n_estimators:
    number of trees in the forest

bootstrap:
    whether each tree trains on a bootstrap sample

max_features:
    number of features considered at each split

random_seed:
    controls reproducibility

tree_options:
    controls each individual DecisionTreeClassifier
```

Important design rule:

```txt
RandomForestOptions controls ensemble behavior.
DecisionTreeOptions controls each tree.
```

This keeps ensemble orchestration separate from base tree logic.

---

### 9.12 How Random Forest should reuse the existing tree

Random Forest should not copy decision tree code.

It should reuse:

```txt
DecisionTreeClassifier
DecisionTreeOptions
max_features
balanced class weights if enabled
sample-weighted split scoring if needed later
```

The forest training loop should look conceptually like:

```txt
for tree_index in 0 .. n_estimators - 1:
    create bootstrap sample
    configure tree options
    set per-tree random seed
    fit DecisionTreeClassifier on bootstrap sample
    store tree
```

Prediction should look like:

```txt
for each sample:
    ask every tree for its prediction
    majority vote
```

This makes Random Forest an orchestration layer over the base tree implementation.

---

### 9.13 Per-tree random seeds

Each tree should receive a deterministic but different seed.

Example:

```txt
forest random_seed = 42

tree 0 seed = 42
tree 1 seed = 43
tree 2 seed = 44
```

or another deterministic seed sequence.

This matters because each tree may use randomness for:

```txt
bootstrap sampling
feature subsampling
```

Requirements:

```txt
same RandomForestOptions -> same fitted forest behavior
different random_seed -> potentially different forest behavior
```

---

### 9.14 Bootstrap utilities

Before implementing `RandomForestClassifier`, Phase 6B should implement bootstrap utilities.

Recommended helper outputs:

```txt
bootstrap sample X
bootstrap sample y
selected row indices
out-of-bag row indices
```

A useful struct:

```cpp
struct BootstrapSample {
    Matrix X;
    Vector y;
    std::vector<Eigen::Index> sampled_indices;
    std::vector<Eigen::Index> out_of_bag_indices;
};
```

For weighted training later, the struct may also include:

```cpp
Vector sample_weight;
```

But this should only be added when needed.

---

### 9.15 Bootstrap with replacement

Implementation behavior:

```txt
input:
    X, y, random_seed

process:
    sample n row indices from [0, n - 1] with replacement

output:
    X_bootstrap
    y_bootstrap
    sampled_indices
    out_of_bag_indices
```

Validation:

```txt
X must be non-empty
y must be non-empty
X.rows() must equal y.size()
```

Expected properties:

```txt
X_bootstrap.rows() == X.rows()
y_bootstrap.size() == y.size()
sampled_indices.size() == X.rows()
out_of_bag_indices.size() can vary
```

---

### 9.16 Bootstrap disabled

Some Random Forest variants allow training each tree on the full dataset while still using feature subsampling.

If:

```txt
bootstrap = false
```

then each tree receives the full training dataset.

This is less random, but it can still produce different trees if `max_features` is used and each tree has a different seed.

For the first implementation:

```txt
bootstrap = true:
    use bootstrap samples

bootstrap = false:
    train every tree on full X/y
```

---

### 9.17 Minimum viable RandomForestClassifier

The minimum useful model should expose:

```cpp
class RandomForestClassifier {
public:
    RandomForestClassifier();

    explicit RandomForestClassifier(RandomForestOptions options);

    void fit(const Matrix& X, const Vector& y);

    Vector predict(const Matrix& X) const;

    Matrix predict_proba(const Matrix& X) const;

    bool is_fitted() const;

    const RandomForestOptions& options() const;

    std::size_t num_classes() const;

    std::size_t num_trees() const;
};
```

Required behavior:

```txt
reject invalid options
reject empty or mismatched training data
reject predict before fit
reject predict_proba before fit
fit n_estimators decision trees
support deterministic random_seed
support bootstrap on/off
support max_features through tree options
predict by majority vote
predict_proba by vote fractions
ensure predict_proba rows sum to 1
use deterministic tie-breaking
```

The forest should infer the number of classes from the training targets.

For the first implementation, labels should be non-negative integer-valued classes:

```txt
0, 1, 2, ...
```

This keeps `predict_proba` simple because each class maps directly to a probability column.

---

### 9.18 First implementation boundaries

The first Random Forest implementation should include:

```txt
implemented:
    bootstrap sampling
    feature subsampling through max_features
    deterministic seed support
    majority-vote prediction with predict
    probability prediction with predict_proba
    basic validation
    experiments

optional / later:
    out-of-bag score
    sample_weight support
    class_weight-specific experiments
    regression forest
```

This keeps the implementation focused and compatible with the current Phase 6B goals while still providing the two most important prediction APIs:

```txt
predict:
    final class prediction

predict_proba:
    vote-based class probability estimates
```

Out-of-bag scoring remains optional because it requires storing and aggregating per-tree out-of-bag predictions.

---

### 9.19 Random Forest experiments

Useful experiments:

```txt
single tree vs random forest
number of trees
max_features comparison
bootstrap vs no bootstrap
```

Expected files:

```txt
outputs/phase-6b-tree-ensembles/random_forest_comparison.csv
outputs/phase-6b-tree-ensembles/random_forest_comparison.txt
```

Metrics:

```txt
accuracy
number of trees
bootstrap enabled
max_features
random_seed
```

For now, accuracy is enough because the project already has classification metrics.

Later, macro F1 can be added for imbalanced datasets.

---

### 9.20 Random Forest completion criteria

This Random Forest sub-phase is complete when:

```txt
bootstrap utilities are implemented and tested
RandomForestClassifier fits multiple trees
RandomForestClassifier predicts by majority vote
same seed gives reproducible predictions
invalid options are rejected
experiments compare single tree and forest behavior
```

At that point, Phase 6B will have its first complete ensemble model.

---

## 10. Gradient Boosting Implementation Details

Gradient Boosting is an ensemble method that builds a strong model by adding many weak models sequentially.

Unlike Random Forest, where trees are trained mostly independently, Gradient Boosting trains trees one after another.

Each new tree tries to correct the errors made by the current ensemble.

The high-level idea is:

```txt
start with a simple initial prediction

repeat:
    measure current prediction errors
    fit a small tree to correct those errors
    add the new tree to the ensemble
```

So the final model is an additive model:

```txt
final_prediction =
    initial_prediction
    + contribution_from_tree_1
    + contribution_from_tree_2
    + ...
    + contribution_from_tree_M
```

Gradient Boosting is powerful because it turns many simple weak learners into a strong predictive model.

---

### 10.1 Boosting vs bagging

Random Forest is based on bagging.

Bagging trains many models independently and combines their predictions.

```txt
Random Forest:
    train trees independently
    aggregate by voting or averaging
```

Gradient Boosting is different.

Boosting trains models sequentially.

```txt
Gradient Boosting:
    train tree 1
    compute errors
    train tree 2 to fix errors from tree 1
    compute new errors
    train tree 3 to fix remaining errors
    ...
```

The main contrast is:

```txt
Random Forest:
    reduce variance through averaging independent trees

Gradient Boosting:
    reduce bias by sequentially improving the model
```

In practice, Gradient Boosting can also control variance through shallow trees, learning rate, and number of estimators.

---

### 10.2 Additive models

Gradient Boosting builds an additive model.

For regression, the model can be written conceptually as:

```txt
F_M(x) = F_0(x) + eta * h_1(x) + eta * h_2(x) + ... + eta * h_M(x)
```

where:

```txt
F_M(x) = final model after M trees
F_0(x) = initial prediction
h_m(x) = tree added at step m
eta = learning rate
M = number of estimators
```

Each tree contributes a small correction.

The model is not replaced at each step.

It is updated by adding another tree.

---

### 10.3 Initial prediction

For regression with squared error, a common initial prediction is the mean target value:

```txt
F_0(x) = mean(y)
```

This is the best constant prediction under mean squared error.

Example:

```txt
y = [2, 4, 6]

F_0 = 4
```

Before adding any trees, every sample receives prediction `4`.

Then the first tree learns how to correct that constant baseline.

---

### 10.4 Residual fitting for squared-error regression

For squared-error regression, the correction target is the residual:

```txt
residual_i = y_i - F(x_i)
```

where:

```txt
y_i = true target
F(x_i) = current model prediction
```

The next tree is trained to predict these residuals.

Example:

```txt
true y:
    [2, 4, 6]

current prediction:
    [4, 4, 4]

residuals:
    [-2, 0, 2]
```

The next tree tries to map each `x_i` to its residual.

Then the model is updated:

```txt
F_new(x) = F_old(x) + eta * tree_prediction(x)
```

---

### 10.5 Gradient fitting

The name “Gradient Boosting” comes from the more general idea of fitting the negative gradient of a loss function.

For squared error:

```txt
loss = 1/2 * (y - F(x))^2
```

The negative gradient with respect to the prediction is:

```txt
y - F(x)
```

which is exactly the residual.

So for squared-error regression:

```txt
negative gradient = residual
```

This is why the first implementation can focus on residual fitting.

It gives a correct and intuitive version of Gradient Boosting without needing to implement the full general loss-function framework immediately.

---

### 10.6 Why start with `GradientBoostingRegressor`

For this project, the recommended first target is:

```txt
GradientBoostingRegressor
```

not `GradientBoostingClassifier`.

Reason:

```txt
regression boosting with squared error is simpler
```

It only requires:

```txt
initial prediction = mean(y)
residuals = y - prediction
fit shallow regression tree to residuals
update prediction additively
```

Classification boosting is more complex because it usually needs:

```txt
log odds
probabilities
cross-entropy loss
class-specific gradients
multi-class handling
```

Therefore, the cleanest Phase 6B path is:

```txt
first:
    implement GradientBoostingRegressor

later:
    implement GradientBoostingClassifier
```

---

### 10.7 Need for regression trees

The current Phase 6 tree is a classifier.

It predicts class labels using majority vote in leaves.

Gradient Boosting for squared-error regression needs trees that predict continuous values.

That means we need a regression-tree weak learner.

A regression tree differs from a classification tree in two main ways:

```txt
split criterion:
    classification tree uses Gini or entropy
    regression tree uses variance / squared-error reduction

leaf prediction:
    classification tree predicts majority class
    regression tree predicts mean target value
```

For a first implementation, a simple regression tree should be enough.

It does not need all advanced classifier features immediately.

---

### 10.8 Regression tree split criterion

For regression, a split is good if it reduces squared error.

At a node, the best constant prediction is the mean target value:

```txt
prediction = mean(y_node)
```

The node error can be measured as sum of squared errors:

```txt
SSE = sum_i (y_i - mean(y_node))^2
```

A candidate split creates left and right children.

The split quality can be measured by error reduction:

```txt
reduction =
    parent_sse - (left_sse + right_sse)
```

A good split has large positive reduction.

This is analogous to impurity reduction in classification trees.

---

### 10.9 Shallow weak learners

Gradient Boosting usually uses shallow trees.

These trees are weak learners.

Typical settings:

```txt
max_depth = 1
max_depth = 2
max_depth = 3
```

A depth-1 tree is called a decision stump.

Using shallow trees is important because each tree should only make a small correction.

If each tree is too strong, the boosting model can overfit quickly.

So Gradient Boosting controls complexity through:

```txt
tree depth
number of estimators
learning rate
minimum samples per split / leaf
```

---

### 10.10 Learning rate / shrinkage

The learning rate controls how much each new tree contributes.

The update is:

```txt
F_new(x) = F_old(x) + learning_rate * h_m(x)
```

where:

```txt
h_m(x) = prediction from the new tree
```

If:

```txt
learning_rate = 1.0
```

the full tree correction is added.

If:

```txt
learning_rate = 0.1
```

only 10% of the tree correction is added.

Smaller learning rates usually require more trees but can generalize better.

This is called shrinkage.

---

### 10.11 Number of estimators

`n_estimators` is the number of boosting rounds.

Each round adds one tree.

Low `n_estimators`:

```txt
may underfit
training is faster
```

High `n_estimators`:

```txt
can fit more complex patterns
may overfit if learning_rate is too large
training is slower
```

There is a trade-off between:

```txt
learning_rate
n_estimators
tree depth
```

Common pattern:

```txt
smaller learning_rate
+ more estimators
= smoother improvement
```

---

### 10.12 Staged predictions

Staged predictions are predictions after each boosting round.

Example:

```txt
stage 0:
    initial mean prediction

stage 1:
    mean + tree_1 correction

stage 2:
    mean + tree_1 correction + tree_2 correction

stage 3:
    mean + tree_1 correction + tree_2 correction + tree_3 correction
```

Staged predictions are useful for understanding training behavior.

They allow us to track:

```txt
training loss over boosting rounds
validation loss over boosting rounds
overfitting behavior
```

In the implementation, staged predictions can be stored as training history.

A simple first version can store:

```txt
training_loss_history
```

where each entry is the mean squared error after adding a tree.

---

### 10.13 Gradient Boosting options

The model should have its own options struct.

Recommended first version:

```cpp
struct GradientBoostingRegressorOptions {
    std::size_t n_estimators{100};
    double learning_rate{0.1};
    std::size_t max_depth{2};
    std::size_t min_samples_split{2};
    std::size_t min_samples_leaf{1};
    unsigned int random_seed{42};
};
```

Validation:

```txt
n_estimators >= 1
learning_rate must be finite and > 0
max_depth >= 1
min_samples_split >= 2
min_samples_leaf >= 1
```

The first implementation does not need randomness unless feature subsampling is added later.

Still, keeping `random_seed` in the options makes the API future-ready.

---

### 10.14 Minimum viable `GradientBoostingRegressor`

The minimum useful model should expose:

```cpp
class GradientBoostingRegressor {
public:
    GradientBoostingRegressor();

    explicit GradientBoostingRegressor(
        GradientBoostingRegressorOptions options
    );

    void fit(const Matrix& X, const Vector& y);

    Vector predict(const Matrix& X) const;

    bool is_fitted() const;

    const GradientBoostingRegressorOptions& options() const;

    const std::vector<double>& training_loss_history() const;
};
```

Required behavior:

```txt
reject invalid options
reject empty or mismatched training data
reject predict before fit
start from mean target prediction
train n_estimators shallow regression trees
update residuals stage by stage
store training loss history
predict by summing all tree contributions
```

---

### 10.15 Required helper: regression tree

Because the current `DecisionTreeClassifier` cannot predict continuous residuals, we need a regression-tree helper.

Recommended minimal API:

```cpp
class DecisionTreeRegressor {
public:
    DecisionTreeRegressor();

    explicit DecisionTreeRegressor(DecisionTreeRegressorOptions options);

    void fit(const Matrix& X, const Vector& y);

    Vector predict(const Matrix& X) const;

    bool is_fitted() const;
};
```

This can reuse ideas from the classifier tree but should be implemented separately if that keeps the code cleaner.

Reason:

```txt
classification tree:
    Gini / entropy
    majority class leaves

regression tree:
    squared-error reduction
    mean-value leaves
```

Trying to force both into the same builder too early may make the tree code harder to understand.

---

### 10.16 Regression tree options

Recommended first version:

```cpp
struct DecisionTreeRegressorOptions {
    std::size_t max_depth{2};
    std::size_t min_samples_split{2};
    std::size_t min_samples_leaf{1};
    double min_error_decrease{0.0};
};
```

This mirrors the classifier tree stopping rules but uses regression terminology.

Validation:

```txt
max_depth >= 1
min_samples_split >= 2
min_samples_leaf >= 1
min_error_decrease must be finite and >= 0
```

---

### 10.17 Regression tree prediction

Each regression-tree leaf predicts the mean target value of samples reaching that leaf.

Example:

```txt
leaf targets:
    [1.5, 2.0, 2.5]

leaf prediction:
    2.0
```

At prediction time, a sample follows the tree splits until it reaches a leaf.

The leaf value is returned as the tree prediction.

This prediction is then used as the boosting correction.

---

### 10.18 Gradient Boosting training loop

High-level training loop for squared-error Gradient Boosting:

```txt
initial_prediction = mean(y)

current_predictions = vector filled with initial_prediction

for m in 1..n_estimators:
    residuals = y - current_predictions

    fit regression_tree_m on X, residuals

    correction = regression_tree_m.predict(X)

    current_predictions =
        current_predictions + learning_rate * correction

    record training MSE
```

After fitting, prediction on new data is:

```txt
prediction = initial_prediction

for each tree:
    prediction += learning_rate * tree.predict(x)
```

---

### 10.19 Loss history

Training loss should be recorded after each boosting round.

For squared-error regression, use mean squared error:

```txt
MSE = mean((y - prediction)^2)
```

Expected behavior:

```txt
training_loss_history.size() == n_estimators
```

In simple datasets, loss should generally decrease.

It may not strictly decrease in every possible dataset or with every option setting, but tests can use a controlled dataset where it does.

---

### 10.20 First implementation boundaries

The first Gradient Boosting implementation should include:

```txt
implemented:
    DecisionTreeRegressor
    GradientBoostingRegressor
    squared-error loss
    residual fitting
    learning_rate
    shallow weak learners
    training loss history
    validation tests
    experiments

deferred:
    GradientBoostingClassifier
    logistic / cross-entropy boosting
    multi-class boosting
    stochastic subsampling
    feature subsampling
    sample weights
    early stopping
```

This gives a serious and coherent first implementation without overloading the phase.

---

### 10.21 Gradient Boosting experiments

Useful experiments:

```txt
number of estimators
learning rate
tree depth
single tree vs boosted trees
training loss history
```

Expected files:

```txt
outputs/phase-6b-tree-ensembles/gradient_boosting_comparison.csv
outputs/phase-6b-tree-ensembles/gradient_boosting_comparison.txt
outputs/phase-6b-tree-ensembles/gradient_boosting_loss_history.csv
```

Metrics:

```txt
MSE
RMSE
number of estimators
learning rate
max_depth
initial prediction
final training loss
```

This will connect naturally with the regression metrics already implemented earlier in the project.

---

### 10.22 Gradient Boosting completion criteria

This Gradient Boosting sub-phase is complete when:

```txt
DecisionTreeRegressor is implemented and tested
GradientBoostingRegressor fits residuals sequentially
learning_rate affects updates
training loss history is stored
predict works on new data
experiments compare estimator count, learning rate, and tree depth
```

At that point, Phase 6B has both major ensemble families:

```txt
Random Forest:
    bagging + feature randomness + voting

Gradient Boosting:
    sequential residual correction + shrinkage
```

---

## Recommended Implementation Order

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

## Phase 6B Scope Decision

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