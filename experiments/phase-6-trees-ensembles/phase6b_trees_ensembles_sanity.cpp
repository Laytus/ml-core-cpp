#include "phase6b_trees_ensembles_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/types.hpp"
#include "ml/trees/best_first_tree_builder.hpp"
#include "ml/trees/decision_tree.hpp"
#include "ml/trees/tree_node.hpp"
#include "ml/trees/class_weights.hpp"
#include "ml/trees/split_scoring.hpp"
#include "ml/common/classification_metrics.hpp"
#include "ml/trees/bootstrap.hpp"
#include "ml/trees/random_forest.hpp"
#include "ml/trees/regression_tree.hpp"
#include "ml/trees/gradient_boosting.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <limits>

namespace test = ml::experiments::test;

namespace {

ml::Vector make_vector(std::initializer_list<double> values) {
    ml::Vector result(static_cast<Eigen::Index>(values.size()));

    Eigen::Index index = 0;
    for (double value : values) {
        result(index) = value;
        ++index;
    }

    return result;
}

ml::Matrix make_max_leaf_tree_X() {
    ml::Matrix X(8, 2);
    X << 0.0, 0.0,
         1.0, 0.0,
         2.0, 0.0,
         3.0, 0.0,
         4.0, 0.0,
         5.0, 0.0,
         6.0, 0.0,
         7.0, 0.0;

    return X;
}

ml::Vector make_max_leaf_tree_y() {
    return make_vector({
        0.0, 0.0,
        1.0, 1.0,
        0.0, 0.0,
        1.0, 1.0
    });
}

ml::Matrix make_simple_tree_X() {
    ml::Matrix X(6, 2);
    X << 0.0, 0.0,
         1.0, 0.0,
         2.0, 0.0,
         3.0, 1.0,
         4.0, 1.0,
         5.0, 1.0;

    return X;
}

ml::Vector make_simple_tree_y() {
    return make_vector({0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
}

std::size_t count_leaves(const ml::TreeNode& node) {
    if (node.is_leaf) {
        return 1;
    }

    if (!node.left || !node.right) {
        throw std::runtime_error("count_leaves: internal node is missing children");
    }

    return count_leaves(*node.left) + count_leaves(*node.right);
}

std::size_t count_internal_nodes(const ml::TreeNode& node) {
    if (node.is_leaf) {
        return 0;
    }

    if (!node.left || !node.right) {
        throw std::runtime_error("count_internal_nodes: internal node is missing children");
    }

    return 1 + count_internal_nodes(*node.left) + count_internal_nodes(*node.right);
}

ml::Matrix make_sample_weight_split_X() {
    ml::Matrix X(4, 1);
    X << 0.0,
         1.0,
         2.0,
         3.0;

    return X;
}

ml::Vector make_sample_weight_split_y() {
    return make_vector({
        0.0,
        0.0,
        1.0,
        1.0
    });
}

ml::Vector make_balanced_sample_weight_vector() {
    return make_vector({
        1.0,
        1.0,
        1.0,
        1.0
    });
}

ml::Vector make_unbalanced_sample_weight_vector() {
    return make_vector({
        1.0,
        1.0,
        3.0,
        3.0
    });
}

// -----------------------------------------------------------------------------
// max_leaf_nodes / best-first builder tests
// -----------------------------------------------------------------------------

void test_best_first_builder_rejects_missing_max_leaf_nodes() {
    const ml::Matrix X = make_max_leaf_tree_X();
    const ml::Vector y = make_max_leaf_tree_y();

    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = std::nullopt;

    static_cast<void>(ml::build_tree_best_first(
        X,
        y,
        options
    ));
}

void test_best_first_builder_with_one_leaf_creates_root_leaf() {
    const ml::Matrix X = make_max_leaf_tree_X();
    const ml::Vector y = make_max_leaf_tree_y();

    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = 1;
    options.max_depth = 10;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree_best_first(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected max_leaf_nodes = 1 to create root leaf");
    }

    if (count_leaves(*root) != 1) {
        throw std::runtime_error("expected exactly one leaf");
    }

    test::assert_almost_equal(
        root->prediction,
        0.0,
        "test_best_first_builder_with_one_leaf_creates_root_leaf prediction"
    );
}

void test_best_first_builder_respects_two_leaf_limit() {
    const ml::Matrix X = make_max_leaf_tree_X();
    const ml::Vector y = make_max_leaf_tree_y();

    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = 2;
    options.max_depth = 10;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree_best_first(
        X,
        y,
        options
    );

    if (count_leaves(*root) != 2) {
        throw std::runtime_error("expected exactly two leaves");
    }

    if (count_internal_nodes(*root) != 1) {
        throw std::runtime_error("expected exactly one internal node");
    }
}

void test_best_first_builder_respects_three_leaf_limit() {
    const ml::Matrix X = make_max_leaf_tree_X();
    const ml::Vector y = make_max_leaf_tree_y();

    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = 3;
    options.max_depth = 10;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree_best_first(
        X,
        y,
        options
    );

    if (count_leaves(*root) != 3) {
        throw std::runtime_error("expected exactly three leaves");
    }
}

void test_best_first_builder_stops_before_limit_when_no_valid_split_exists() {
    ml::Matrix X(4, 1);
    X << 1.0,
         1.0,
         1.0,
         1.0;

    const ml::Vector y = make_vector({0.0, 0.0, 1.0, 1.0});

    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = 4;
    options.max_depth = 10;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree_best_first(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected constant-feature data to remain root leaf");
    }

    if (count_leaves(*root) != 1) {
        throw std::runtime_error("expected one leaf when no valid split exists");
    }
}

void test_decision_tree_classifier_uses_best_first_when_max_leaf_nodes_is_set() {
    const ml::Matrix X = make_max_leaf_tree_X();
    const ml::Vector y = make_max_leaf_tree_y();

    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = 2;
    options.max_depth = 10;

    ml::DecisionTreeClassifier tree(options);
    tree.fit(X, y);

    const ml::Vector predictions = tree.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected predictions for all training rows");
    }

    if (!tree.is_fitted()) {
        throw std::runtime_error("expected fitted tree");
    }
}

void test_decision_tree_classifier_rejects_zero_max_leaf_nodes() {
    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = 0;

    ml::DecisionTreeClassifier tree(options);
}

// -----------------------------------------------------------------------------
// Phase 6B advanced option validation and missing-value rejection tests
// -----------------------------------------------------------------------------

void test_decision_tree_options_accept_advanced_defaults() {
    ml::DecisionTreeOptions options;

    ml::validate_decision_tree_options(
        options,
        "test_decision_tree_options_accept_advanced_defaults"
    );
}

void test_decision_tree_options_accept_valid_max_leaf_nodes() {
    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = 3;

    ml::validate_decision_tree_options(
        options,
        "test_decision_tree_options_accept_valid_max_leaf_nodes"
    );
}

void test_decision_tree_options_reject_zero_max_leaf_nodes() {
    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = 0;

    ml::validate_decision_tree_options(
        options,
        "test_decision_tree_options_reject_zero_max_leaf_nodes"
    );
}

void test_decision_tree_options_accept_valid_max_features() {
    ml::DecisionTreeOptions options;
    options.max_features = 2;

    ml::validate_decision_tree_options(
        options,
        "test_decision_tree_options_accept_valid_max_features"
    );
}

void test_decision_tree_options_reject_zero_max_features() {
    ml::DecisionTreeOptions options;
    options.max_features = 0;

    ml::validate_decision_tree_options(
        options,
        "test_decision_tree_options_reject_zero_max_features"
    );
}

void test_decision_tree_options_store_balanced_class_weight_flag() {
    ml::DecisionTreeOptions options;
    options.use_balanced_class_weight = true;

    ml::validate_decision_tree_options(
        options,
        "test_decision_tree_options_store_balanced_class_weight_flag"
    );

    if (!options.use_balanced_class_weight) {
        throw std::runtime_error("expected balanced class weight flag to be stored");
    }
}

void test_decision_tree_options_store_random_seed() {
    ml::DecisionTreeOptions options;
    options.random_seed = 123;

    ml::validate_decision_tree_options(
        options,
        "test_decision_tree_options_store_random_seed"
    );

    if (options.random_seed != 123) {
        throw std::runtime_error("expected random_seed to be stored");
    }
}

void test_decision_tree_rejects_missing_values_during_fit() {
    ml::Matrix X = make_simple_tree_X();
    X(0, 0) = std::numeric_limits<double>::quiet_NaN();

    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeClassifier tree;
    tree.fit(X, y);
}

void test_decision_tree_rejects_missing_values_during_predict() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeClassifier tree;
    tree.fit(X, y);

    ml::Matrix X_bad = X;
    X_bad(0, 0) = std::numeric_limits<double>::quiet_NaN();

    static_cast<void>(tree.predict(X_bad));
}

// -----------------------------------------------------------------------------
// max_features feature-subsampling tests
// -----------------------------------------------------------------------------

ml::Matrix make_max_features_tree_X() {
    ml::Matrix X(6, 3);
    X << 0.0, 0.0, 5.0,
         0.0, 1.0, 4.0,
         0.0, 2.0, 3.0,
         1.0, 3.0, 2.0,
         1.0, 4.0, 1.0,
         1.0, 5.0, 0.0;

    return X;
}

ml::Vector make_max_features_tree_y() {
    return make_vector({
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0
    });
}

void test_find_best_split_accepts_max_features_equal_to_feature_count() {
    const ml::Matrix X = make_max_features_tree_X();
    const ml::Vector y = make_max_features_tree_y();

    ml::DecisionTreeOptions options;
    options.max_features = static_cast<std::size_t>(X.cols());
    options.random_seed = 42;

    const ml::SplitCandidate split = ml::find_best_split(
        X,
        y,
        options
    );

    if (!split.valid) {
        throw std::runtime_error("expected valid split with max_features equal to feature count");
    }
}

void test_find_best_split_rejects_max_features_larger_than_feature_count() {
    const ml::Matrix X = make_max_features_tree_X();
    const ml::Vector y = make_max_features_tree_y();

    ml::DecisionTreeOptions options;
    options.max_features = static_cast<std::size_t>(X.cols()) + 1;
    options.random_seed = 42;

    static_cast<void>(ml::find_best_split(
        X,
        y,
        options
    ));
}

void test_find_best_split_with_max_features_one_is_deterministic_for_same_seed() {
    const ml::Matrix X = make_max_features_tree_X();
    const ml::Vector y = make_max_features_tree_y();

    ml::DecisionTreeOptions options;
    options.max_features = 1;
    options.random_seed = 123;

    const ml::SplitCandidate split_a = ml::find_best_split(
        X,
        y,
        options
    );

    const ml::SplitCandidate split_b = ml::find_best_split(
        X,
        y,
        options
    );

    if (split_a.valid != split_b.valid) {
        throw std::runtime_error("expected deterministic split validity with same seed");
    }

    if (split_a.feature_index != split_b.feature_index) {
        throw std::runtime_error("expected deterministic feature selection with same seed");
    }

    test::assert_almost_equal(
        split_a.threshold,
        split_b.threshold,
        "test_find_best_split_with_max_features_one_is_deterministic_for_same_seed threshold"
    );

    test::assert_almost_equal(
        split_a.impurity_decrease,
        split_b.impurity_decrease,
        "test_find_best_split_with_max_features_one_is_deterministic_for_same_seed impurity_decrease"
    );
}

void test_decision_tree_classifier_accepts_max_features() {
    const ml::Matrix X = make_max_features_tree_X();
    const ml::Vector y = make_max_features_tree_y();

    ml::DecisionTreeOptions options;
    options.max_features = 1;
    options.random_seed = 123;
    options.max_depth = 3;

    ml::DecisionTreeClassifier tree(options);
    tree.fit(X, y);

    const ml::Vector predictions = tree.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected prediction for each sample");
    }

    if (!tree.is_fitted()) {
        throw std::runtime_error("expected tree to be fitted");
    }
}

void test_decision_tree_classifier_rejects_max_features_larger_than_feature_count() {
    const ml::Matrix X = make_max_features_tree_X();
    const ml::Vector y = make_max_features_tree_y();

    ml::DecisionTreeOptions options;
    options.max_features = static_cast<std::size_t>(X.cols()) + 1;
    options.max_depth = 3;

    ml::DecisionTreeClassifier tree(options);

    tree.fit(X, y);
}

void test_best_first_builder_accepts_max_features() {
    const ml::Matrix X = make_max_features_tree_X();
    const ml::Vector y = make_max_features_tree_y();

    ml::DecisionTreeOptions options;
    options.max_leaf_nodes = 3;
    options.max_features = 1;
    options.random_seed = 123;
    options.max_depth = 10;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree_best_first(
        X,
        y,
        options
    );

    if (!root) {
        throw std::runtime_error("expected best-first builder to return root");
    }

    if (count_leaves(*root) > options.max_leaf_nodes.value()) {
        throw std::runtime_error("expected best-first builder to respect max_leaf_nodes");
    }
}

// -----------------------------------------------------------------------------
// balanced class_weight tests
// -----------------------------------------------------------------------------

void test_balanced_class_weights_compute_expected_values() {
    const ml::Vector y = make_vector({
        0.0, 0.0, 0.0,
        1.0
    });

    const std::map<int, double> weights =
        ml::balanced_class_weights(y);

    test::assert_almost_equal(
        weights.at(0),
        4.0 / (2.0 * 3.0),
        "test_balanced_class_weights_compute_expected_values class 0"
    );

    test::assert_almost_equal(
        weights.at(1),
        4.0 / (2.0 * 1.0),
        "test_balanced_class_weights_compute_expected_values class 1"
    );
}

void test_sample_weights_from_class_weights_creates_expected_vector() {
    const ml::Vector y = make_vector({
        0.0, 0.0, 0.0,
        1.0
    });

    const std::map<int, double> class_weights =
        ml::balanced_class_weights(y);

    const ml::Vector sample_weight =
        ml::sample_weights_from_class_weights(
            y,
            class_weights
        );

    test::assert_almost_equal(
        sample_weight(0),
        4.0 / (2.0 * 3.0),
        "test_sample_weights_from_class_weights_creates_expected_vector sample 0"
    );

    test::assert_almost_equal(
        sample_weight(3),
        4.0 / (2.0 * 1.0),
        "test_sample_weights_from_class_weights_creates_expected_vector sample 3"
    );
}

void test_weighted_gini_impurity_uses_sample_weights() {
    const ml::Vector y = make_vector({
        0.0, 0.0, 0.0,
        1.0
    });

    const std::map<int, double> class_weights =
        ml::balanced_class_weights(y);

    const ml::Vector sample_weight =
        ml::sample_weights_from_class_weights(
            y,
            class_weights
        );

    test::assert_almost_equal(
        ml::weighted_gini_impurity(y, sample_weight),
        0.5,
        "test_weighted_gini_impurity_uses_sample_weights"
    );
}

void test_weighted_entropy_uses_sample_weights() {
    const ml::Vector y = make_vector({
        0.0, 0.0, 0.0,
        1.0
    });

    const std::map<int, double> class_weights =
        ml::balanced_class_weights(y);

    const ml::Vector sample_weight =
        ml::sample_weights_from_class_weights(
            y,
            class_weights
        );

    test::assert_almost_equal(
        ml::weighted_entropy(y, sample_weight),
        1.0,
        "test_weighted_entropy_uses_sample_weights"
    );
}

void test_decision_tree_classifier_accepts_balanced_class_weight() {
    ml::Matrix X(6, 1);
    X << 0.0,
         1.0,
         2.0,
         3.0,
         4.0,
         5.0;

    const ml::Vector y = make_vector({
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0
    });

    ml::DecisionTreeOptions options;
    options.use_balanced_class_weight = true;
    options.max_depth = 3;

    ml::DecisionTreeClassifier tree(options);
    tree.fit(X, y);

    const ml::Vector predictions = tree.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected predictions for all samples");
    }

    if (!tree.is_fitted()) {
        throw std::runtime_error("expected fitted tree");
    }
}

void test_decision_tree_classifier_accepts_balanced_class_weight_with_best_first() {
    ml::Matrix X(6, 1);
    X << 0.0,
         1.0,
         2.0,
         3.0,
         4.0,
         5.0;

    const ml::Vector y = make_vector({
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0
    });

    ml::DecisionTreeOptions options;
    options.use_balanced_class_weight = true;
    options.max_leaf_nodes = 3;
    options.max_depth = 10;

    ml::DecisionTreeClassifier tree(options);
    tree.fit(X, y);

    const ml::Vector predictions = tree.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected predictions for all samples");
    }

    if (!tree.is_fitted()) {
        throw std::runtime_error("expected fitted tree");
    }
}

// -----------------------------------------------------------------------------
// sample_weight impurity and split-scoring tests
// -----------------------------------------------------------------------------

void test_weighted_gini_impurity_matches_unweighted_when_weights_are_equal() {
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_balanced_sample_weight_vector();

    test::assert_almost_equal(
        ml::weighted_gini_impurity(y, sample_weight),
        ml::gini_impurity(y),
        "test_weighted_gini_impurity_matches_unweighted_when_weights_are_equal"
    );
}

void test_weighted_entropy_matches_unweighted_when_weights_are_equal() {
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_balanced_sample_weight_vector();

    test::assert_almost_equal(
        ml::weighted_entropy(y, sample_weight),
        ml::entropy(y),
        "test_weighted_entropy_matches_unweighted_when_weights_are_equal"
    );
}

void test_weighted_gini_impurity_changes_with_unequal_weights() {
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_unbalanced_sample_weight_vector();

    const double class_0_weight = 2.0;
    const double class_1_weight = 6.0;
    const double total_weight = 8.0;

    const double p0 = class_0_weight / total_weight;
    const double p1 = class_1_weight / total_weight;

    const double expected = 1.0 - (p0 * p0 + p1 * p1);

    test::assert_almost_equal(
        ml::weighted_gini_impurity(y, sample_weight),
        expected,
        "test_weighted_gini_impurity_changes_with_unequal_weights"
    );
}

void test_weighted_entropy_changes_with_unequal_weights() {
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_unbalanced_sample_weight_vector();

    const double p0 = 2.0 / 8.0;
    const double p1 = 6.0 / 8.0;

    const double expected =
        -p0 * std::log2(p0) -
        p1 * std::log2(p1);

    test::assert_almost_equal(
        ml::weighted_entropy(y, sample_weight),
        expected,
        "test_weighted_entropy_changes_with_unequal_weights"
    );
}

void test_weighted_child_impurity_by_weight_computes_expected_value() {
    const double result = ml::weighted_child_impurity_by_weight(
        0.0,
        0.5,
        2.0,
        6.0
    );

    const double expected =
        (2.0 / 8.0) * 0.0 +
        (6.0 / 8.0) * 0.5;

    test::assert_almost_equal(
        result,
        expected,
        "test_weighted_child_impurity_by_weight_computes_expected_value"
    );
}

void test_evaluate_candidate_threshold_with_sample_weight_finds_valid_split() {
    const ml::Matrix X = make_sample_weight_split_X();
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_unbalanced_sample_weight_vector();

    const ml::SplitCandidate split = ml::evaluate_candidate_threshold(
        X,
        y,
        sample_weight,
        0,
        1.5
    );

    if (!split.valid) {
        throw std::runtime_error("expected weighted split candidate to be valid");
    }

    if (split.left_count != 2 || split.right_count != 2) {
        throw std::runtime_error("expected weighted split to preserve child counts");
    }

    test::assert_almost_equal(
        split.left_impurity,
        0.0,
        "test_evaluate_candidate_threshold_with_sample_weight_finds_valid_split left_impurity"
    );

    test::assert_almost_equal(
        split.right_impurity,
        0.0,
        "test_evaluate_candidate_threshold_with_sample_weight_finds_valid_split right_impurity"
    );

    test::assert_almost_equal(
        split.weighted_child_impurity,
        0.0,
        "test_evaluate_candidate_threshold_with_sample_weight_finds_valid_split weighted_child_impurity"
    );
}

void test_find_best_split_with_sample_weight_returns_valid_split() {
    const ml::Matrix X = make_sample_weight_split_X();
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_unbalanced_sample_weight_vector();

    const ml::SplitCandidate split = ml::find_best_split(
        X,
        y,
        sample_weight
    );

    if (!split.valid) {
        throw std::runtime_error("expected weighted best split to be valid");
    }

    if (split.feature_index != 0) {
        throw std::runtime_error("expected weighted best split to use feature 0");
    }

    test::assert_almost_equal(
        split.threshold,
        1.5,
        "test_find_best_split_with_sample_weight_returns_valid_split threshold"
    );
}

void test_weighted_gini_impurity_rejects_mismatched_sample_weight_size() {
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_vector({1.0, 1.0});

    static_cast<void>(ml::weighted_gini_impurity(
        y,
        sample_weight
    ));
}

void test_weighted_gini_impurity_rejects_negative_sample_weight() {
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_vector({
        1.0,
        -1.0,
        1.0,
        1.0
    });

    static_cast<void>(ml::weighted_gini_impurity(
        y,
        sample_weight
    ));
}

void test_weighted_gini_impurity_rejects_zero_total_sample_weight() {
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_vector({
        0.0,
        0.0,
        0.0,
        0.0
    });

    static_cast<void>(ml::weighted_gini_impurity(
        y,
        sample_weight
    ));
}

void test_weighted_gini_impurity_rejects_non_finite_sample_weight() {
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_vector({
        1.0,
        std::numeric_limits<double>::infinity(),
        1.0,
        1.0
    });

    static_cast<void>(ml::weighted_gini_impurity(
        y,
        sample_weight
    ));
}

void test_evaluate_candidate_threshold_with_sample_weight_rejects_mismatched_weights() {
    const ml::Matrix X = make_sample_weight_split_X();
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_vector({1.0, 1.0});

    static_cast<void>(ml::evaluate_candidate_threshold(
        X,
        y,
        sample_weight,
        0,
        1.5
    ));
}

void test_find_best_split_with_sample_weight_rejects_mismatched_weights() {
    const ml::Matrix X = make_sample_weight_split_X();
    const ml::Vector y = make_sample_weight_split_y();
    const ml::Vector sample_weight = make_vector({1.0, 1.0});

    static_cast<void>(ml::find_best_split(
        X,
        y,
        sample_weight
    ));
}

// -----------------------------------------------------------------------------
// Advanced tree controls experiment export helpers
// -----------------------------------------------------------------------------

const std::string k_phase6b_output_dir = "outputs/phase-6b-tree-ensembles";

void ensure_phase6b_output_dir_exists() {
    std::filesystem::create_directories(k_phase6b_output_dir);
}

struct AdvancedTreeControlExperimentResult {
    std::string experiment_name;
    std::string variant_name;
    std::string status;
    std::string notes;

    std::size_t max_depth{0};
    std::string max_leaf_nodes;
    std::string max_features;
    bool use_balanced_class_weight{false};

    double accuracy{0.0};
    std::size_t num_predictions{0};
};

ml::Matrix make_advanced_tree_control_X() {
    ml::Matrix X(12, 3);
    X << 0.0, 0.0, 3.0,
         0.5, 0.1, 3.0,
         1.0, 0.0, 3.0,
         1.5, 0.2, 3.0,
         2.0, 1.0, 2.0,
         2.5, 1.1, 2.0,
         3.0, 1.0, 1.0,
         3.5, 1.2, 1.0,
         4.0, 0.0, 0.0,
         4.5, 0.1, 0.0,
         5.0, 0.0, 0.0,
         5.5, 0.2, 0.0;

    return X;
}

ml::Vector make_advanced_tree_control_y() {
    return make_vector({
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0
    });
}

ml::Matrix make_advanced_tree_control_X_eval() {
    ml::Matrix X(6, 3);
    X << 0.25, 0.0, 3.0,
         1.25, 0.1, 3.0,
         2.25, 1.0, 2.0,
         3.25, 1.1, 1.0,
         4.25, 0.0, 0.0,
         5.25, 0.1, 0.0;

    return X;
}

ml::Vector make_advanced_tree_control_y_eval() {
    return make_vector({
        0.0, 0.0,
        1.0, 1.0,
        0.0, 0.0
    });
}

std::string optional_size_to_string(
    const std::optional<std::size_t>& value
) {
    if (!value.has_value()) {
        return "none";
    }

    return std::to_string(value.value());
}

AdvancedTreeControlExperimentResult run_advanced_tree_control_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const std::string& notes,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::DecisionTreeOptions& options
) {
    ml::DecisionTreeClassifier tree(options);
    tree.fit(X_train, y_train);

    const ml::Vector predictions = tree.predict(X_eval);

    return AdvancedTreeControlExperimentResult{
        experiment_name,
        variant_name,
        "implemented",
        notes,
        options.max_depth,
        optional_size_to_string(options.max_leaf_nodes),
        optional_size_to_string(options.max_features),
        options.use_balanced_class_weight,
        ml::accuracy_score(predictions, y_eval),
        static_cast<std::size_t>(predictions.size())
    };
}

AdvancedTreeControlExperimentResult make_deferred_tree_control_result(
    const std::string& experiment_name,
    const std::string& variant_name,
    const std::string& notes
) {
    return AdvancedTreeControlExperimentResult{
        experiment_name,
        variant_name,
        "deferred",
        notes,
        0,
        "n/a",
        "n/a",
        false,
        0.0,
        0
    };
}

void export_advanced_tree_control_results_csv(
    const std::vector<AdvancedTreeControlExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_advanced_tree_control_results_csv: failed to open output file"
        );
    }

    file << "experiment_name,variant_name,status,notes,max_depth,"
         << "max_leaf_nodes,max_features,use_balanced_class_weight,"
         << "accuracy,num_predictions\n";

    for (const AdvancedTreeControlExperimentResult& result : results) {
        file << result.experiment_name << ","
             << result.variant_name << ","
             << result.status << ","
             << result.notes << ","
             << result.max_depth << ","
             << result.max_leaf_nodes << ","
             << result.max_features << ","
             << (result.use_balanced_class_weight ? "true" : "false") << ","
             << result.accuracy << ","
             << result.num_predictions << "\n";
    }
}

void export_advanced_tree_control_results_txt(
    const std::vector<AdvancedTreeControlExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_advanced_tree_control_results_txt: failed to open output file"
        );
    }

    file << "Advanced Tree Controls Comparison\n\n";

    file << "This file compares Phase 6B advanced tree controls.\n"
         << "Implemented comparisons are exported with metrics.\n"
         << "Deferred features are explicitly listed instead of being silently skipped.\n\n";

    for (const AdvancedTreeControlExperimentResult& result : results) {
        file << "Experiment: " << result.experiment_name << "\n"
             << "Variant: " << result.variant_name << "\n"
             << "Status: " << result.status << "\n"
             << "Notes: " << result.notes << "\n"
             << "max_depth: " << result.max_depth << "\n"
             << "max_leaf_nodes: " << result.max_leaf_nodes << "\n"
             << "max_features: " << result.max_features << "\n"
             << "use_balanced_class_weight: "
             << (result.use_balanced_class_weight ? "true" : "false") << "\n"
             << "Accuracy: " << result.accuracy << "\n"
             << "Predictions: " << result.num_predictions << "\n\n";
    }
}

// -----------------------------------------------------------------------------
// Advanced tree controls experiment export tests
// -----------------------------------------------------------------------------

void test_experiment_exports_advanced_tree_controls_comparison() {
    ensure_phase6b_output_dir_exists();

    const ml::Matrix X_train = make_advanced_tree_control_X();
    const ml::Vector y_train = make_advanced_tree_control_y();
    const ml::Matrix X_eval = make_advanced_tree_control_X_eval();
    const ml::Vector y_eval = make_advanced_tree_control_y_eval();

    std::vector<AdvancedTreeControlExperimentResult> results;

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 10;

        results.push_back(run_advanced_tree_control_experiment(
            "unrestricted_vs_max_leaf_nodes",
            "unrestricted_recursive_tree",
            "No max_leaf_nodes limit; uses the regular recursive builder.",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 10;
        options.max_leaf_nodes = 2;

        results.push_back(run_advanced_tree_control_experiment(
            "unrestricted_vs_max_leaf_nodes",
            "best_first_max_leaf_nodes_2",
            "Uses best-first growth with a two-leaf budget.",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 10;

        results.push_back(run_advanced_tree_control_experiment(
            "all_features_vs_feature_subsampling",
            "all_features",
            "All features are considered at each split.",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 10;
        options.max_features = 1;
        options.random_seed = 123;

        results.push_back(run_advanced_tree_control_experiment(
            "all_features_vs_feature_subsampling",
            "max_features_1",
            "Only one seeded random feature is considered per split.",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 10;

        results.push_back(run_advanced_tree_control_experiment(
            "unweighted_vs_balanced_class_weight",
            "unweighted",
            "Standard unweighted impurity and majority-class leaves.",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 10;
        options.use_balanced_class_weight = true;

        results.push_back(run_advanced_tree_control_experiment(
            "unweighted_vs_balanced_class_weight",
            "balanced_class_weight",
            "Balanced class weights are converted into sample weights during fitting.",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    results.push_back(make_deferred_tree_control_result(
        "complete_data_vs_missing_value_handling",
        "complete_data",
        "Complete finite data is supported by the current tree implementation."
    ));

    results.push_back(make_deferred_tree_control_result(
        "complete_data_vs_missing_value_handling",
        "missing_value_rejection",
        "Missing values are explicitly rejected during fit and predict. Learned missing direction is deferred."
    ));

    results.push_back(make_deferred_tree_control_result(
        "unpruned_vs_pruned_tree_behavior",
        "unpruned_tree",
        "Unpruned behavior is represented by the standard fitted tree with stopping rules."
    ));

    results.push_back(make_deferred_tree_control_result(
        "unpruned_vs_pruned_tree_behavior",
        "cost_complexity_pruning_deferred",
        "Simple cost-complexity post-pruning is feasible but deferred to avoid blocking Random Forest and Gradient Boosting."
    ));

    const std::string csv_path =
        k_phase6b_output_dir + "/advanced_tree_controls_comparison.csv";

    const std::string txt_path =
        k_phase6b_output_dir + "/advanced_tree_controls_comparison.txt";

    export_advanced_tree_control_results_csv(
        results,
        csv_path
    );

    export_advanced_tree_control_results_txt(
        results,
        txt_path
    );

    if (!std::filesystem::exists(csv_path)) {
        throw std::runtime_error(
            "expected advanced_tree_controls_comparison.csv to exist"
        );
    }

    if (!std::filesystem::exists(txt_path)) {
        throw std::runtime_error(
            "expected advanced_tree_controls_comparison.txt to exist"
        );
    }
}

// -----------------------------------------------------------------------------
// Bootstrap sampling utility tests
// -----------------------------------------------------------------------------

ml::Matrix make_bootstrap_test_X() {
    ml::Matrix X(5, 2);
    X << 0.0, 10.0,
         1.0, 11.0,
         2.0, 12.0,
         3.0, 13.0,
         4.0, 14.0;

    return X;
}

ml::Vector make_bootstrap_test_y() {
    return make_vector({
        0.0,
        1.0,
        0.0,
        1.0,
        0.0
    });
}

void test_make_bootstrap_sample_preserves_shape() {
    const ml::Matrix X = make_bootstrap_test_X();
    const ml::Vector y = make_bootstrap_test_y();

    const ml::BootstrapSample sample =
        ml::make_bootstrap_sample(
            X,
            y,
            42
        );

    if (sample.X.rows() != X.rows()) {
        throw std::runtime_error("expected bootstrap X to preserve row count");
    }

    if (sample.X.cols() != X.cols()) {
        throw std::runtime_error("expected bootstrap X to preserve feature count");
    }

    if (sample.y.size() != y.size()) {
        throw std::runtime_error("expected bootstrap y to preserve target count");
    }

    if (sample.sampled_indices.size() != static_cast<std::size_t>(X.rows())) {
        throw std::runtime_error("expected one sampled index per bootstrap row");
    }
}

void test_make_bootstrap_sample_indices_are_in_range() {
    const ml::Matrix X = make_bootstrap_test_X();
    const ml::Vector y = make_bootstrap_test_y();

    const ml::BootstrapSample sample =
        ml::make_bootstrap_sample(
            X,
            y,
            42
        );

    for (Eigen::Index index : sample.sampled_indices) {
        if (index < 0 || index >= X.rows()) {
            throw std::runtime_error("bootstrap sampled index out of range");
        }
    }

    for (Eigen::Index index : sample.out_of_bag_indices) {
        if (index < 0 || index >= X.rows()) {
            throw std::runtime_error("bootstrap out-of-bag index out of range");
        }
    }
}

void test_make_bootstrap_sample_rows_match_sampled_indices() {
    const ml::Matrix X = make_bootstrap_test_X();
    const ml::Vector y = make_bootstrap_test_y();

    const ml::BootstrapSample sample =
        ml::make_bootstrap_sample(
            X,
            y,
            42
        );

    for (Eigen::Index row = 0; row < sample.X.rows(); ++row) {
        const Eigen::Index sampled_index =
            sample.sampled_indices[static_cast<std::size_t>(row)];

        test::assert_almost_equal(
            sample.X(row, 0),
            X(sampled_index, 0),
            "test_make_bootstrap_sample_rows_match_sampled_indices X column 0"
        );

        test::assert_almost_equal(
            sample.X(row, 1),
            X(sampled_index, 1),
            "test_make_bootstrap_sample_rows_match_sampled_indices X column 1"
        );

        test::assert_almost_equal(
            sample.y(row),
            y(sampled_index),
            "test_make_bootstrap_sample_rows_match_sampled_indices y"
        );
    }
}

void test_make_bootstrap_sample_is_reproducible_with_same_seed() {
    const ml::Matrix X = make_bootstrap_test_X();
    const ml::Vector y = make_bootstrap_test_y();

    const ml::BootstrapSample sample_a =
        ml::make_bootstrap_sample(
            X,
            y,
            123
        );

    const ml::BootstrapSample sample_b =
        ml::make_bootstrap_sample(
            X,
            y,
            123
        );

    if (sample_a.sampled_indices != sample_b.sampled_indices) {
        throw std::runtime_error("expected same sampled indices with same seed");
    }

    if (sample_a.out_of_bag_indices != sample_b.out_of_bag_indices) {
        throw std::runtime_error("expected same out-of-bag indices with same seed");
    }

    test::assert_matrix_almost_equal(
        sample_a.X,
        sample_b.X,
        "test_make_bootstrap_sample_is_reproducible_with_same_seed X"
    );

    test::assert_vector_almost_equal(
        sample_a.y,
        sample_b.y,
        "test_make_bootstrap_sample_is_reproducible_with_same_seed y"
    );
}

void test_make_bootstrap_sample_changes_with_different_seed() {
    const ml::Matrix X = make_bootstrap_test_X();
    const ml::Vector y = make_bootstrap_test_y();

    const ml::BootstrapSample sample_a =
        ml::make_bootstrap_sample(
            X,
            y,
            123
        );

    const ml::BootstrapSample sample_b =
        ml::make_bootstrap_sample(
            X,
            y,
            456
        );

    if (sample_a.sampled_indices == sample_b.sampled_indices) {
        throw std::runtime_error(
            "expected different sampled indices with different seeds"
        );
    }
}

void test_make_bootstrap_sample_tracks_out_of_bag_indices() {
    const ml::Matrix X = make_bootstrap_test_X();
    const ml::Vector y = make_bootstrap_test_y();

    const ml::BootstrapSample sample =
        ml::make_bootstrap_sample(
            X,
            y,
            42
        );

    std::vector<bool> was_sampled(
        static_cast<std::size_t>(X.rows()),
        false
    );

    for (Eigen::Index index : sample.sampled_indices) {
        was_sampled[static_cast<std::size_t>(index)] = true;
    }

    for (Eigen::Index index : sample.out_of_bag_indices) {
        if (was_sampled[static_cast<std::size_t>(index)]) {
            throw std::runtime_error(
                "out-of-bag index should not appear in sampled indices"
            );
        }
    }
}

void test_make_full_sample_preserves_original_data() {
    const ml::Matrix X = make_bootstrap_test_X();
    const ml::Vector y = make_bootstrap_test_y();

    const ml::BootstrapSample sample =
        ml::make_full_sample(
            X,
            y
        );

    test::assert_matrix_almost_equal(
        sample.X,
        X,
        "test_make_full_sample_preserves_original_data X"
    );

    test::assert_vector_almost_equal(
        sample.y,
        y,
        "test_make_full_sample_preserves_original_data y"
    );

    if (!sample.out_of_bag_indices.empty()) {
        throw std::runtime_error("expected full sample to have no out-of-bag rows");
    }

    if (sample.sampled_indices.size() != static_cast<std::size_t>(X.rows())) {
        throw std::runtime_error("expected full sample to track all row indices");
    }

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        if (sample.sampled_indices[static_cast<std::size_t>(i)] != i) {
            throw std::runtime_error("expected full sample indices to preserve order");
        }
    }
}

void test_make_bootstrap_sample_rejects_empty_X() {
    const ml::Matrix X;
    const ml::Vector y = make_bootstrap_test_y();

    static_cast<void>(ml::make_bootstrap_sample(
        X,
        y,
        42
    ));
}

void test_make_bootstrap_sample_rejects_empty_y() {
    const ml::Matrix X = make_bootstrap_test_X();
    const ml::Vector y;

    static_cast<void>(ml::make_bootstrap_sample(
        X,
        y,
        42
    ));
}

void test_make_bootstrap_sample_rejects_mismatched_X_y() {
    ml::Matrix X(3, 1);
    X << 0.0,
         1.0,
         2.0;

    const ml::Vector y = make_vector({
        0.0,
        1.0
    });

    static_cast<void>(ml::make_bootstrap_sample(
        X,
        y,
        42
    ));
}

void test_make_full_sample_rejects_empty_X() {
    const ml::Matrix X;
    const ml::Vector y = make_bootstrap_test_y();

    static_cast<void>(ml::make_full_sample(
        X,
        y
    ));
}

void test_make_full_sample_rejects_empty_y() {
    const ml::Matrix X = make_bootstrap_test_X();
    const ml::Vector y;

    static_cast<void>(ml::make_full_sample(
        X,
        y
    ));
}

void test_make_full_sample_rejects_mismatched_X_y() {
    ml::Matrix X(3, 1);
    X << 0.0,
         1.0,
         2.0;

    const ml::Vector y = make_vector({
        0.0,
        1.0
    });

    static_cast<void>(ml::make_full_sample(
        X,
        y
    ));
}

// -----------------------------------------------------------------------------
// RandomForestClassifier tests
// -----------------------------------------------------------------------------

ml::Matrix make_random_forest_test_X() {
    ml::Matrix X(8, 2);
    X << 0.0, 0.0,
         0.5, 0.1,
         1.0, 0.0,
         1.5, 0.1,
         3.0, 1.0,
         3.5, 1.1,
         4.0, 1.0,
         4.5, 1.1;

    return X;
}

ml::Vector make_random_forest_test_y() {
    return make_vector({
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0
    });
}

void test_random_forest_options_accept_defaults() {
    const ml::RandomForestOptions options;

    ml::validate_random_forest_options(
        options,
        "test_random_forest_options_accept_defaults"
    );
}

void test_random_forest_options_reject_zero_estimators() {
    ml::RandomForestOptions options;
    options.n_estimators = 0;

    ml::validate_random_forest_options(
        options,
        "test_random_forest_options_reject_zero_estimators"
    );
}

void test_random_forest_options_reject_zero_max_features() {
    ml::RandomForestOptions options;
    options.max_features = 0;

    ml::validate_random_forest_options(
        options,
        "test_random_forest_options_reject_zero_max_features"
    );
}

void test_random_forest_reports_not_fitted_initially() {
    const ml::RandomForestClassifier forest;

    if (forest.is_fitted()) {
        throw std::runtime_error("expected forest to start unfitted");
    }
}

void test_random_forest_fit_marks_model_as_fitted() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestOptions options;
    options.n_estimators = 5;
    options.random_seed = 42;
    options.tree_options.max_depth = 3;

    ml::RandomForestClassifier forest(options);
    forest.fit(X, y);

    if (!forest.is_fitted()) {
        throw std::runtime_error("expected forest to be fitted");
    }

    if (forest.num_trees() != 5) {
        throw std::runtime_error("expected forest to store 5 trees");
    }

    if (forest.num_classes() != 2) {
        throw std::runtime_error("expected forest to infer 2 classes");
    }
}

void test_random_forest_predict_returns_expected_shape() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestOptions options;
    options.n_estimators = 7;
    options.random_seed = 42;
    options.tree_options.max_depth = 3;

    ml::RandomForestClassifier forest(options);
    forest.fit(X, y);

    const ml::Vector predictions = forest.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected one prediction per sample");
    }
}

void test_random_forest_predict_proba_returns_expected_shape() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();
    
    ml::RandomForestOptions options;
    options.n_estimators = 7;
    options.random_seed = 42;
    options.tree_options.max_depth = 3;
    
    ml::RandomForestClassifier forest(options);
    forest.fit(X, y);
    
    const ml::Matrix probabilities = forest.predict_proba(X);
    
    if (probabilities.rows() != X.rows()) {
        throw std::runtime_error("expected one probability row per sample");
    }
    
    if (probabilities.cols() != 2) {
        throw std::runtime_error("expected one probability column per class");
    }
}

void test_random_forest_predict_proba_rows_sum_to_one() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestOptions options;
    options.n_estimators = 9;
    options.random_seed = 42;
    options.tree_options.max_depth = 3;

    ml::RandomForestClassifier forest(options);
    forest.fit(X, y);

    const ml::Matrix probabilities = forest.predict_proba(X);

    for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
        test::assert_almost_equal(
            probabilities.row(i).sum(),
            1.0,
            "test_random_forest_predict_proba_rows_sum_to_one"
        );
    }
}

void test_random_forest_predict_matches_predict_proba_argmax() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestOptions options;
    options.n_estimators = 9;
    options.random_seed = 42;
    options.tree_options.max_depth = 3;

    ml::RandomForestClassifier forest(options);
    forest.fit(X, y);

    const ml::Vector predictions = forest.predict(X);
    const ml::Matrix probabilities = forest.predict_proba(X);

    for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
        Eigen::Index best_class = 0;
        double best_probability = probabilities(i, 0);

        for (Eigen::Index class_index = 1; class_index < probabilities.cols(); ++class_index) {
            if (probabilities(i, class_index) > best_probability) {
                best_probability = probabilities(i, class_index);
                best_class = class_index;
            }
        }

        test::assert_almost_equal(
            predictions(i),
            static_cast<double>(best_class),
            "test_random_forest_predict_matches_predict_proba_argmax"
        );
    }
}

void test_random_forest_is_reproducible_with_same_seed() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestOptions options;
    options.n_estimators = 11;
    options.random_seed = 123;
    options.max_features = 1;
    options.tree_options.max_depth = 3;

    ml::RandomForestClassifier forest_a(options);
    ml::RandomForestClassifier forest_b(options);

    forest_a.fit(X, y);
    forest_b.fit(X, y);

    const ml::Vector predictions_a = forest_a.predict(X);
    const ml::Vector predictions_b = forest_b.predict(X);

    test::assert_vector_almost_equal(
        predictions_a,
        predictions_b,
        "test_random_forest_is_reproducible_with_same_seed"
    );

    const ml::Matrix probabilities_a = forest_a.predict_proba(X);
    const ml::Matrix probabilities_b = forest_b.predict_proba(X);

    test::assert_matrix_almost_equal(
        probabilities_a,
        probabilities_b,
        "test_random_forest_is_reproducible_with_same_seed probabilities"
    );
}

void test_random_forest_supports_bootstrap_false() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestOptions options;
    options.n_estimators = 5;
    options.bootstrap = false;
    options.random_seed = 42;
    options.tree_options.max_depth = 3;

    ml::RandomForestClassifier forest(options);
    forest.fit(X, y);

    const ml::Vector predictions = forest.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected predictions with bootstrap disabled");
    }
}

void test_random_forest_supports_max_features() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestOptions options;
    options.n_estimators = 5;
    options.max_features = 1;
    options.random_seed = 42;
    options.tree_options.max_depth = 3;

    ml::RandomForestClassifier forest(options);
    forest.fit(X, y);

    const ml::Vector predictions = forest.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected predictions with max_features");
    }
}

void test_random_forest_rejects_predict_before_fit() {
    const ml::Matrix X = make_random_forest_test_X();

    const ml::RandomForestClassifier forest;

    static_cast<void>(forest.predict(X));
}

void test_random_forest_rejects_predict_proba_before_fit() {
    const ml::Matrix X = make_random_forest_test_X();

    const ml::RandomForestClassifier forest;

    static_cast<void>(forest.predict_proba(X));
}

void test_random_forest_rejects_empty_fit_X() {
    const ml::Matrix X;
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestClassifier forest;

    forest.fit(X, y);
}

void test_random_forest_rejects_empty_fit_y() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y;

    ml::RandomForestClassifier forest;

    forest.fit(X, y);
}

void test_random_forest_rejects_mismatched_fit_data() {
    ml::Matrix X(3, 1);
    X << 0.0,
         1.0,
         2.0;

    const ml::Vector y = make_vector({0.0, 1.0});

    ml::RandomForestClassifier forest;

    forest.fit(X, y);
}

void test_random_forest_rejects_invalid_class_labels() {
    const ml::Matrix X = make_random_forest_test_X();
    ml::Vector y = make_random_forest_test_y();
    y(0) = 0.5;

    ml::RandomForestClassifier forest;

    forest.fit(X, y);
}

void test_random_forest_rejects_predict_feature_mismatch() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestClassifier forest;
    forest.fit(X, y);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(forest.predict(X_bad));
}

void test_random_forest_rejects_predict_proba_feature_mismatch() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestClassifier forest;
    forest.fit(X, y);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(forest.predict_proba(X_bad));
}

void test_random_forest_rejects_non_finite_predict_values() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestClassifier forest;
    forest.fit(X, y);

    ml::Matrix X_bad = X;
    X_bad(0, 0) = std::numeric_limits<double>::quiet_NaN();

    static_cast<void>(forest.predict(X_bad));
}

void test_random_forest_rejects_max_features_larger_than_feature_count() {
    const ml::Matrix X = make_random_forest_test_X();
    const ml::Vector y = make_random_forest_test_y();

    ml::RandomForestOptions options;
    options.max_features = static_cast<std::size_t>(X.cols()) + 1;

    ml::RandomForestClassifier forest(options);

    forest.fit(X, y);
}

// -----------------------------------------------------------------------------
// Random Forest experiment export helpers
// -----------------------------------------------------------------------------

struct RandomForestExperimentResult {
    std::string experiment_name;
    std::string variant_name;
    std::string model_type;

    std::size_t n_estimators{0};
    bool bootstrap{false};
    std::string max_features;
    std::size_t max_depth{0};
    unsigned int random_seed{0};

    double accuracy{0.0};
    std::size_t num_predictions{0};
};

ml::Matrix make_random_forest_experiment_X() {
    ml::Matrix X(16, 4);

    X << 0.0, 0.0, 0.0, 3.0,
         0.4, 0.1, 0.0, 3.0,
         0.8, 0.0, 0.1, 3.0,
         1.2, 0.2, 0.0, 3.0,

         2.0, 1.0, 1.0, 2.0,
         2.4, 1.1, 1.0, 2.0,
         2.8, 1.0, 1.1, 2.0,
         3.2, 1.2, 1.0, 2.0,

         4.0, 0.0, 2.0, 1.0,
         4.4, 0.1, 2.0, 1.0,
         4.8, 0.0, 2.1, 1.0,
         5.2, 0.2, 2.0, 1.0,

         6.0, 1.0, 3.0, 0.0,
         6.4, 1.1, 3.0, 0.0,
         6.8, 1.0, 3.1, 0.0,
         7.2, 1.2, 3.0, 0.0;

    return X;
}

ml::Vector make_random_forest_experiment_y() {
    return make_vector({
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0
    });
}

ml::Matrix make_random_forest_experiment_X_eval() {
    ml::Matrix X(8, 4);

    X << 0.2, 0.0, 0.0, 3.0,
         1.0, 0.1, 0.1, 3.0,

         2.2, 1.0, 1.0, 2.0,
         3.0, 1.1, 1.1, 2.0,

         4.2, 0.0, 2.0, 1.0,
         5.0, 0.1, 2.1, 1.0,

         6.2, 1.0, 3.0, 0.0,
         7.0, 1.1, 3.1, 0.0;

    return X;
}

ml::Vector make_random_forest_experiment_y_eval() {
    return make_vector({
        0.0, 0.0,
        1.0, 1.0,
        0.0, 0.0,
        1.0, 1.0
    });
}

RandomForestExperimentResult run_single_tree_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::DecisionTreeOptions& options
) {
    ml::DecisionTreeClassifier tree(options);
    tree.fit(X_train, y_train);

    const ml::Vector predictions = tree.predict(X_eval);

    return RandomForestExperimentResult{
        experiment_name,
        variant_name,
        "DecisionTreeClassifier",
        1,
        false,
        optional_size_to_string(options.max_features),
        options.max_depth,
        options.random_seed,
        ml::accuracy_score(predictions, y_eval),
        static_cast<std::size_t>(predictions.size())
    };
}

RandomForestExperimentResult run_random_forest_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::RandomForestOptions& options
) {
    ml::RandomForestClassifier forest(options);
    forest.fit(X_train, y_train);

    const ml::Vector predictions = forest.predict(X_eval);

    return RandomForestExperimentResult{
        experiment_name,
        variant_name,
        "RandomForestClassifier",
        options.n_estimators,
        options.bootstrap,
        optional_size_to_string(options.max_features),
        options.tree_options.max_depth,
        options.random_seed,
        ml::accuracy_score(predictions, y_eval),
        static_cast<std::size_t>(predictions.size())
    };
}

void export_random_forest_results_csv(
    const std::vector<RandomForestExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_random_forest_results_csv: failed to open output file"
        );
    }

    file << "experiment_name,variant_name,model_type,n_estimators,bootstrap,"
         << "max_features,max_depth,random_seed,accuracy,num_predictions\n";

    for (const RandomForestExperimentResult& result : results) {
        file << result.experiment_name << ","
             << result.variant_name << ","
             << result.model_type << ","
             << result.n_estimators << ","
             << (result.bootstrap ? "true" : "false") << ","
             << result.max_features << ","
             << result.max_depth << ","
             << result.random_seed << ","
             << result.accuracy << ","
             << result.num_predictions << "\n";
    }
}

void export_random_forest_results_txt(
    const std::vector<RandomForestExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_random_forest_results_txt: failed to open output file"
        );
    }

    file << "Random Forest Comparison\n\n";

    file << "This experiment compares:\n"
         << "- single tree vs random forest\n"
         << "- number of trees\n"
         << "- max_features\n"
         << "- bootstrap vs no bootstrap\n"
         << "- predicted classes vs predicted probabilities\n\n";

    for (const RandomForestExperimentResult& result : results) {
        file << "Experiment: " << result.experiment_name << "\n"
             << "Variant: " << result.variant_name << "\n"
             << "Model: " << result.model_type << "\n"
             << "n_estimators: " << result.n_estimators << "\n"
             << "bootstrap: " << (result.bootstrap ? "true" : "false") << "\n"
             << "max_features: " << result.max_features << "\n"
             << "max_depth: " << result.max_depth << "\n"
             << "random_seed: " << result.random_seed << "\n"
             << "accuracy: " << result.accuracy << "\n"
             << "num_predictions: " << result.num_predictions << "\n\n";
    }
}

void export_random_forest_prediction_probabilities_csv(
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::Vector& predictions,
    const ml::Matrix& probabilities,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_random_forest_prediction_probabilities_csv: failed to open output file"
        );
    }

    file << "sample_index,y_true,y_pred,prob_class_0,prob_class_1,correct\n";

    for (Eigen::Index i = 0; i < X_eval.rows(); ++i) {
        static_cast<void>(X_eval);

        const bool correct = predictions(i) == y_eval(i);

        file << i << ","
             << y_eval(i) << ","
             << predictions(i) << ","
             << probabilities(i, 0) << ","
             << probabilities(i, 1) << ","
             << (correct ? "true" : "false") << "\n";
    }
}

// -----------------------------------------------------------------------------
// Random Forest experiment export tests
// -----------------------------------------------------------------------------

void test_experiment_exports_random_forest_comparison() {
    ensure_phase6b_output_dir_exists();

    const ml::Matrix X_train = make_random_forest_experiment_X();
    const ml::Vector y_train = make_random_forest_experiment_y();
    const ml::Matrix X_eval = make_random_forest_experiment_X_eval();
    const ml::Vector y_eval = make_random_forest_experiment_y_eval();

    std::vector<RandomForestExperimentResult> results;

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 3;
        options.random_seed = 42;

        results.push_back(run_single_tree_experiment(
            "single_tree_vs_random_forest",
            "single_tree",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::RandomForestOptions options;
        options.n_estimators = 9;
        options.bootstrap = true;
        options.random_seed = 42;
        options.tree_options.max_depth = 3;

        results.push_back(run_random_forest_experiment(
            "single_tree_vs_random_forest",
            "random_forest_9_trees",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    for (std::size_t n_estimators : {1, 5, 15}) {
        ml::RandomForestOptions options;
        options.n_estimators = n_estimators;
        options.bootstrap = true;
        options.random_seed = 42;
        options.tree_options.max_depth = 3;

        results.push_back(run_random_forest_experiment(
            "number_of_trees",
            "n_estimators_" + std::to_string(n_estimators),
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::RandomForestOptions options;
        options.n_estimators = 11;
        options.bootstrap = true;
        options.random_seed = 42;
        options.max_features = std::nullopt;
        options.tree_options.max_depth = 3;

        results.push_back(run_random_forest_experiment(
            "max_features",
            "all_features",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::RandomForestOptions options;
        options.n_estimators = 11;
        options.bootstrap = true;
        options.random_seed = 42;
        options.max_features = 2;
        options.tree_options.max_depth = 3;

        results.push_back(run_random_forest_experiment(
            "max_features",
            "max_features_2",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::RandomForestOptions options;
        options.n_estimators = 11;
        options.bootstrap = true;
        options.random_seed = 42;
        options.max_features = 2;
        options.tree_options.max_depth = 3;

        results.push_back(run_random_forest_experiment(
            "bootstrap_vs_no_bootstrap",
            "bootstrap_true",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::RandomForestOptions options;
        options.n_estimators = 11;
        options.bootstrap = false;
        options.random_seed = 42;
        options.max_features = 2;
        options.tree_options.max_depth = 3;

        results.push_back(run_random_forest_experiment(
            "bootstrap_vs_no_bootstrap",
            "bootstrap_false",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    const std::string comparison_csv_path =
        k_phase6b_output_dir + "/random_forest_comparison.csv";

    const std::string comparison_txt_path =
        k_phase6b_output_dir + "/random_forest_comparison.txt";

    export_random_forest_results_csv(
        results,
        comparison_csv_path
    );

    export_random_forest_results_txt(
        results,
        comparison_txt_path
    );

    {
        ml::RandomForestOptions options;
        options.n_estimators = 11;
        options.bootstrap = true;
        options.random_seed = 42;
        options.max_features = 2;
        options.tree_options.max_depth = 3;

        ml::RandomForestClassifier forest(options);
        forest.fit(X_train, y_train);

        const ml::Vector predictions = forest.predict(X_eval);
        const ml::Matrix probabilities = forest.predict_proba(X_eval);

        export_random_forest_prediction_probabilities_csv(
            X_eval,
            y_eval,
            predictions,
            probabilities,
            k_phase6b_output_dir + "/random_forest_prediction_probabilities.csv"
        );
    }

    if (!std::filesystem::exists(comparison_csv_path)) {
        throw std::runtime_error(
            "expected random_forest_comparison.csv to exist"
        );
    }

    if (!std::filesystem::exists(comparison_txt_path)) {
        throw std::runtime_error(
            "expected random_forest_comparison.txt to exist"
        );
    }

    if (!std::filesystem::exists(
            k_phase6b_output_dir + "/random_forest_prediction_probabilities.csv"
        )) {
        throw std::runtime_error(
            "expected random_forest_prediction_probabilities.csv to exist"
        );
    }
}

// -----------------------------------------------------------------------------
// GradientBoostingRegressor shallow-tree tests
// -----------------------------------------------------------------------------

ml::Matrix make_gradient_boosting_test_X() {
    ml::Matrix X(8, 2);
    X << 0.0, 0.0,
         1.0, 0.0,
         2.0, 0.0,
         3.0, 0.0,
         4.0, 1.0,
         5.0, 1.0,
         6.0, 1.0,
         7.0, 1.0;

    return X;
}

ml::Vector make_gradient_boosting_test_y() {
    return make_vector({
        1.0,
        2.0,
        3.0,
        4.0,
        8.0,
        9.0,
        10.0,
        11.0
    });
}

void test_regression_tree_options_accept_defaults() {
    const ml::RegressionTreeOptions options;

    ml::validate_regression_tree_options(
        options,
        "test_regression_tree_options_accept_defaults"
    );
}

void test_regression_tree_options_reject_invalid_max_depth() {
    ml::RegressionTreeOptions options;
    options.max_depth = 0;

    ml::validate_regression_tree_options(
        options,
        "test_regression_tree_options_reject_invalid_max_depth"
    );
}

void test_decision_tree_regressor_fits_simple_data() {
    const ml::Matrix X = make_gradient_boosting_test_X();
    const ml::Vector y = make_gradient_boosting_test_y();

    ml::RegressionTreeOptions options;
    options.max_depth = 3;

    ml::DecisionTreeRegressor tree(options);
    tree.fit(X, y);

    if (!tree.is_fitted()) {
        throw std::runtime_error("expected regression tree to be fitted");
    }

    const ml::Vector predictions = tree.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected regression tree predictions for each row");
    }
}

void test_decision_tree_regressor_rejects_predict_before_fit() {
    const ml::Matrix X = make_gradient_boosting_test_X();

    ml::DecisionTreeRegressor tree;

    static_cast<void>(tree.predict(X));
}

void test_gradient_boosting_options_accept_defaults() {
    const ml::GradientBoostingRegressorOptions options;

    ml::validate_gradient_boosting_regressor_options(
        options,
        "test_gradient_boosting_options_accept_defaults"
    );
}

void test_gradient_boosting_options_reject_zero_estimators() {
    ml::GradientBoostingRegressorOptions options;
    options.n_estimators = 0;

    ml::validate_gradient_boosting_regressor_options(
        options,
        "test_gradient_boosting_options_reject_zero_estimators"
    );
}

void test_gradient_boosting_options_reject_invalid_learning_rate() {
    ml::GradientBoostingRegressorOptions options;
    options.learning_rate = 0.0;

    ml::validate_gradient_boosting_regressor_options(
        options,
        "test_gradient_boosting_options_reject_invalid_learning_rate"
    );
}

void test_gradient_boosting_regressor_reports_not_fitted_initially() {
    const ml::GradientBoostingRegressor model;

    if (model.is_fitted()) {
        throw std::runtime_error("expected GradientBoostingRegressor to start unfitted");
    }
}

void test_gradient_boosting_regressor_fits_with_shallow_trees() {
    const ml::Matrix X = make_gradient_boosting_test_X();
    const ml::Vector y = make_gradient_boosting_test_y();

    ml::GradientBoostingRegressorOptions options;
    options.n_estimators = 5;
    options.max_depth = 1;
    options.learning_rate = 0.5;

    ml::GradientBoostingRegressor model(options);
    model.fit(X, y);

    if (!model.is_fitted()) {
        throw std::runtime_error("expected GradientBoostingRegressor to be fitted");
    }

    if (model.num_trees() != 5) {
        throw std::runtime_error("expected GradientBoostingRegressor to store 5 trees");
    }
}

void test_gradient_boosting_regressor_predict_returns_expected_shape() {
    const ml::Matrix X = make_gradient_boosting_test_X();
    const ml::Vector y = make_gradient_boosting_test_y();

    ml::GradientBoostingRegressorOptions options;
    options.n_estimators = 5;
    options.max_depth = 1;
    options.learning_rate = 0.5;

    ml::GradientBoostingRegressor model(options);
    model.fit(X, y);

    const ml::Vector predictions = model.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected one prediction per sample");
    }
}

void test_gradient_boosting_regressor_improves_over_initial_mean() {
    const ml::Matrix X = make_gradient_boosting_test_X();
    const ml::Vector y = make_gradient_boosting_test_y();

    ml::GradientBoostingRegressorOptions options;
    options.n_estimators = 10;
    options.max_depth = 1;
    options.learning_rate = 0.5;

    ml::GradientBoostingRegressor model(options);
    model.fit(X, y);

    const double initial_prediction = model.initial_prediction();

    const ml::Vector baseline =
        ml::Vector::Constant(
            y.size(),
            initial_prediction
        );

    const ml::Vector predictions = model.predict(X);

    double baseline_mse = 0.0;
    double model_mse = 0.0;

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double baseline_residual = baseline(i) - y(i);
        const double model_residual = predictions(i) - y(i);

        baseline_mse += baseline_residual * baseline_residual;
        model_mse += model_residual * model_residual;
    }

    baseline_mse /= static_cast<double>(y.size());
    model_mse /= static_cast<double>(y.size());

    if (model_mse >= baseline_mse) {
        throw std::runtime_error("expected boosted model to improve over initial mean baseline");
    }
}

void test_gradient_boosting_regressor_stores_loss_history() {
    const ml::Matrix X = make_gradient_boosting_test_X();
    const ml::Vector y = make_gradient_boosting_test_y();

    ml::GradientBoostingRegressorOptions options;
    options.n_estimators = 6;
    options.max_depth = 1;
    options.learning_rate = 0.5;

    ml::GradientBoostingRegressor model(options);
    model.fit(X, y);

    if (model.training_loss_history().size() != 6) {
        throw std::runtime_error("expected one loss entry per estimator");
    }
}

void test_gradient_boosting_regressor_rejects_predict_before_fit() {
    const ml::Matrix X = make_gradient_boosting_test_X();

    ml::GradientBoostingRegressor model;

    static_cast<void>(model.predict(X));
}

void test_gradient_boosting_regressor_rejects_empty_fit_X() {
    const ml::Matrix X;
    const ml::Vector y = make_gradient_boosting_test_y();

    ml::GradientBoostingRegressor model;

    model.fit(X, y);
}

void test_gradient_boosting_regressor_rejects_empty_fit_y() {
    const ml::Matrix X = make_gradient_boosting_test_X();
    const ml::Vector y;

    ml::GradientBoostingRegressor model;

    model.fit(X, y);
}

void test_gradient_boosting_regressor_rejects_mismatched_fit_data() {
    ml::Matrix X(3, 1);
    X << 0.0,
         1.0,
         2.0;

    const ml::Vector y = make_vector({1.0, 2.0});

    ml::GradientBoostingRegressor model;

    model.fit(X, y);
}

void test_gradient_boosting_regressor_learning_rate_affects_predictions() {
    const ml::Matrix X = make_gradient_boosting_test_X();
    const ml::Vector y = make_gradient_boosting_test_y();

    ml::GradientBoostingRegressorOptions slow_options;
    slow_options.n_estimators = 3;
    slow_options.max_depth = 1;
    slow_options.learning_rate = 0.1;

    ml::GradientBoostingRegressorOptions fast_options;
    fast_options.n_estimators = 3;
    fast_options.max_depth = 1;
    fast_options.learning_rate = 0.8;

    ml::GradientBoostingRegressor slow_model(slow_options);
    ml::GradientBoostingRegressor fast_model(fast_options);

    slow_model.fit(X, y);
    fast_model.fit(X, y);

    const ml::Vector slow_predictions = slow_model.predict(X);
    const ml::Vector fast_predictions = fast_model.predict(X);

    double total_difference = 0.0;

    for (Eigen::Index i = 0; i < slow_predictions.size(); ++i) {
        total_difference += std::abs(
            slow_predictions(i) - fast_predictions(i)
        );
    }

    if (total_difference <= 1e-8) {
        throw std::runtime_error(
            "expected different learning rates to produce different predictions"
        );
    }

    if (
        slow_model.training_loss_history().empty() ||
        fast_model.training_loss_history().empty()
    ) {
        throw std::runtime_error(
            "expected both models to store training loss history"
        );
    }

    const double slow_final_loss =
        slow_model.training_loss_history().back();

    const double fast_final_loss =
        fast_model.training_loss_history().back();

    if (std::abs(slow_final_loss - fast_final_loss) <= 1e-8) {
        throw std::runtime_error(
            "expected different learning rates to produce different final losses"
        );
    }
}

void test_gradient_boosting_regressor_training_loss_decreases() {
    const ml::Matrix X = make_gradient_boosting_test_X();
    const ml::Vector y = make_gradient_boosting_test_y();

    ml::GradientBoostingRegressorOptions options;
    options.n_estimators = 8;
    options.max_depth = 1;
    options.learning_rate = 0.5;

    ml::GradientBoostingRegressor model(options);
    model.fit(X, y);

    const std::vector<double>& loss_history =
        model.training_loss_history();

    if (loss_history.size() != options.n_estimators) {
        throw std::runtime_error(
            "expected one staged loss value per estimator"
        );
    }

    if (loss_history.empty()) {
        throw std::runtime_error(
            "expected non-empty training loss history"
        );
    }

    for (double loss : loss_history) {
        if (!std::isfinite(loss) || loss < 0.0) {
            throw std::runtime_error(
                "expected finite non-negative staged training loss"
            );
        }
    }

    if (loss_history.back() >= loss_history.front()) {
        throw std::runtime_error(
            "expected final staged loss to be lower than first staged loss"
        );
    }
}

// -----------------------------------------------------------------------------
// Gradient Boosting experiment export helpers
// -----------------------------------------------------------------------------

struct GradientBoostingExperimentResult {
    std::string experiment_name;
    std::string variant_name;
    std::string model_type;

    std::size_t n_estimators{0};
    double learning_rate{0.0};
    std::size_t max_depth{0};

    double mse{0.0};
    double rmse{0.0};
    std::size_t num_predictions{0};
};

struct GradientBoostingLossHistoryRow {
    std::string experiment_name;
    std::string variant_name;
    std::size_t stage{0};
    double training_loss{0.0};
};

ml::Matrix make_gradient_boosting_experiment_X() {
    ml::Matrix X(12, 2);
    X << 0.0, 0.0,
         0.5, 0.0,
         1.0, 0.0,
         1.5, 0.0,
         2.0, 1.0,
         2.5, 1.0,
         3.0, 1.0,
         3.5, 1.0,
         4.0, 0.0,
         4.5, 0.0,
         5.0, 0.0,
         5.5, 0.0;

    return X;
}

ml::Vector make_gradient_boosting_experiment_y() {
    return make_vector({
        1.0,
        1.5,
        2.0,
        2.5,
        6.0,
        6.5,
        7.0,
        7.5,
        4.0,
        4.5,
        5.0,
        5.5
    });
}

ml::Matrix make_gradient_boosting_experiment_X_eval() {
    ml::Matrix X(6, 2);
    X << 0.25, 0.0,
         1.25, 0.0,
         2.25, 1.0,
         3.25, 1.0,
         4.25, 0.0,
         5.25, 0.0;

    return X;
}

ml::Vector make_gradient_boosting_experiment_y_eval() {
    return make_vector({
        1.25,
        2.25,
        6.25,
        7.25,
        4.25,
        5.25
    });
}

double mse_for_vectors(
    const ml::Vector& predictions,
    const ml::Vector& targets
) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("mse_for_vectors: vectors must have same size");
    }

    double total = 0.0;

    for (Eigen::Index i = 0; i < predictions.size(); ++i) {
        const double residual = predictions(i) - targets(i);
        total += residual * residual;
    }

    return total / static_cast<double>(predictions.size());
}

GradientBoostingExperimentResult run_single_regression_tree_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::RegressionTreeOptions& options
) {
    ml::DecisionTreeRegressor tree(options);
    tree.fit(X_train, y_train);

    const ml::Vector predictions = tree.predict(X_eval);

    const double mse = mse_for_vectors(
        predictions,
        y_eval
    );

    return GradientBoostingExperimentResult{
        experiment_name,
        variant_name,
        "DecisionTreeRegressor",
        1,
        0.0,
        options.max_depth,
        mse,
        std::sqrt(mse),
        static_cast<std::size_t>(predictions.size())
    };
}

GradientBoostingExperimentResult run_gradient_boosting_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::GradientBoostingRegressorOptions& options,
    std::vector<GradientBoostingLossHistoryRow>& loss_rows
) {
    ml::GradientBoostingRegressor model(options);
    model.fit(X_train, y_train);

    const ml::Vector predictions = model.predict(X_eval);

    const double mse = mse_for_vectors(
        predictions,
        y_eval
    );

    const std::vector<double>& loss_history =
        model.training_loss_history();

    for (std::size_t i = 0; i < loss_history.size(); ++i) {
        loss_rows.push_back(
            GradientBoostingLossHistoryRow{
                experiment_name,
                variant_name,
                i + 1,
                loss_history[i]
            }
        );
    }

    return GradientBoostingExperimentResult{
        experiment_name,
        variant_name,
        "GradientBoostingRegressor",
        options.n_estimators,
        options.learning_rate,
        options.max_depth,
        mse,
        std::sqrt(mse),
        static_cast<std::size_t>(predictions.size())
    };
}

void export_gradient_boosting_results_csv(
    const std::vector<GradientBoostingExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_gradient_boosting_results_csv: failed to open output file"
        );
    }

    file << "experiment_name,variant_name,model_type,n_estimators,"
         << "learning_rate,max_depth,mse,rmse,num_predictions\n";

    for (const GradientBoostingExperimentResult& result : results) {
        file << result.experiment_name << ","
             << result.variant_name << ","
             << result.model_type << ","
             << result.n_estimators << ","
             << result.learning_rate << ","
             << result.max_depth << ","
             << result.mse << ","
             << result.rmse << ","
             << result.num_predictions << "\n";
    }
}

void export_gradient_boosting_results_txt(
    const std::vector<GradientBoostingExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_gradient_boosting_results_txt: failed to open output file"
        );
    }

    file << "Gradient Boosting Comparison\n\n";

    file << "This experiment compares:\n"
         << "- number of estimators\n"
         << "- learning rate\n"
         << "- tree depth\n"
         << "- single regression tree vs boosted trees\n\n";

    for (const GradientBoostingExperimentResult& result : results) {
        file << "Experiment: " << result.experiment_name << "\n"
             << "Variant: " << result.variant_name << "\n"
             << "Model: " << result.model_type << "\n"
             << "n_estimators: " << result.n_estimators << "\n"
             << "learning_rate: " << result.learning_rate << "\n"
             << "max_depth: " << result.max_depth << "\n"
             << "MSE: " << result.mse << "\n"
             << "RMSE: " << result.rmse << "\n"
             << "num_predictions: " << result.num_predictions << "\n\n";
    }
}

void export_gradient_boosting_loss_history_csv(
    const std::vector<GradientBoostingLossHistoryRow>& rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_gradient_boosting_loss_history_csv: failed to open output file"
        );
    }

    file << "experiment_name,variant_name,stage,training_loss\n";

    for (const GradientBoostingLossHistoryRow& row : rows) {
        file << row.experiment_name << ","
             << row.variant_name << ","
             << row.stage << ","
             << row.training_loss << "\n";
    }
}

// -----------------------------------------------------------------------------
// Gradient Boosting experiment export tests
// -----------------------------------------------------------------------------

void test_experiment_exports_gradient_boosting_comparison() {
    ensure_phase6b_output_dir_exists();

    const ml::Matrix X_train = make_gradient_boosting_experiment_X();
    const ml::Vector y_train = make_gradient_boosting_experiment_y();
    const ml::Matrix X_eval = make_gradient_boosting_experiment_X_eval();
    const ml::Vector y_eval = make_gradient_boosting_experiment_y_eval();

    std::vector<GradientBoostingExperimentResult> results;
    std::vector<GradientBoostingLossHistoryRow> loss_rows;

    {
        ml::RegressionTreeOptions options;
        options.max_depth = 2;
        options.min_samples_split = 2;
        options.min_samples_leaf = 1;

        results.push_back(
            run_single_regression_tree_experiment(
                "single_tree_vs_boosted_trees",
                "single_regression_tree_depth_2",
                X_train,
                y_train,
                X_eval,
                y_eval,
                options
            )
        );
    }

    {
        ml::GradientBoostingRegressorOptions options;
        options.n_estimators = 10;
        options.learning_rate = 0.3;
        options.max_depth = 2;

        results.push_back(
            run_gradient_boosting_experiment(
                "single_tree_vs_boosted_trees",
                "gradient_boosting_10_trees",
                X_train,
                y_train,
                X_eval,
                y_eval,
                options,
                loss_rows
            )
        );
    }

    for (std::size_t n_estimators : {1, 5, 20}) {
        ml::GradientBoostingRegressorOptions options;
        options.n_estimators = n_estimators;
        options.learning_rate = 0.3;
        options.max_depth = 2;

        results.push_back(
            run_gradient_boosting_experiment(
                "number_of_estimators",
                "n_estimators_" + std::to_string(n_estimators),
                X_train,
                y_train,
                X_eval,
                y_eval,
                options,
                loss_rows
            )
        );
    }

    for (double learning_rate : {0.1, 0.3, 0.8}) {
        ml::GradientBoostingRegressorOptions options;
        options.n_estimators = 10;
        options.learning_rate = learning_rate;
        options.max_depth = 2;

        results.push_back(
            run_gradient_boosting_experiment(
                "learning_rate",
                "learning_rate_" + std::to_string(learning_rate),
                X_train,
                y_train,
                X_eval,
                y_eval,
                options,
                loss_rows
            )
        );
    }

    for (std::size_t max_depth : {1, 2, 3}) {
        ml::GradientBoostingRegressorOptions options;
        options.n_estimators = 10;
        options.learning_rate = 0.3;
        options.max_depth = max_depth;

        results.push_back(
            run_gradient_boosting_experiment(
                "tree_depth",
                "max_depth_" + std::to_string(max_depth),
                X_train,
                y_train,
                X_eval,
                y_eval,
                options,
                loss_rows
            )
        );
    }

    const std::string comparison_csv_path =
        k_phase6b_output_dir + "/gradient_boosting_comparison.csv";

    const std::string comparison_txt_path =
        k_phase6b_output_dir + "/gradient_boosting_comparison.txt";

    const std::string loss_history_csv_path =
        k_phase6b_output_dir + "/gradient_boosting_loss_history.csv";

    export_gradient_boosting_results_csv(
        results,
        comparison_csv_path
    );

    export_gradient_boosting_results_txt(
        results,
        comparison_txt_path
    );

    export_gradient_boosting_loss_history_csv(
        loss_rows,
        loss_history_csv_path
    );

    if (!std::filesystem::exists(comparison_csv_path)) {
        throw std::runtime_error(
            "expected gradient_boosting_comparison.csv to exist"
        );
    }

    if (!std::filesystem::exists(comparison_txt_path)) {
        throw std::runtime_error(
            "expected gradient_boosting_comparison.txt to exist"
        );
    }

    if (!std::filesystem::exists(loss_history_csv_path)) {
        throw std::runtime_error(
            "expected gradient_boosting_loss_history.csv to exist"
        );
    }

    if (results.empty()) {
        throw std::runtime_error(
            "expected gradient boosting experiment results"
        );
    }

    if (loss_rows.empty()) {
        throw std::runtime_error(
            "expected gradient boosting loss history rows"
        );
    }
}

// -----------------------------------------------------------------------------
// Test runners
// -----------------------------------------------------------------------------

void run_phase6b_advanced_option_tests() {
    std::cout << "\n[Phase 6B.1] Advanced tree option and missing-value tests\n\n";

    test::expect_no_throw(
        "DecisionTreeOptions accepts advanced defaults",
        test_decision_tree_options_accept_advanced_defaults
    );

    test::expect_no_throw(
        "DecisionTreeOptions accepts valid max_leaf_nodes",
        test_decision_tree_options_accept_valid_max_leaf_nodes
    );

    test::expect_invalid_argument(
        "DecisionTreeOptions rejects zero max_leaf_nodes",
        test_decision_tree_options_reject_zero_max_leaf_nodes
    );

    test::expect_no_throw(
        "DecisionTreeOptions accepts valid max_features",
        test_decision_tree_options_accept_valid_max_features
    );

    test::expect_invalid_argument(
        "DecisionTreeOptions rejects zero max_features",
        test_decision_tree_options_reject_zero_max_features
    );

    test::expect_no_throw(
        "DecisionTreeOptions stores balanced class weight flag",
        test_decision_tree_options_store_balanced_class_weight_flag
    );

    test::expect_no_throw(
        "DecisionTreeOptions stores random seed",
        test_decision_tree_options_store_random_seed
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects missing values during fit",
        test_decision_tree_rejects_missing_values_during_fit
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects missing values during predict",
        test_decision_tree_rejects_missing_values_during_predict
    );
}

void run_max_leaf_nodes_tests() {
    std::cout << "\n[Phase 6B.2] max_leaf_nodes best-first builder tests\n\n";

    test::expect_invalid_argument(
        "best-first builder rejects missing max_leaf_nodes",
        test_best_first_builder_rejects_missing_max_leaf_nodes
    );

    test::expect_no_throw(
        "best-first builder with one leaf creates root leaf",
        test_best_first_builder_with_one_leaf_creates_root_leaf
    );

    test::expect_no_throw(
        "best-first builder respects two-leaf limit",
        test_best_first_builder_respects_two_leaf_limit
    );

    test::expect_no_throw(
        "best-first builder respects three-leaf limit",
        test_best_first_builder_respects_three_leaf_limit
    );

    test::expect_no_throw(
        "best-first builder stops before limit when no valid split exists",
        test_best_first_builder_stops_before_limit_when_no_valid_split_exists
    );

    test::expect_no_throw(
        "DecisionTreeClassifier uses best-first when max_leaf_nodes is set",
        test_decision_tree_classifier_uses_best_first_when_max_leaf_nodes_is_set
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects zero max_leaf_nodes",
        test_decision_tree_classifier_rejects_zero_max_leaf_nodes
    );
}

void run_max_features_tests() {
    std::cout << "\n[Phase 6B.3] max_features feature-subsampling tests\n\n";

    test::expect_no_throw(
        "find_best_split accepts max_features equal to feature count",
        test_find_best_split_accepts_max_features_equal_to_feature_count
    );

    test::expect_invalid_argument(
        "find_best_split rejects max_features larger than feature count",
        test_find_best_split_rejects_max_features_larger_than_feature_count
    );

    test::expect_no_throw(
        "find_best_split with max_features one is deterministic for same seed",
        test_find_best_split_with_max_features_one_is_deterministic_for_same_seed
    );

    test::expect_no_throw(
        "DecisionTreeClassifier accepts max_features",
        test_decision_tree_classifier_accepts_max_features
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects max_features larger than feature count",
        test_decision_tree_classifier_rejects_max_features_larger_than_feature_count
    );

    test::expect_no_throw(
        "best-first builder accepts max_features",
        test_best_first_builder_accepts_max_features
    );
}

void run_balanced_class_weight_tests() {
    std::cout << "\n[Phase 6B.4] balanced class_weight tests\n\n";

    test::expect_no_throw(
        "balanced_class_weights computes expected values",
        test_balanced_class_weights_compute_expected_values
    );

    test::expect_no_throw(
        "sample_weights_from_class_weights creates expected vector",
        test_sample_weights_from_class_weights_creates_expected_vector
    );

    test::expect_no_throw(
        "weighted_gini_impurity uses sample weights",
        test_weighted_gini_impurity_uses_sample_weights
    );

    test::expect_no_throw(
        "weighted_entropy uses sample weights",
        test_weighted_entropy_uses_sample_weights
    );

    test::expect_no_throw(
        "DecisionTreeClassifier accepts balanced class_weight",
        test_decision_tree_classifier_accepts_balanced_class_weight
    );

    test::expect_no_throw(
        "DecisionTreeClassifier accepts balanced class_weight with best-first",
        test_decision_tree_classifier_accepts_balanced_class_weight_with_best_first
    );
}

void run_sample_weight_split_scoring_tests() {
    std::cout << "\n[Phase 6B.5] sample_weight impurity and split-scoring tests\n\n";

    test::expect_no_throw(
        "weighted_gini_impurity matches unweighted when weights are equal",
        test_weighted_gini_impurity_matches_unweighted_when_weights_are_equal
    );

    test::expect_no_throw(
        "weighted_entropy matches unweighted when weights are equal",
        test_weighted_entropy_matches_unweighted_when_weights_are_equal
    );

    test::expect_no_throw(
        "weighted_gini_impurity changes with unequal weights",
        test_weighted_gini_impurity_changes_with_unequal_weights
    );

    test::expect_no_throw(
        "weighted_entropy changes with unequal weights",
        test_weighted_entropy_changes_with_unequal_weights
    );

    test::expect_no_throw(
        "weighted_child_impurity_by_weight computes expected value",
        test_weighted_child_impurity_by_weight_computes_expected_value
    );

    test::expect_no_throw(
        "evaluate_candidate_threshold with sample_weight finds valid split",
        test_evaluate_candidate_threshold_with_sample_weight_finds_valid_split
    );

    test::expect_no_throw(
        "find_best_split with sample_weight returns valid split",
        test_find_best_split_with_sample_weight_returns_valid_split
    );

    test::expect_invalid_argument(
        "weighted_gini_impurity rejects mismatched sample_weight size",
        test_weighted_gini_impurity_rejects_mismatched_sample_weight_size
    );

    test::expect_invalid_argument(
        "weighted_gini_impurity rejects negative sample_weight",
        test_weighted_gini_impurity_rejects_negative_sample_weight
    );

    test::expect_invalid_argument(
        "weighted_gini_impurity rejects zero total sample_weight",
        test_weighted_gini_impurity_rejects_zero_total_sample_weight
    );

    test::expect_invalid_argument(
        "weighted_gini_impurity rejects non-finite sample_weight",
        test_weighted_gini_impurity_rejects_non_finite_sample_weight
    );

    test::expect_invalid_argument(
        "evaluate_candidate_threshold with sample_weight rejects mismatched weights",
        test_evaluate_candidate_threshold_with_sample_weight_rejects_mismatched_weights
    );

    test::expect_invalid_argument(
        "find_best_split with sample_weight rejects mismatched weights",
        test_find_best_split_with_sample_weight_rejects_mismatched_weights
    );
}

void run_advanced_tree_control_experiment_tests() {
    std::cout << "\n[Phase 6B.6] Advanced tree control experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports advanced tree controls comparison",
        test_experiment_exports_advanced_tree_controls_comparison
    );
}

void run_bootstrap_sampling_tests() {
    std::cout << "\n[Phase 6B.7] Bootstrap sampling utility tests\n\n";

    test::expect_no_throw(
        "make_bootstrap_sample preserves shape",
        test_make_bootstrap_sample_preserves_shape
    );

    test::expect_no_throw(
        "make_bootstrap_sample indices are in range",
        test_make_bootstrap_sample_indices_are_in_range
    );

    test::expect_no_throw(
        "make_bootstrap_sample rows match sampled indices",
        test_make_bootstrap_sample_rows_match_sampled_indices
    );

    test::expect_no_throw(
        "make_bootstrap_sample is reproducible with same seed",
        test_make_bootstrap_sample_is_reproducible_with_same_seed
    );

    test::expect_no_throw(
        "make_bootstrap_sample changes with different seed",
        test_make_bootstrap_sample_changes_with_different_seed
    );

    test::expect_no_throw(
        "make_bootstrap_sample tracks out-of-bag indices",
        test_make_bootstrap_sample_tracks_out_of_bag_indices
    );

    test::expect_no_throw(
        "make_full_sample preserves original data",
        test_make_full_sample_preserves_original_data
    );

    test::expect_invalid_argument(
        "make_bootstrap_sample rejects empty X",
        test_make_bootstrap_sample_rejects_empty_X
    );

    test::expect_invalid_argument(
        "make_bootstrap_sample rejects empty y",
        test_make_bootstrap_sample_rejects_empty_y
    );

    test::expect_invalid_argument(
        "make_bootstrap_sample rejects mismatched X/y",
        test_make_bootstrap_sample_rejects_mismatched_X_y
    );

    test::expect_invalid_argument(
        "make_full_sample rejects empty X",
        test_make_full_sample_rejects_empty_X
    );

    test::expect_invalid_argument(
        "make_full_sample rejects empty y",
        test_make_full_sample_rejects_empty_y
    );

    test::expect_invalid_argument(
        "make_full_sample rejects mismatched X/y",
        test_make_full_sample_rejects_mismatched_X_y
    );
}

void run_random_forest_classifier_tests() {
    std::cout << "\n[Phase 6B.8] RandomForestClassifier tests\n\n";

    test::expect_no_throw(
        "RandomForestOptions accepts defaults",
        test_random_forest_options_accept_defaults
    );

    test::expect_invalid_argument(
        "RandomForestOptions rejects zero estimators",
        test_random_forest_options_reject_zero_estimators
    );

    test::expect_invalid_argument(
        "RandomForestOptions rejects zero max_features",
        test_random_forest_options_reject_zero_max_features
    );

    test::expect_no_throw(
        "RandomForestClassifier reports not fitted initially",
        test_random_forest_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "RandomForestClassifier fit marks model as fitted",
        test_random_forest_fit_marks_model_as_fitted
    );

    test::expect_no_throw(
        "RandomForestClassifier predict returns expected shape",
        test_random_forest_predict_returns_expected_shape
    );

    test::expect_no_throw(
        "RandomForestClassifier predict_proba returns expected shape",
        test_random_forest_predict_proba_returns_expected_shape
    );

    test::expect_no_throw(
        "RandomForestClassifier predict_proba rows sum to one",
        test_random_forest_predict_proba_rows_sum_to_one
    );

    test::expect_no_throw(
        "RandomForestClassifier predict matches predict_proba argmax",
        test_random_forest_predict_matches_predict_proba_argmax
    );

    test::expect_no_throw(
        "RandomForestClassifier is reproducible with same seed",
        test_random_forest_is_reproducible_with_same_seed
    );

    test::expect_no_throw(
        "RandomForestClassifier supports bootstrap false",
        test_random_forest_supports_bootstrap_false
    );

    test::expect_no_throw(
        "RandomForestClassifier supports max_features",
        test_random_forest_supports_max_features
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects predict before fit",
        test_random_forest_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects predict_proba before fit",
        test_random_forest_rejects_predict_proba_before_fit
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects empty fit X",
        test_random_forest_rejects_empty_fit_X
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects empty fit y",
        test_random_forest_rejects_empty_fit_y
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects mismatched fit data",
        test_random_forest_rejects_mismatched_fit_data
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects invalid class labels",
        test_random_forest_rejects_invalid_class_labels
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects predict feature mismatch",
        test_random_forest_rejects_predict_feature_mismatch
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects predict_proba feature mismatch",
        test_random_forest_rejects_predict_proba_feature_mismatch
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects non-finite predict values",
        test_random_forest_rejects_non_finite_predict_values
    );

    test::expect_invalid_argument(
        "RandomForestClassifier rejects max_features larger than feature count",
        test_random_forest_rejects_max_features_larger_than_feature_count
    );
}

void run_random_forest_experiment_export_tests() {
    std::cout << "\n[Phase 6B.9] Random Forest experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports random forest comparison",
        test_experiment_exports_random_forest_comparison
    );
}

void run_gradient_boosting_regressor_tests() {
    std::cout << "\n[Phase 6B.10] GradientBoostingRegressor shallow-tree tests\n\n";

    test::expect_no_throw(
        "RegressionTreeOptions accepts defaults",
        test_regression_tree_options_accept_defaults
    );

    test::expect_invalid_argument(
        "RegressionTreeOptions rejects invalid max_depth",
        test_regression_tree_options_reject_invalid_max_depth
    );

    test::expect_no_throw(
        "DecisionTreeRegressor fits simple data",
        test_decision_tree_regressor_fits_simple_data
    );

    test::expect_invalid_argument(
        "DecisionTreeRegressor rejects predict before fit",
        test_decision_tree_regressor_rejects_predict_before_fit
    );

    test::expect_no_throw(
        "GradientBoostingRegressorOptions accepts defaults",
        test_gradient_boosting_options_accept_defaults
    );

    test::expect_invalid_argument(
        "GradientBoostingRegressorOptions rejects zero estimators",
        test_gradient_boosting_options_reject_zero_estimators
    );

    test::expect_invalid_argument(
        "GradientBoostingRegressorOptions rejects invalid learning rate",
        test_gradient_boosting_options_reject_invalid_learning_rate
    );

    test::expect_no_throw(
        "GradientBoostingRegressor reports not fitted initially",
        test_gradient_boosting_regressor_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "GradientBoostingRegressor fits with shallow trees",
        test_gradient_boosting_regressor_fits_with_shallow_trees
    );

    test::expect_no_throw(
        "GradientBoostingRegressor predict returns expected shape",
        test_gradient_boosting_regressor_predict_returns_expected_shape
    );

    test::expect_no_throw(
        "GradientBoostingRegressor improves over initial mean",
        test_gradient_boosting_regressor_improves_over_initial_mean
    );

    test::expect_no_throw(
        "GradientBoostingRegressor learning rate affects predictions",
        test_gradient_boosting_regressor_learning_rate_affects_predictions
    );

    test::expect_no_throw(
        "GradientBoostingRegressor stores loss history",
        test_gradient_boosting_regressor_stores_loss_history
    );

    test::expect_no_throw(
        "GradientBoostingRegressor training loss decreases",
        test_gradient_boosting_regressor_training_loss_decreases
    );

    test::expect_invalid_argument(
        "GradientBoostingRegressor rejects predict before fit",
        test_gradient_boosting_regressor_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "GradientBoostingRegressor rejects empty fit X",
        test_gradient_boosting_regressor_rejects_empty_fit_X
    );

    test::expect_invalid_argument(
        "GradientBoostingRegressor rejects empty fit y",
        test_gradient_boosting_regressor_rejects_empty_fit_y
    );

    test::expect_invalid_argument(
        "GradientBoostingRegressor rejects mismatched fit data",
        test_gradient_boosting_regressor_rejects_mismatched_fit_data
    );
}

void run_gradient_boosting_experiment_export_tests() {
    std::cout << "\n[Phase 6B.11] Gradient Boosting experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports gradient boosting comparison",
        test_experiment_exports_gradient_boosting_comparison
    );
}

}  // namespace

namespace ml::experiments {

void run_phase6b_trees_ensembles_sanity() {
    run_phase6b_advanced_option_tests();
    run_max_leaf_nodes_tests();
    run_max_features_tests();
    run_balanced_class_weight_tests();
    run_sample_weight_split_scoring_tests();
    run_advanced_tree_control_experiment_tests();
    run_bootstrap_sampling_tests();
    run_random_forest_classifier_tests();
    run_random_forest_experiment_export_tests();
    run_gradient_boosting_regressor_tests();
    run_gradient_boosting_experiment_export_tests();
}

}  // namespace ml::experiments