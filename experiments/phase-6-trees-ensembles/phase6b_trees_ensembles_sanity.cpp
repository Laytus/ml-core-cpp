#include "phase6b_trees_ensembles_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/types.hpp"
#include "ml/trees/best_first_tree_builder.hpp"
#include "ml/trees/decision_tree.hpp"
#include "ml/trees/tree_node.hpp"
#include "ml/trees/class_weights.hpp"
#include "ml/trees/split_scoring.hpp"
#include "ml/common/classification_metrics.hpp"

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

}  // namespace

namespace ml::experiments {

void run_phase6b_trees_ensembles_sanity() {
    run_phase6b_advanced_option_tests();
    run_max_leaf_nodes_tests();
    run_max_features_tests();
    run_balanced_class_weight_tests();
    run_sample_weight_split_scoring_tests();
    run_advanced_tree_control_experiment_tests();
}

}  // namespace ml::experiments