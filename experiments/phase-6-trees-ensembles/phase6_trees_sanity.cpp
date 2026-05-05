#include "phase6_trees_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/types.hpp"
#include "ml/trees/split_scoring.hpp"
#include "ml/trees/tree_builder.hpp"
#include "ml/trees/decision_tree.hpp"
#include "ml/common/classification_metrics.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

ml::Matrix make_tree_experiment_X() {
    ml::Matrix X(12, 2);
    X << 0.0, 0.0,
         0.5, 0.1,
         1.0, 0.0,
         1.5, 0.2,
         2.0, 1.0,
         2.5, 1.1,
         3.0, 1.0,
         3.5, 1.2,
         4.0, 0.0,
         4.5, 0.1,
         5.0, 0.0,
         5.5, 0.2;

    return X;
}

ml::Vector make_tree_experiment_y() {
    return make_vector({
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0
    });
}

ml::Matrix make_tree_experiment_X_eval() {
    ml::Matrix X(6, 2);
    X << 0.25, 0.0,
         1.25, 0.1,
         2.25, 1.0,
         3.25, 1.1,
         4.25, 0.0,
         5.25, 0.1;

    return X;
}

ml::Vector make_tree_experiment_y_eval() {
    return make_vector({
        0.0, 0.0,
        1.0, 1.0,
        0.0, 0.0
    });
}

// -----------------------------------------------------------------------------
// Gini impurity tests
// -----------------------------------------------------------------------------

void test_gini_impurity_returns_zero_for_pure_node() {
    const ml::Vector y = make_vector({1.0, 1.0, 1.0, 1.0});

    test::assert_almost_equal(
        ml::gini_impurity(y),
        0.0,
        "test_gini_impurity_returns_zero_for_pure_node"
    );
}

void test_gini_impurity_computes_binary_mixed_node() {
    const ml::Vector y = make_vector({0.0, 0.0, 1.0, 1.0});

    test::assert_almost_equal(
        ml::gini_impurity(y),
        0.5,
        "test_gini_impurity_computes_binary_mixed_node"
    );
}

void test_gini_impurity_computes_multiclass_node() {
    const ml::Vector y = make_vector({0.0, 1.0, 2.0});

    const double expected = 1.0 - (
        (1.0 / 3.0) * (1.0 / 3.0) +
        (1.0 / 3.0) * (1.0 / 3.0) +
        (1.0 / 3.0) * (1.0 / 3.0)
    );

    test::assert_almost_equal(
        ml::gini_impurity(y),
        expected,
        "test_gini_impurity_computes_multiclass_node"
    );
}

void test_gini_impurity_rejects_empty_targets() {
    const ml::Vector y;

    static_cast<void>(ml::gini_impurity(y));
}

void test_gini_impurity_rejects_non_integer_labels() {
    const ml::Vector y = make_vector({0.0, 1.5, 1.0});

    static_cast<void>(ml::gini_impurity(y));
}

void test_gini_impurity_rejects_negative_labels() {
    const ml::Vector y = make_vector({0.0, -1.0, 1.0});

    static_cast<void>(ml::gini_impurity(y));
}

// -----------------------------------------------------------------------------
// Entropy tests
// -----------------------------------------------------------------------------

void test_entropy_returns_zero_for_pure_node() {
    const ml::Vector y = make_vector({2.0, 2.0, 2.0, 2.0});

    test::assert_almost_equal(
        ml::entropy(y),
        0.0,
        "test_entropy_returns_zero_for_pure_node"
    );
}

void test_entropy_computes_binary_mixed_node() {
    const ml::Vector y = make_vector({0.0, 0.0, 1.0, 1.0});

    test::assert_almost_equal(
        ml::entropy(y),
        1.0,
        "test_entropy_computes_binary_mixed_node"
    );
}

void test_entropy_computes_unbalanced_binary_node() {
    const ml::Vector y = make_vector({0.0, 0.0, 0.0, 1.0});

    const double p0 = 3.0 / 4.0;
    const double p1 = 1.0 / 4.0;

    const double expected =
        -p0 * std::log2(p0) -
        p1 * std::log2(p1);

    test::assert_almost_equal(
        ml::entropy(y),
        expected,
        "test_entropy_computes_unbalanced_binary_node"
    );
}

void test_entropy_rejects_empty_targets() {
    const ml::Vector y;

    static_cast<void>(ml::entropy(y));
}

void test_entropy_rejects_non_integer_labels() {
    const ml::Vector y = make_vector({0.0, 0.25, 1.0});

    static_cast<void>(ml::entropy(y));
}

void test_entropy_rejects_negative_labels() {
    const ml::Vector y = make_vector({0.0, -1.0, 1.0});

    static_cast<void>(ml::entropy(y));
}

// -----------------------------------------------------------------------------
// Weighted child impurity tests
// -----------------------------------------------------------------------------

void test_weighted_child_impurity_computes_expected_value() {
    const double result = ml::weighted_child_impurity(
        0.0,
        0.5,
        2,
        2
    );

    test::assert_almost_equal(
        result,
        0.25,
        "test_weighted_child_impurity_computes_expected_value"
    );
}

void test_weighted_child_impurity_handles_uneven_child_sizes() {
    const double result = ml::weighted_child_impurity(
        0.0,
        0.5,
        3,
        1
    );

    const double expected =
        (3.0 / 4.0) * 0.0 +
        (1.0 / 4.0) * 0.5;

    test::assert_almost_equal(
        result,
        expected,
        "test_weighted_child_impurity_handles_uneven_child_sizes"
    );
}

void test_weighted_child_impurity_rejects_zero_left_count() {
    static_cast<void>(ml::weighted_child_impurity(
        0.0,
        0.5,
        0,
        2
    ));
}

void test_weighted_child_impurity_rejects_zero_right_count() {
    static_cast<void>(ml::weighted_child_impurity(
        0.0,
        0.5,
        2,
        0
    ));
}

void test_weighted_child_impurity_rejects_negative_left_impurity() {
    static_cast<void>(ml::weighted_child_impurity(
        -0.1,
        0.5,
        2,
        2
    ));
}

void test_weighted_child_impurity_rejects_negative_right_impurity() {
    static_cast<void>(ml::weighted_child_impurity(
        0.1,
        -0.5,
        2,
        2
    ));
}

void test_weighted_child_impurity_rejects_non_finite_impurity() {
    static_cast<void>(ml::weighted_child_impurity(
        std::numeric_limits<double>::infinity(),
        0.5,
        2,
        2
    ));
}

// -----------------------------------------------------------------------------
// Impurity reduction tests
// -----------------------------------------------------------------------------

void test_impurity_reduction_computes_expected_value() {
    const double result = ml::impurity_reduction(
        0.5,
        0.25
    );

    test::assert_almost_equal(
        result,
        0.25,
        "test_impurity_reduction_computes_expected_value"
    );
}

void test_impurity_reduction_allows_zero_improvement() {
    const double result = ml::impurity_reduction(
        0.5,
        0.5
    );

    test::assert_almost_equal(
        result,
        0.0,
        "test_impurity_reduction_allows_zero_improvement"
    );
}

void test_impurity_reduction_allows_negative_improvement() {
    const double result = ml::impurity_reduction(
        0.25,
        0.5
    );

    test::assert_almost_equal(
        result,
        -0.25,
        "test_impurity_reduction_allows_negative_improvement"
    );
}

void test_impurity_reduction_rejects_negative_parent_impurity() {
    static_cast<void>(ml::impurity_reduction(
        -0.1,
        0.2
    ));
}

void test_impurity_reduction_rejects_negative_child_impurity() {
    static_cast<void>(ml::impurity_reduction(
        0.5,
        -0.1
    ));
}

void test_impurity_reduction_rejects_non_finite_parent_impurity() {
    static_cast<void>(ml::impurity_reduction(
        std::numeric_limits<double>::infinity(),
        0.1
    ));
}

void test_impurity_reduction_rejects_non_finite_child_impurity() {
    static_cast<void>(ml::impurity_reduction(
        0.5,
        std::numeric_limits<double>::quiet_NaN()
    ));
}

// -----------------------------------------------------------------------------
// Candidate threshold evaluation tests
// -----------------------------------------------------------------------------

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

void test_evaluate_candidate_threshold_finds_valid_split() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    const ml::SplitCandidate candidate = ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        2.5
    );

    if (!candidate.valid) {
        throw std::runtime_error("expected candidate split to be valid");
    }

    if (candidate.feature_index != 0) {
        throw std::runtime_error("expected feature_index 0");
    }

    test::assert_almost_equal(
        candidate.threshold,
        2.5,
        "test_evaluate_candidate_threshold_finds_valid_split threshold"
    );

    if (candidate.left_count != 3 || candidate.right_count != 3) {
        throw std::runtime_error("expected 3 samples on each side");
    }

    test::assert_almost_equal(
        candidate.parent_impurity,
        0.5,
        "test_evaluate_candidate_threshold_finds_valid_split parent_impurity"
    );

    test::assert_almost_equal(
        candidate.left_impurity,
        0.0,
        "test_evaluate_candidate_threshold_finds_valid_split left_impurity"
    );

    test::assert_almost_equal(
        candidate.right_impurity,
        0.0,
        "test_evaluate_candidate_threshold_finds_valid_split right_impurity"
    );

    test::assert_almost_equal(
        candidate.weighted_child_impurity,
        0.0,
        "test_evaluate_candidate_threshold_finds_valid_split weighted_child_impurity"
    );

    test::assert_almost_equal(
        candidate.impurity_decrease,
        0.5,
        "test_evaluate_candidate_threshold_finds_valid_split impurity_decrease"
    );
}

void test_evaluate_candidate_threshold_returns_invalid_for_empty_left_child() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    const ml::SplitCandidate candidate = ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        -1.0
    );

    if (candidate.valid) {
        throw std::runtime_error("expected candidate split to be invalid");
    }

    if (candidate.left_count != 0 || candidate.right_count != 6) {
        throw std::runtime_error("expected empty left child");
    }
}

void test_evaluate_candidate_threshold_returns_invalid_for_empty_right_child() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    const ml::SplitCandidate candidate = ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        10.0
    );

    if (candidate.valid) {
        throw std::runtime_error("expected candidate split to be invalid");
    }

    if (candidate.left_count != 6 || candidate.right_count != 0) {
        throw std::runtime_error("expected empty right child");
    }
}

void test_evaluate_candidate_threshold_respects_min_samples_leaf() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_samples_leaf = 4;

    const ml::SplitCandidate candidate = ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        2.5,
        options
    );

    if (candidate.valid) {
        throw std::runtime_error("expected split to be invalid due to min_samples_leaf");
    }

    if (candidate.left_count != 3 || candidate.right_count != 3) {
        throw std::runtime_error("expected split counts to still be recorded");
    }
}

void test_evaluate_candidate_threshold_respects_min_impurity_decrease() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_impurity_decrease = 0.6;

    const ml::SplitCandidate candidate = ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        2.5,
        options
    );

    if (candidate.valid) {
        throw std::runtime_error("expected split to be invalid due to min_impurity_decrease");
    }

    test::assert_almost_equal(
        candidate.impurity_decrease,
        0.5,
        "test_evaluate_candidate_threshold_respects_min_impurity_decrease"
    );
}

void test_evaluate_candidate_threshold_rejects_empty_X() {
    const ml::Matrix X;
    const ml::Vector y = make_simple_tree_y();

    static_cast<void>(ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        1.0
    ));
}

void test_evaluate_candidate_threshold_rejects_empty_y() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y;

    static_cast<void>(ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        1.0
    ));
}

void test_evaluate_candidate_threshold_rejects_mismatched_X_y() {
    ml::Matrix X(3, 1);
    X << 0.0,
         1.0,
         2.0;

    const ml::Vector y = make_vector({0.0, 1.0});

    static_cast<void>(ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        1.0
    ));
}

void test_evaluate_candidate_threshold_rejects_invalid_feature_index() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    static_cast<void>(ml::evaluate_candidate_threshold(
        X,
        y,
        5,
        1.0
    ));
}

void test_evaluate_candidate_threshold_rejects_non_finite_threshold() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    static_cast<void>(ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        std::numeric_limits<double>::infinity()
    ));
}

void test_evaluate_candidate_threshold_rejects_invalid_options() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_samples_leaf = 0;

    static_cast<void>(ml::evaluate_candidate_threshold(
        X,
        y,
        0,
        2.5,
        options
    ));
}

// -----------------------------------------------------------------------------
// Best split selection tests
// -----------------------------------------------------------------------------

void test_find_best_split_returns_best_feature_and_threshold() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    const ml::SplitCandidate split = ml::find_best_split(X, y);

    if (!split.valid) {
        throw std::runtime_error("expected best split to be valid");
    }

    if (split.feature_index != 0) {
        throw std::runtime_error("expected best split to use feature 0");
    }

    test::assert_almost_equal(
        split.threshold,
        2.5,
        "test_find_best_split_returns_best_feature_and_threshold threshold"
    );

    test::assert_almost_equal(
        split.impurity_decrease,
        0.5,
        "test_find_best_split_returns_best_feature_and_threshold impurity_decrease"
    );
}

void test_find_best_split_returns_invalid_when_all_features_constant() {
    ml::Matrix X(4, 2);
    X << 1.0, 5.0,
         1.0, 5.0,
         1.0, 5.0,
         1.0, 5.0;

    const ml::Vector y = make_vector({0.0, 0.0, 1.0, 1.0});

    const ml::SplitCandidate split = ml::find_best_split(X, y);

    if (split.valid) {
        throw std::runtime_error("expected no valid split for constant features");
    }
}

void test_find_best_split_respects_min_samples_leaf() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_samples_leaf = 4;

    const ml::SplitCandidate split = ml::find_best_split(X, y, options);

    if (split.valid) {
        throw std::runtime_error("expected no valid split due to min_samples_leaf");
    }
}

void test_find_best_split_respects_min_impurity_decrease() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_impurity_decrease = 0.6;

    const ml::SplitCandidate split = ml::find_best_split(X, y, options);

    if (split.valid) {
        throw std::runtime_error("expected no valid split due to min_impurity_decrease");
    }
}

void test_find_best_split_uses_deterministic_feature_tie_breaking() {
    ml::Matrix X(6, 2);
    X << 0.0, 0.0,
         1.0, 1.0,
         2.0, 2.0,
         3.0, 3.0,
         4.0, 4.0,
         5.0, 5.0;

    const ml::Vector y = make_vector({0.0, 0.0, 0.0, 1.0, 1.0, 1.0});

    const ml::SplitCandidate split = ml::find_best_split(X, y);

    if (!split.valid) {
        throw std::runtime_error("expected valid split");
    }

    if (split.feature_index != 0) {
        throw std::runtime_error("expected tie-breaking to choose smaller feature index");
    }

    test::assert_almost_equal(
        split.threshold,
        2.5,
        "test_find_best_split_uses_deterministic_feature_tie_breaking threshold"
    );
}

void test_find_best_split_uses_deterministic_threshold_tie_breaking() {
    ml::Matrix X(4, 1);
    X << 0.0,
         1.0,
         2.0,
         3.0;

    const ml::Vector y = make_vector({0.0, 1.0, 0.0, 1.0});

    const ml::SplitCandidate split = ml::find_best_split(X, y);

    if (!split.valid) {
        throw std::runtime_error("expected valid split");
    }

    test::assert_almost_equal(
        split.threshold,
        0.5,
        "test_find_best_split_uses_deterministic_threshold_tie_breaking threshold"
    );
}

void test_find_best_split_rejects_empty_X() {
    const ml::Matrix X;
    const ml::Vector y = make_simple_tree_y();

    static_cast<void>(ml::find_best_split(X, y));
}

void test_find_best_split_rejects_empty_y() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y;

    static_cast<void>(ml::find_best_split(X, y));
}

void test_find_best_split_rejects_mismatched_X_y() {
    ml::Matrix X(3, 1);
    X << 0.0,
         1.0,
         2.0;

    const ml::Vector y = make_vector({0.0, 1.0});

    static_cast<void>(ml::find_best_split(X, y));
}

void test_find_best_split_rejects_non_finite_feature_value() {
    ml::Matrix X(3, 1);
    X << 0.0,
         std::numeric_limits<double>::infinity(),
         2.0;

    const ml::Vector y = make_vector({0.0, 1.0, 1.0});

    static_cast<void>(ml::find_best_split(X, y));
}

void test_find_best_split_rejects_invalid_options() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_impurity_decrease = -0.1;

    static_cast<void>(ml::find_best_split(X, y, options));
}

// -----------------------------------------------------------------------------
// Recursive tree growth tests
// -----------------------------------------------------------------------------

void test_majority_class_returns_most_common_label() {
    const ml::Vector y = make_vector({0.0, 1.0, 1.0, 2.0, 1.0});

    test::assert_almost_equal(
        ml::majority_class(y),
        1.0,
        "test_majority_class_returns_most_common_label"
    );
}

void test_majority_class_uses_smallest_label_on_tie() {
    const ml::Vector y = make_vector({2.0, 1.0, 2.0, 1.0});

    test::assert_almost_equal(
        ml::majority_class(y),
        1.0,
        "test_majority_class_uses_smallest_label_on_tie"
    );
}

void test_is_pure_node_detects_pure_node() {
    const ml::Vector y = make_vector({1.0, 1.0, 1.0});

    if (!ml::is_pure_node(y)) {
        throw std::runtime_error("expected node to be pure");
    }
}

void test_is_pure_node_detects_mixed_node() {
    const ml::Vector y = make_vector({1.0, 0.0, 1.0});

    if (ml::is_pure_node(y)) {
        throw std::runtime_error("expected node to be mixed");
    }
}

void test_split_dataset_splits_rows_correctly() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    const ml::DatasetSplit split = ml::split_dataset(
        X,
        y,
        0,
        2.5
    );

    if (split.X_left.rows() != 3 || split.X_right.rows() != 3) {
        throw std::runtime_error("expected 3 rows in each split side");
    }

    test::assert_almost_equal(
        split.y_left(0),
        0.0,
        "test_split_dataset_splits_rows_correctly y_left 0"
    );

    test::assert_almost_equal(
        split.y_right(0),
        1.0,
        "test_split_dataset_splits_rows_correctly y_right 0"
    );
}

void test_build_tree_creates_leaf_for_pure_node() {
    ml::Matrix X(3, 1);
    X << 0.0,
         1.0,
         2.0;

    const ml::Vector y = make_vector({1.0, 1.0, 1.0});

    const ml::DecisionTreeOptions options;
    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected pure node to become a leaf");
    }

    test::assert_almost_equal(
        root->prediction,
        1.0,
        "test_build_tree_creates_leaf_for_pure_node prediction"
    );

    if (root->num_samples != 3) {
        throw std::runtime_error("expected root to store sample count");
    }
}

void test_build_tree_creates_internal_root_for_separable_data() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 3;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (root->is_leaf) {
        throw std::runtime_error("expected root to be an internal node");
    }

    if (root->feature_index != 0) {
        throw std::runtime_error("expected root split to use feature 0");
    }

    test::assert_almost_equal(
        root->threshold,
        2.5,
        "test_build_tree_creates_internal_root_for_separable_data threshold"
    );

    if (!root->left || !root->right) {
        throw std::runtime_error("expected root to have left and right children");
    }

    if (!root->left->is_leaf || !root->right->is_leaf) {
        throw std::runtime_error("expected both children to be leaves for simple data");
    }

    test::assert_almost_equal(
        root->left->prediction,
        0.0,
        "test_build_tree_creates_internal_root_for_separable_data left prediction"
    );

    test::assert_almost_equal(
        root->right->prediction,
        1.0,
        "test_build_tree_creates_internal_root_for_separable_data right prediction"
    );
}

void test_build_tree_respects_max_depth() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 0;

    static_cast<void>(ml::build_tree(
        X,
        y,
        options
    ));
}

void test_build_tree_respects_min_samples_split() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_samples_split = 10;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected node to be leaf due to min_samples_split");
    }
}

void test_build_tree_creates_leaf_when_no_valid_split_exists() {
    ml::Matrix X(4, 1);
    X << 1.0,
         1.0,
         1.0,
         1.0;

    const ml::Vector y = make_vector({0.0, 0.0, 1.0, 1.0});

    const ml::DecisionTreeOptions options;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected leaf when no valid split exists");
    }

    test::assert_almost_equal(
        root->prediction,
        0.0,
        "test_build_tree_creates_leaf_when_no_valid_split_exists prediction"
    );
}

void test_build_tree_rejects_invalid_labels() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_vector({0.0, 0.0, 0.5, 1.0, 1.0, 1.0});

    const ml::DecisionTreeOptions options;

    static_cast<void>(ml::build_tree(
        X,
        y,
        options
    ));
}

// -----------------------------------------------------------------------------
// Stopping rule tests
// -----------------------------------------------------------------------------

ml::Matrix make_two_level_tree_X() {
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

ml::Vector make_two_level_tree_y() {
    return make_vector({0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0});
}

void test_build_tree_stops_when_node_is_pure() {
    ml::Matrix X(4, 1);
    X << 0.0,
         1.0,
         2.0,
         3.0;

    const ml::Vector y = make_vector({1.0, 1.0, 1.0, 1.0});

    ml::DecisionTreeOptions options;
    options.max_depth = 5;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected pure node stopping rule to create leaf");
    }

    test::assert_almost_equal(
        root->impurity,
        0.0,
        "test_build_tree_stops_when_node_is_pure impurity"
    );

    test::assert_almost_equal(
        root->prediction,
        1.0,
        "test_build_tree_stops_when_node_is_pure prediction"
    );
}

void test_build_tree_stops_at_max_depth_zero_rejected_by_options() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 0;

    static_cast<void>(ml::build_tree(
        X,
        y,
        options
    ));
}

void test_build_tree_stops_at_max_depth_one() {
    const ml::Matrix X = make_two_level_tree_X();
    const ml::Vector y = make_two_level_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 1;
    options.min_samples_leaf = 1;
    options.min_samples_split = 2;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (root->is_leaf) {
        throw std::runtime_error("expected root to split at depth 0");
    }

    if (!root->left || !root->right) {
        throw std::runtime_error("expected root children");
    }

    if (!root->left->is_leaf || !root->right->is_leaf) {
        throw std::runtime_error("expected children to stop at max_depth = 1");
    }

    if (root->left->left || root->left->right || root->right->left || root->right->right) {
        throw std::runtime_error("expected no grandchildren when max_depth = 1");
    }
}

void test_build_tree_allows_deeper_tree_when_max_depth_allows() {
    const ml::Matrix X = make_two_level_tree_X();
    const ml::Vector y = make_two_level_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 3;
    options.min_samples_leaf = 1;
    options.min_samples_split = 2;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (root->is_leaf) {
        throw std::runtime_error("expected root to be internal");
    }

    const bool has_grandchild =
        (root->left && (!root->left->is_leaf)) ||
        (root->right && (!root->right->is_leaf));

    if (!has_grandchild) {
        throw std::runtime_error("expected deeper tree when max_depth allows it");
    }
}

void test_build_tree_stops_when_min_samples_split_is_not_met() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 5;
    options.min_samples_split = 7;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected min_samples_split stopping rule to create leaf");
    }

    if (root->num_samples != 6) {
        throw std::runtime_error("expected root to store all samples");
    }
}

void test_build_tree_allows_split_when_min_samples_split_is_met() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 5;
    options.min_samples_split = 6;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (root->is_leaf) {
        throw std::runtime_error("expected split when min_samples_split is met");
    }
}

void test_build_tree_stops_when_min_samples_leaf_blocks_all_splits() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 5;
    options.min_samples_leaf = 4;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected min_samples_leaf to block all valid splits");
    }
}

void test_build_tree_allows_split_when_min_samples_leaf_is_satisfied() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 5;
    options.min_samples_leaf = 3;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (root->is_leaf) {
        throw std::runtime_error("expected split when min_samples_leaf is satisfied");
    }

    if (!root->left || !root->right) {
        throw std::runtime_error("expected children after valid split");
    }

    if (root->left->num_samples != 3 || root->right->num_samples != 3) {
        throw std::runtime_error("expected both leaves to satisfy min_samples_leaf");
    }
}

void test_build_tree_stops_when_min_impurity_decrease_blocks_best_split() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 5;
    options.min_impurity_decrease = 0.6;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected min_impurity_decrease to block split");
    }
}

void test_build_tree_allows_split_when_min_impurity_decrease_is_satisfied() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 5;
    options.min_impurity_decrease = 0.5;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (root->is_leaf) {
        throw std::runtime_error("expected split when impurity decrease threshold is satisfied");
    }

    test::assert_almost_equal(
        root->impurity_decrease,
        0.5,
        "test_build_tree_allows_split_when_min_impurity_decrease_is_satisfied impurity_decrease"
    );
}

void test_build_tree_stops_when_no_valid_split_exists_due_to_constant_features() {
    ml::Matrix X(4, 2);
    X << 1.0, 2.0,
         1.0, 2.0,
         1.0, 2.0,
         1.0, 2.0;

    const ml::Vector y = make_vector({0.0, 0.0, 1.0, 1.0});

    ml::DecisionTreeOptions options;
    options.max_depth = 5;

    const std::unique_ptr<ml::TreeNode> root = ml::build_tree(
        X,
        y,
        options
    );

    if (!root->is_leaf) {
        throw std::runtime_error("expected leaf when all features are constant");
    }

    test::assert_almost_equal(
        root->prediction,
        0.0,
        "test_build_tree_stops_when_no_valid_split_exists_due_to_constant_features prediction"
    );
}

void test_build_tree_rejects_invalid_min_samples_split_option() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_samples_split = 1;

    static_cast<void>(ml::build_tree(
        X,
        y,
        options
    ));
}

void test_build_tree_rejects_invalid_min_samples_leaf_option() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_samples_leaf = 0;

    static_cast<void>(ml::build_tree(
        X,
        y,
        options
    ));
}

void test_build_tree_rejects_invalid_min_impurity_decrease_option() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.min_impurity_decrease = -1e-3;

    static_cast<void>(ml::build_tree(
        X,
        y,
        options
    ));
}

// -----------------------------------------------------------------------------
// Decision tree prediction traversal tests
// -----------------------------------------------------------------------------

void test_decision_tree_classifier_reports_not_fitted_initially() {
    const ml::DecisionTreeClassifier tree;

    if (tree.is_fitted()) {
        throw std::runtime_error("expected DecisionTreeClassifier to start unfitted");
    }
}

void test_decision_tree_classifier_fit_marks_model_as_fitted() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeClassifier tree;
    tree.fit(X, y);

    if (!tree.is_fitted()) {
        throw std::runtime_error("expected DecisionTreeClassifier to be fitted after fit");
    }
}

void test_decision_tree_classifier_predicts_training_data() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 3;

    ml::DecisionTreeClassifier tree(options);
    tree.fit(X, y);

    const ml::Vector predictions = tree.predict(X);

    test::assert_vector_almost_equal(
        predictions,
        y,
        "test_decision_tree_classifier_predicts_training_data"
    );
}

void test_decision_tree_classifier_predicts_new_samples_by_traversal() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 3;

    ml::DecisionTreeClassifier tree(options);
    tree.fit(X, y);

    ml::Matrix X_new(4, 2);
    X_new << 0.5, 0.0,
             2.0, 0.0,
             3.0, 1.0,
             5.5, 1.0;

    const ml::Vector predictions = tree.predict(X_new);

    const ml::Vector expected = make_vector({
        0.0,
        0.0,
        1.0,
        1.0
    });

    test::assert_vector_almost_equal(
        predictions,
        expected,
        "test_decision_tree_classifier_predicts_new_samples_by_traversal"
    );
}

void test_decision_tree_classifier_predicts_majority_leaf_when_stopped_early() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeOptions options;
    options.max_depth = 1;
    options.min_samples_split = 10;

    ml::DecisionTreeClassifier tree(options);
    tree.fit(X, y);

    const ml::Vector predictions = tree.predict(X);

    const ml::Vector expected = make_vector({
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    });

    test::assert_vector_almost_equal(
        predictions,
        expected,
        "test_decision_tree_classifier_predicts_majority_leaf_when_stopped_early"
    );
}

void test_decision_tree_classifier_rejects_predict_before_fit() {
    const ml::DecisionTreeClassifier tree;
    const ml::Matrix X = make_simple_tree_X();

    static_cast<void>(tree.predict(X));
}

void test_decision_tree_classifier_rejects_empty_predict_matrix() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeClassifier tree;
    tree.fit(X, y);

    const ml::Matrix X_empty;

    static_cast<void>(tree.predict(X_empty));
}

void test_decision_tree_classifier_rejects_predict_feature_mismatch() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeClassifier tree;
    tree.fit(X, y);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(tree.predict(X_bad));
}

void test_decision_tree_classifier_rejects_empty_fit_matrix() {
    const ml::Matrix X;
    const ml::Vector y = make_simple_tree_y();

    ml::DecisionTreeClassifier tree;

    tree.fit(X, y);
}

void test_decision_tree_classifier_rejects_empty_fit_targets() {
    const ml::Matrix X = make_simple_tree_X();
    const ml::Vector y;

    ml::DecisionTreeClassifier tree;

    tree.fit(X, y);
}

void test_decision_tree_classifier_rejects_mismatched_fit_data() {
    ml::Matrix X(3, 1);
    X << 0.0,
         1.0,
         2.0;

    const ml::Vector y = make_vector({0.0, 1.0});

    ml::DecisionTreeClassifier tree;

    tree.fit(X, y);
}

void test_decision_tree_classifier_rejects_invalid_options() {
    ml::DecisionTreeOptions options;
    options.min_samples_split = 1;

    ml::DecisionTreeClassifier tree(options);
}

// -----------------------------------------------------------------------------
// Experiment export helpers
// -----------------------------------------------------------------------------

const std::string k_phase6_output_dir = "outputs/phase-6-trees-ensembles";

void ensure_phase6_output_dir_exists() {
    std::filesystem::create_directories(k_phase6_output_dir);
}

struct TreeExperimentResult {
    std::string experiment_name;
    std::string dataset_name;

    std::size_t max_depth{0};
    std::size_t min_samples_split{0};
    std::size_t min_samples_leaf{0};
    double min_impurity_decrease{0.0};

    double accuracy{0.0};
    std::size_t num_predictions{0};

    bool fitted{false};
};

TreeExperimentResult run_tree_experiment(
    const std::string& experiment_name,
    const std::string& dataset_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::DecisionTreeOptions& options
) {
    ml::DecisionTreeClassifier tree(options);
    tree.fit(X_train, y_train);

    const ml::Vector predictions = tree.predict(X_eval);

    return TreeExperimentResult{
        experiment_name,
        dataset_name,
        options.max_depth,
        options.min_samples_split,
        options.min_samples_leaf,
        options.min_impurity_decrease,
        ml::accuracy_score(predictions, y_eval),
        static_cast<std::size_t>(predictions.size()),
        tree.is_fitted()
    };
}

void export_tree_results_csv(
    const std::vector<TreeExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_tree_results_csv: failed to open output file"
        );
    }

    file << "experiment_name,dataset_name,max_depth,min_samples_split,"
         << "min_samples_leaf,min_impurity_decrease,accuracy,"
         << "num_predictions,fitted\n";

    for (const TreeExperimentResult& result : results) {
        file << result.experiment_name << ","
             << result.dataset_name << ","
             << result.max_depth << ","
             << result.min_samples_split << ","
             << result.min_samples_leaf << ","
             << result.min_impurity_decrease << ","
             << result.accuracy << ","
             << result.num_predictions << ","
             << (result.fitted ? "true" : "false") << "\n";
    }
}

void export_tree_results_txt(
    const std::vector<TreeExperimentResult>& results,
    const std::string& title,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_tree_results_txt: failed to open output file"
        );
    }

    file << title << "\n\n";

    for (const TreeExperimentResult& result : results) {
        file << "Experiment: " << result.experiment_name << "\n"
             << "Dataset: " << result.dataset_name << "\n"
             << "max_depth: " << result.max_depth << "\n"
             << "min_samples_split: " << result.min_samples_split << "\n"
             << "min_samples_leaf: " << result.min_samples_leaf << "\n"
             << "min_impurity_decrease: " << result.min_impurity_decrease << "\n"
             << "Accuracy: " << result.accuracy << "\n"
             << "Predictions: " << result.num_predictions << "\n"
             << "Fitted: " << (result.fitted ? "true" : "false") << "\n\n";
    }
}

// -----------------------------------------------------------------------------
// Decision tree experiment export tests
// -----------------------------------------------------------------------------

void test_experiment_exports_depth_comparison() {
    ensure_phase6_output_dir_exists();

    const ml::Matrix X_train = make_tree_experiment_X();
    const ml::Vector y_train = make_tree_experiment_y();
    const ml::Matrix X_eval = make_tree_experiment_X_eval();
    const ml::Vector y_eval = make_tree_experiment_y_eval();

    std::vector<TreeExperimentResult> results;

    for (std::size_t max_depth : {1, 2, 3}) {
        ml::DecisionTreeOptions options;
        options.max_depth = max_depth;
        options.min_samples_split = 2;
        options.min_samples_leaf = 1;
        options.min_impurity_decrease = 0.0;

        results.push_back(run_tree_experiment(
            "depth_comparison",
            "middle_band_classification",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    export_tree_results_csv(
        results,
        k_phase6_output_dir + "/depth_comparison.csv"
    );

    export_tree_results_txt(
        results,
        "Decision Tree Depth Comparison",
        k_phase6_output_dir + "/depth_comparison.txt"
    );

    if (!std::filesystem::exists(k_phase6_output_dir + "/depth_comparison.csv")) {
        throw std::runtime_error("expected depth_comparison.csv to exist");
    }

    if (!std::filesystem::exists(k_phase6_output_dir + "/depth_comparison.txt")) {
        throw std::runtime_error("expected depth_comparison.txt to exist");
    }
}

void test_experiment_exports_stopping_rules_comparison() {
    ensure_phase6_output_dir_exists();

    const ml::Matrix X_train = make_tree_experiment_X();
    const ml::Vector y_train = make_tree_experiment_y();
    const ml::Matrix X_eval = make_tree_experiment_X_eval();
    const ml::Vector y_eval = make_tree_experiment_y_eval();

    std::vector<TreeExperimentResult> results;

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 3;
        options.min_samples_split = 2;
        options.min_samples_leaf = 1;
        options.min_impurity_decrease = 0.0;

        results.push_back(run_tree_experiment(
            "stopping_rules_comparison",
            "baseline_tree",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 3;
        options.min_samples_split = 8;
        options.min_samples_leaf = 1;
        options.min_impurity_decrease = 0.0;

        results.push_back(run_tree_experiment(
            "stopping_rules_comparison",
            "large_min_samples_split",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 3;
        options.min_samples_split = 2;
        options.min_samples_leaf = 4;
        options.min_impurity_decrease = 0.0;

        results.push_back(run_tree_experiment(
            "stopping_rules_comparison",
            "large_min_samples_leaf",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    {
        ml::DecisionTreeOptions options;
        options.max_depth = 3;
        options.min_samples_split = 2;
        options.min_samples_leaf = 1;
        options.min_impurity_decrease = 0.4;

        results.push_back(run_tree_experiment(
            "stopping_rules_comparison",
            "large_min_impurity_decrease",
            X_train,
            y_train,
            X_eval,
            y_eval,
            options
        ));
    }

    export_tree_results_csv(
        results,
        k_phase6_output_dir + "/stopping_rules_comparison.csv"
    );

    export_tree_results_txt(
        results,
        "Decision Tree Stopping Rules Comparison",
        k_phase6_output_dir + "/stopping_rules_comparison.txt"
    );

    if (!std::filesystem::exists(k_phase6_output_dir + "/stopping_rules_comparison.csv")) {
        throw std::runtime_error("expected stopping_rules_comparison.csv to exist");
    }

    if (!std::filesystem::exists(k_phase6_output_dir + "/stopping_rules_comparison.txt")) {
        throw std::runtime_error("expected stopping_rules_comparison.txt to exist");
    }
}

void test_experiment_exports_tree_prediction_summary() {
    ensure_phase6_output_dir_exists();

    const ml::Matrix X_train = make_tree_experiment_X();
    const ml::Vector y_train = make_tree_experiment_y();
    const ml::Matrix X_eval = make_tree_experiment_X_eval();
    const ml::Vector y_eval = make_tree_experiment_y_eval();

    ml::DecisionTreeOptions options;
    options.max_depth = 3;
    options.min_samples_split = 2;
    options.min_samples_leaf = 1;
    options.min_impurity_decrease = 0.0;

    ml::DecisionTreeClassifier tree(options);
    tree.fit(X_train, y_train);

    const ml::Vector predictions = tree.predict(X_eval);

    const std::string csv_path =
        k_phase6_output_dir + "/tree_prediction_summary.csv";

    const std::string txt_path =
        k_phase6_output_dir + "/tree_prediction_summary.txt";

    {
        std::ofstream file(csv_path);

        if (!file.is_open()) {
            throw std::runtime_error(
                "test_experiment_exports_tree_prediction_summary: failed to open csv"
            );
        }

        file << "sample_index,x0,x1,y_true,y_pred,correct\n";

        for (Eigen::Index i = 0; i < X_eval.rows(); ++i) {
            const bool correct = predictions(i) == y_eval(i);

            file << i << ","
                 << X_eval(i, 0) << ","
                 << X_eval(i, 1) << ","
                 << y_eval(i) << ","
                 << predictions(i) << ","
                 << (correct ? "true" : "false") << "\n";
        }
    }

    {
        std::ofstream file(txt_path);

        if (!file.is_open()) {
            throw std::runtime_error(
                "test_experiment_exports_tree_prediction_summary: failed to open txt"
            );
        }

        file << "Decision Tree Prediction Summary\n\n"
             << "Dataset: middle_band_classification\n"
             << "max_depth: " << options.max_depth << "\n"
             << "min_samples_split: " << options.min_samples_split << "\n"
             << "min_samples_leaf: " << options.min_samples_leaf << "\n"
             << "min_impurity_decrease: " << options.min_impurity_decrease << "\n"
             << "Accuracy: " << ml::accuracy_score(predictions, y_eval) << "\n";
    }

    if (!std::filesystem::exists(csv_path)) {
        throw std::runtime_error("expected tree_prediction_summary.csv to exist");
    }

    if (!std::filesystem::exists(txt_path)) {
        throw std::runtime_error("expected tree_prediction_summary.txt to exist");
    }
}

// -----------------------------------------------------------------------------
// Test runners
// -----------------------------------------------------------------------------

void run_split_scoring_tests() {
    std::cout << "\n[Phase 6.1] Split scoring tests\n\n";

    test::expect_no_throw(
        "gini_impurity returns zero for pure node",
        test_gini_impurity_returns_zero_for_pure_node
    );

    test::expect_no_throw(
        "gini_impurity computes binary mixed node",
        test_gini_impurity_computes_binary_mixed_node
    );

    test::expect_no_throw(
        "gini_impurity computes multiclass node",
        test_gini_impurity_computes_multiclass_node
    );

    test::expect_invalid_argument(
        "gini_impurity rejects empty targets",
        test_gini_impurity_rejects_empty_targets
    );

    test::expect_invalid_argument(
        "gini_impurity rejects non-integer labels",
        test_gini_impurity_rejects_non_integer_labels
    );

    test::expect_invalid_argument(
        "gini_impurity rejects negative labels",
        test_gini_impurity_rejects_negative_labels
    );

    test::expect_no_throw(
        "entropy returns zero for pure node",
        test_entropy_returns_zero_for_pure_node
    );

    test::expect_no_throw(
        "entropy computes binary mixed node",
        test_entropy_computes_binary_mixed_node
    );

    test::expect_no_throw(
        "entropy computes unbalanced binary node",
        test_entropy_computes_unbalanced_binary_node
    );

    test::expect_invalid_argument(
        "entropy rejects empty targets",
        test_entropy_rejects_empty_targets
    );

    test::expect_invalid_argument(
        "entropy rejects non-integer labels",
        test_entropy_rejects_non_integer_labels
    );

    test::expect_invalid_argument(
        "entropy rejects negative labels",
        test_entropy_rejects_negative_labels
    );

    test::expect_no_throw(
        "weighted_child_impurity computes expected value",
        test_weighted_child_impurity_computes_expected_value
    );

    test::expect_no_throw(
        "weighted_child_impurity handles uneven child sizes",
        test_weighted_child_impurity_handles_uneven_child_sizes
    );

    test::expect_invalid_argument(
        "weighted_child_impurity rejects zero left count",
        test_weighted_child_impurity_rejects_zero_left_count
    );

    test::expect_invalid_argument(
        "weighted_child_impurity rejects zero right count",
        test_weighted_child_impurity_rejects_zero_right_count
    );

    test::expect_invalid_argument(
        "weighted_child_impurity rejects negative left impurity",
        test_weighted_child_impurity_rejects_negative_left_impurity
    );

    test::expect_invalid_argument(
        "weighted_child_impurity rejects negative right impurity",
        test_weighted_child_impurity_rejects_negative_right_impurity
    );

    test::expect_invalid_argument(
        "weighted_child_impurity rejects non-finite impurity",
        test_weighted_child_impurity_rejects_non_finite_impurity
    );

    test::expect_no_throw(
        "impurity_reduction computes expected value",
        test_impurity_reduction_computes_expected_value
    );

    test::expect_no_throw(
        "impurity_reduction allows zero improvement",
        test_impurity_reduction_allows_zero_improvement
    );

    test::expect_no_throw(
        "impurity_reduction allows negative improvement",
        test_impurity_reduction_allows_negative_improvement
    );

    test::expect_invalid_argument(
        "impurity_reduction rejects negative parent impurity",
        test_impurity_reduction_rejects_negative_parent_impurity
    );

    test::expect_invalid_argument(
        "impurity_reduction rejects negative child impurity",
        test_impurity_reduction_rejects_negative_child_impurity
    );

    test::expect_invalid_argument(
        "impurity_reduction rejects non-finite parent impurity",
        test_impurity_reduction_rejects_non_finite_parent_impurity
    );

    test::expect_invalid_argument(
        "impurity_reduction rejects non-finite child impurity",
        test_impurity_reduction_rejects_non_finite_child_impurity
    );
}

void run_candidate_threshold_tests() {
    std::cout << "\n[Phase 6.2] Candidate threshold evaluation tests\n\n";

    test::expect_no_throw(
        "evaluate_candidate_threshold finds valid split",
        test_evaluate_candidate_threshold_finds_valid_split
    );

    test::expect_no_throw(
        "evaluate_candidate_threshold returns invalid for empty left child",
        test_evaluate_candidate_threshold_returns_invalid_for_empty_left_child
    );

    test::expect_no_throw(
        "evaluate_candidate_threshold returns invalid for empty right child",
        test_evaluate_candidate_threshold_returns_invalid_for_empty_right_child
    );

    test::expect_no_throw(
        "evaluate_candidate_threshold respects min_samples_leaf",
        test_evaluate_candidate_threshold_respects_min_samples_leaf
    );

    test::expect_no_throw(
        "evaluate_candidate_threshold respects min_impurity_decrease",
        test_evaluate_candidate_threshold_respects_min_impurity_decrease
    );

    test::expect_invalid_argument(
        "evaluate_candidate_threshold rejects empty X",
        test_evaluate_candidate_threshold_rejects_empty_X
    );

    test::expect_invalid_argument(
        "evaluate_candidate_threshold rejects empty y",
        test_evaluate_candidate_threshold_rejects_empty_y
    );

    test::expect_invalid_argument(
        "evaluate_candidate_threshold rejects mismatched X/y",
        test_evaluate_candidate_threshold_rejects_mismatched_X_y
    );

    test::expect_invalid_argument(
        "evaluate_candidate_threshold rejects invalid feature index",
        test_evaluate_candidate_threshold_rejects_invalid_feature_index
    );

    test::expect_invalid_argument(
        "evaluate_candidate_threshold rejects non-finite threshold",
        test_evaluate_candidate_threshold_rejects_non_finite_threshold
    );

    test::expect_invalid_argument(
        "evaluate_candidate_threshold rejects invalid options",
        test_evaluate_candidate_threshold_rejects_invalid_options
    );
}

void run_best_split_tests() {
    std::cout << "\n[Phase 6.3] Best split selection tests\n\n";

    test::expect_no_throw(
        "find_best_split returns best feature and threshold",
        test_find_best_split_returns_best_feature_and_threshold
    );

    test::expect_no_throw(
        "find_best_split returns invalid when all features constant",
        test_find_best_split_returns_invalid_when_all_features_constant
    );

    test::expect_no_throw(
        "find_best_split respects min_samples_leaf",
        test_find_best_split_respects_min_samples_leaf
    );

    test::expect_no_throw(
        "find_best_split respects min_impurity_decrease",
        test_find_best_split_respects_min_impurity_decrease
    );

    test::expect_no_throw(
        "find_best_split uses deterministic feature tie-breaking",
        test_find_best_split_uses_deterministic_feature_tie_breaking
    );

    test::expect_no_throw(
        "find_best_split uses deterministic threshold tie-breaking",
        test_find_best_split_uses_deterministic_threshold_tie_breaking
    );

    test::expect_invalid_argument(
        "find_best_split rejects empty X",
        test_find_best_split_rejects_empty_X
    );

    test::expect_invalid_argument(
        "find_best_split rejects empty y",
        test_find_best_split_rejects_empty_y
    );

    test::expect_invalid_argument(
        "find_best_split rejects mismatched X/y",
        test_find_best_split_rejects_mismatched_X_y
    );

    test::expect_invalid_argument(
        "find_best_split rejects non-finite feature value",
        test_find_best_split_rejects_non_finite_feature_value
    );

    test::expect_invalid_argument(
        "find_best_split rejects invalid options",
        test_find_best_split_rejects_invalid_options
    );
}

void run_recursive_tree_growth_tests() {
    std::cout << "\n[Phase 6.4] Recursive tree growth tests\n\n";

    test::expect_no_throw(
        "majority_class returns most common label",
        test_majority_class_returns_most_common_label
    );

    test::expect_no_throw(
        "majority_class uses smallest label on tie",
        test_majority_class_uses_smallest_label_on_tie
    );

    test::expect_no_throw(
        "is_pure_node detects pure node",
        test_is_pure_node_detects_pure_node
    );

    test::expect_no_throw(
        "is_pure_node detects mixed node",
        test_is_pure_node_detects_mixed_node
    );

    test::expect_no_throw(
        "split_dataset splits rows correctly",
        test_split_dataset_splits_rows_correctly
    );

    test::expect_no_throw(
        "build_tree creates leaf for pure node",
        test_build_tree_creates_leaf_for_pure_node
    );

    test::expect_no_throw(
        "build_tree creates internal root for separable data",
        test_build_tree_creates_internal_root_for_separable_data
    );

    test::expect_invalid_argument(
        "build_tree rejects invalid max_depth",
        test_build_tree_respects_max_depth
    );

    test::expect_no_throw(
        "build_tree respects min_samples_split",
        test_build_tree_respects_min_samples_split
    );

    test::expect_no_throw(
        "build_tree creates leaf when no valid split exists",
        test_build_tree_creates_leaf_when_no_valid_split_exists
    );

    test::expect_invalid_argument(
        "build_tree rejects invalid labels",
        test_build_tree_rejects_invalid_labels
    );
}

void run_stopping_rule_tests() {
    std::cout << "\n[Phase 6.5] Decision tree stopping rule tests\n\n";

    test::expect_no_throw(
        "build_tree stops when node is pure",
        test_build_tree_stops_when_node_is_pure
    );

    test::expect_invalid_argument(
        "build_tree rejects max_depth zero",
        test_build_tree_stops_at_max_depth_zero_rejected_by_options
    );

    test::expect_no_throw(
        "build_tree stops at max_depth one",
        test_build_tree_stops_at_max_depth_one
    );

    test::expect_no_throw(
        "build_tree allows deeper tree when max_depth allows",
        test_build_tree_allows_deeper_tree_when_max_depth_allows
    );

    test::expect_no_throw(
        "build_tree stops when min_samples_split is not met",
        test_build_tree_stops_when_min_samples_split_is_not_met
    );

    test::expect_no_throw(
        "build_tree allows split when min_samples_split is met",
        test_build_tree_allows_split_when_min_samples_split_is_met
    );

    test::expect_no_throw(
        "build_tree stops when min_samples_leaf blocks all splits",
        test_build_tree_stops_when_min_samples_leaf_blocks_all_splits
    );

    test::expect_no_throw(
        "build_tree allows split when min_samples_leaf is satisfied",
        test_build_tree_allows_split_when_min_samples_leaf_is_satisfied
    );

    test::expect_no_throw(
        "build_tree stops when min_impurity_decrease blocks best split",
        test_build_tree_stops_when_min_impurity_decrease_blocks_best_split
    );

    test::expect_no_throw(
        "build_tree allows split when min_impurity_decrease is satisfied",
        test_build_tree_allows_split_when_min_impurity_decrease_is_satisfied
    );

    test::expect_no_throw(
        "build_tree stops when no valid split exists due to constant features",
        test_build_tree_stops_when_no_valid_split_exists_due_to_constant_features
    );

    test::expect_invalid_argument(
        "build_tree rejects invalid min_samples_split option",
        test_build_tree_rejects_invalid_min_samples_split_option
    );

    test::expect_invalid_argument(
        "build_tree rejects invalid min_samples_leaf option",
        test_build_tree_rejects_invalid_min_samples_leaf_option
    );

    test::expect_invalid_argument(
        "build_tree rejects invalid min_impurity_decrease option",
        test_build_tree_rejects_invalid_min_impurity_decrease_option
    );
}

void run_decision_tree_prediction_tests() {
    std::cout << "\n[Phase 6.6] Decision tree prediction traversal tests\n\n";

    test::expect_no_throw(
        "DecisionTreeClassifier reports not fitted initially",
        test_decision_tree_classifier_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "DecisionTreeClassifier fit marks model as fitted",
        test_decision_tree_classifier_fit_marks_model_as_fitted
    );

    test::expect_no_throw(
        "DecisionTreeClassifier predicts training data",
        test_decision_tree_classifier_predicts_training_data
    );

    test::expect_no_throw(
        "DecisionTreeClassifier predicts new samples by traversal",
        test_decision_tree_classifier_predicts_new_samples_by_traversal
    );

    test::expect_no_throw(
        "DecisionTreeClassifier predicts majority leaf when stopped early",
        test_decision_tree_classifier_predicts_majority_leaf_when_stopped_early
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects predict before fit",
        test_decision_tree_classifier_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects empty predict matrix",
        test_decision_tree_classifier_rejects_empty_predict_matrix
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects predict feature mismatch",
        test_decision_tree_classifier_rejects_predict_feature_mismatch
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects empty fit matrix",
        test_decision_tree_classifier_rejects_empty_fit_matrix
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects empty fit targets",
        test_decision_tree_classifier_rejects_empty_fit_targets
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects mismatched fit data",
        test_decision_tree_classifier_rejects_mismatched_fit_data
    );

    test::expect_invalid_argument(
        "DecisionTreeClassifier rejects invalid options",
        test_decision_tree_classifier_rejects_invalid_options
    );
}

void run_decision_tree_experiment_export_tests() {
    std::cout << "\n[Phase 6.7] Decision tree experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports depth comparison",
        test_experiment_exports_depth_comparison
    );

    test::expect_no_throw(
        "Experiment exports stopping rules comparison",
        test_experiment_exports_stopping_rules_comparison
    );

    test::expect_no_throw(
        "Experiment exports tree prediction summary",
        test_experiment_exports_tree_prediction_summary
    );
}

}  // namespace

namespace ml::experiments {

void run_phase6_trees_sanity() {
    run_split_scoring_tests();
    run_candidate_threshold_tests();
    run_best_split_tests();
    run_recursive_tree_growth_tests();
    run_stopping_rule_tests();
    run_decision_tree_prediction_tests();
    run_decision_tree_experiment_export_tests();
}

}  // namespace ml::experiments