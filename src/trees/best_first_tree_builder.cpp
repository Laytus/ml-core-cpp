#include "ml/trees/best_first_tree_builder.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/trees/tree_builder.hpp"

#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace ml {

namespace {

struct BestFirstLeaf {
    TreeNode* node{nullptr};
    Matrix X;
    Vector y;
    Vector sample_weight;
    SplitCandidate split;
};

void validate_best_first_training_data(
    const Matrix& X,
    const Vector& y,
    const std::string& context
) {
    validate_non_empty_matrix(X, context);
    validate_non_empty_vector(y, context);
    validate_same_number_of_rows(X, y, context);
}

std::unique_ptr<TreeNode> make_best_first_leaf(
    const Vector& y
) {
    auto node = std::make_unique<TreeNode>();

    node->is_leaf = true;
    node->prediction = majority_class(y);
    node->impurity = gini_impurity(y);
    node->impurity_decrease = 0.0;
    node->num_samples = static_cast<std::size_t>(y.size());

    return node;
}

std::unique_ptr<TreeNode> make_best_first_leaf(
    const Vector& y,
    const Vector& sample_weight
) {
    auto node = std::make_unique<TreeNode>();

    node->is_leaf = true;
    node->prediction = majority_class(y, sample_weight);
    node->impurity = weighted_gini_impurity(y, sample_weight);
    node->impurity_decrease = 0.0;
    node->num_samples = static_cast<std::size_t>(y.size());

    return node;
}

bool can_attempt_split(
    const Vector& y,
    const DecisionTreeOptions& options
) {
    if (is_pure_node(y)) {
        return false;
    }

    if (static_cast<std::size_t>(y.size()) < options.min_samples_split) {
        return false;
    }

    return true;
}

BestFirstLeaf make_leaf_candidate(
    TreeNode* node,
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options
) {
    BestFirstLeaf candidate;
    candidate.node = node;
    candidate.X = X;
    candidate.y = y;

    if (can_attempt_split(y, options)) {
        candidate.split = find_best_split(
            X,
            y,
            options
        );
    }

    return candidate;
}

BestFirstLeaf make_leaf_candidate(
    TreeNode* node,
    const Matrix& X,
    const Vector& y,
    const Vector& sample_weight,
    const DecisionTreeOptions& options
) {
    BestFirstLeaf candidate;
    candidate.node = node;
    candidate.X = X;
    candidate.y = y;
    candidate.sample_weight = sample_weight;

    if (can_attempt_split(y, options)) {
        candidate.split = find_best_split(
            X,
            y,
            sample_weight,
            options
        );
    }

    return candidate;
}

bool is_better_leaf_to_split(
    const BestFirstLeaf& candidate,
    const BestFirstLeaf& best
) {
    if (!candidate.split.valid) {
        return false;
    }

    if (!best.split.valid) {
        return true;
    }

    constexpr double epsilon = 1e-12;

    if (candidate.split.impurity_decrease > best.split.impurity_decrease + epsilon) {
        return true;
    }

    if (
        std::abs(candidate.split.impurity_decrease - best.split.impurity_decrease) <= epsilon
    ) {
        if (candidate.split.feature_index < best.split.feature_index) {
            return true;
        }

        if (
            candidate.split.feature_index == best.split.feature_index &&
            candidate.split.threshold < best.split.threshold
        ) {
            return true;
        }
    }
    return false; 
}

std::size_t select_best_leaf_index(
    const std::vector<BestFirstLeaf>& leaves
) {
    BestFirstLeaf best;
    std::size_t best_index = leaves.size();

    for (std::size_t i = 0; i < leaves.size(); ++i) {
        if (is_better_leaf_to_split(leaves[i], best)) {
            best = leaves[i];
            best_index = i;
        }
    }

    return best_index;
}

}  // namespace 

std::unique_ptr<TreeNode> build_tree_best_first(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options
) {
    validate_decision_tree_options(options, "build_tree_best_first");
    validate_best_first_training_data(X, y, "build_tree_best_first");

    if (!options.max_leaf_nodes.has_value()) {
        throw std::invalid_argument(
            "build_tree_best_first: max_leaf_nodes must be provided"
        );
    }

    const std::size_t max_leaf_nodes = options.max_leaf_nodes.value();

    auto root = make_best_first_leaf(y);

    std::vector<BestFirstLeaf> leaves;
    leaves.push_back(
        make_leaf_candidate(
            root.get(),
            X,
            y,
            options
        )
    );

    while (leaves.size() < max_leaf_nodes) {
        const std::size_t best_index = select_best_leaf_index(leaves);

        if (best_index == leaves.size()) {
            break;
        }

        BestFirstLeaf selected = leaves[best_index];

        DatasetSplit dataset_split = split_dataset(
            selected.X,
            selected.y,
            selected.split.feature_index,
            selected.split.threshold
        );

        selected.node->is_leaf = false;
        selected.node->prediction = majority_class(selected.y);
        selected.node->feature_index = selected.split.feature_index;
        selected.node->threshold = selected.split.threshold;
        selected.node->impurity = selected.split.parent_impurity;
        selected.node->impurity_decrease = selected.split.impurity_decrease;
        selected.node->num_samples = static_cast<std::size_t>(selected.y.size());

        selected.node->left = make_best_first_leaf(dataset_split.y_left);
        selected.node->right = make_best_first_leaf(dataset_split.y_right);

        TreeNode* left_ptr = selected.node->left.get();
        TreeNode* right_ptr = selected.node->right.get();

        leaves.erase(leaves.begin() + static_cast<std::ptrdiff_t>(best_index));

        leaves.push_back(
            make_leaf_candidate(
                left_ptr,
                dataset_split.X_left,
                dataset_split.y_left,
                options
            )
        );

        leaves.push_back(
            make_leaf_candidate(
                right_ptr,
                dataset_split.X_right,
                dataset_split.y_right,
                options
            )
        );
    }

    return root;
}

std::unique_ptr<TreeNode> build_tree_best_first(
    const Matrix& X,
    const Vector& y,
    const Vector& sample_weight,
    const DecisionTreeOptions& options
) {
    validate_decision_tree_options(
        options,
        "build_tree_best_first weighted"
    );

    validate_best_first_training_data(
        X,
        y,
        "build_tree_best_first weighted"
    );

    validate_non_empty_vector(
        sample_weight,
        "build_tree_best_first weighted sample_weight"
    );

    validate_same_size(
        y,
        sample_weight,
        "build_tree_best_first weighted"
    );

    if (!options.max_leaf_nodes.has_value()) {
        throw std::invalid_argument(
            "build_tree_best_first weighted: max_leaf_nodes must be provided"
        );
    }

    const std::size_t max_leaf_nodes = options.max_leaf_nodes.value();

    auto root = make_best_first_leaf(
        y,
        sample_weight
    );

    std::vector<BestFirstLeaf> leaves;
    leaves.push_back(
        make_leaf_candidate(
            root.get(),
            X,
            y,
            sample_weight,
            options
        )
    );

    while (leaves.size() < max_leaf_nodes) {
        const std::size_t best_index = select_best_leaf_index(leaves);

        if (best_index == leaves.size()) {
            break;
        }

        BestFirstLeaf selected = leaves[best_index];

        DatasetSplit dataset_split = split_dataset(
            selected.X,
            selected.y,
            selected.sample_weight,
            selected.split.feature_index,
            selected.split.threshold
        );

        selected.node->is_leaf = false;
        selected.node->prediction = majority_class(
            selected.y,
            selected.sample_weight
        );
        selected.node->feature_index = selected.split.feature_index;
        selected.node->threshold = selected.split.threshold;
        selected.node->impurity = selected.split.parent_impurity;
        selected.node->impurity_decrease = selected.split.impurity_decrease;
        selected.node->num_samples = static_cast<std::size_t>(selected.y.size());

        selected.node->left = make_best_first_leaf(
            dataset_split.y_left,
            dataset_split.sample_weight_left
        );

        selected.node->right = make_best_first_leaf(
            dataset_split.y_right,
            dataset_split.sample_weight_right
        );

        TreeNode* left_ptr = selected.node->left.get();
        TreeNode* right_ptr = selected.node->right.get();

        leaves.erase(
            leaves.begin() + static_cast<std::ptrdiff_t>(best_index)
        );

        leaves.push_back(
            make_leaf_candidate(
                left_ptr,
                dataset_split.X_left,
                dataset_split.y_left,
                dataset_split.sample_weight_left,
                options
            )
        );

        leaves.push_back(
            make_leaf_candidate(
                right_ptr,
                dataset_split.X_right,
                dataset_split.y_right,
                dataset_split.sample_weight_right,
                options
            )
        );
    }

    return root;
}

}  // namespace ml