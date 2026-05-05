#include "ml/trees/tree_builder.hpp"

#include "ml/common/shape_validation.hpp"

#include <cmath>
#include <map>
#include <stdexcept>

namespace ml {

namespace {

std::map<int, std::size_t> label_counts(const Vector& y) {
    validate_non_empty_vector(y, "label_counts");

    std::map<int, std::size_t> counts;

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double value = y(i);
        const double rounded = std::round(value);

        if (std::abs(value - rounded) > 1e-12) {
            throw std::invalid_argument(
                "label_counts: class labels must be integer valued"
            );
        }

        if (rounded < 0.0) {
            throw std::invalid_argument(
                "label_counts: class labels must be non-negative"
            );
        }

        counts[static_cast<int>(rounded)] += 1;
    }

    return counts;
}

void validate_tree_training_data(
    const Matrix& X,
    const Vector& y,
    const std::string& context
) {
    validate_non_empty_matrix(X, context);
    validate_non_empty_vector(y, context);
    validate_same_number_of_rows(X, y, context);
}

std::unique_ptr<TreeNode> make_leaf(
    const Vector& y,
    double impurity
) {
    auto node = std::make_unique<TreeNode>();

    node->is_leaf = true;
    node->prediction = majority_class(y);
    node->impurity = impurity;
    node->impurity_decrease = 0.0;
    node->num_samples = static_cast<std::size_t>(y.size());

    return node;
}

}  // namespace

double majority_class(const Vector& y) {
    const auto counts = label_counts(y);

    int best_label = -1;
    std::size_t best_count = 0;

    for (const auto& [label, count] : counts) {
        if (count > best_count) {
            best_label = label;
            best_count = count;
        }
    }

    return static_cast<double>(best_label);
}

bool is_pure_node(const Vector& y) {
    validate_non_empty_vector(y, "is_pure_node");

    const double first = y(0);

    for (Eigen::Index i = 1; i < y.size(); ++i) {
        if (y(i) != first) {
            return false;
        }
    }

    return true;
}

DatasetSplit split_dataset(
    const Matrix& X,
    const Vector& y,
    Eigen::Index feature_index,
    double threshold
) {
    validate_tree_training_data(X, y, "split_dataset");

    if (feature_index < 0 || feature_index >= X.cols()) {
        throw std::invalid_argument(
            "split_dataset: feature_index is out of range"
        );
    }

    if (!std::isfinite(threshold)) {
        throw std::invalid_argument(
            "split_dataset: threshold must be finite"
        );
    }

    std::size_t left_count = 0;
    std::size_t right_count = 0;

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        if (X(i, feature_index) <= threshold) {
            ++left_count;
        } else {
            ++right_count;
        }
    }

    DatasetSplit split;

    split.X_left = Matrix(
        static_cast<Eigen::Index>(left_count),
        X.cols()
    );
    
    split.y_left = Vector(
        static_cast<Eigen::Index>(left_count)
    );

    split.X_right = Matrix(
        static_cast<Eigen::Index>(right_count),
        X.cols()
    );
    
    split.y_right = Vector(
        static_cast<Eigen::Index>(right_count)
    );

    Eigen::Index left_index = 0;
    Eigen::Index right_index = 0;

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        if (X(i, feature_index) <= threshold) {
            split.X_left.row(left_index) = X.row(i);
            split.y_left(left_index) = y(i);
            ++left_index;
        } else {
            split.X_right.row(right_index) = X.row(i);
            split.y_right(right_index) = y(i);
            ++right_index;
        }
    }

    return split;
}

std::unique_ptr<TreeNode> build_tree(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options,
    std::size_t depth
) {
    validate_decision_tree_options(options, "build_tree");
    validate_tree_training_data(X, y, "build_tree");

    const double node_impurity = gini_impurity(y);

    if (is_pure_node(y)) {
        return make_leaf(y, node_impurity);
    }

    if (depth >= options.max_depth) {
        return make_leaf(y, node_impurity);
    }
    
    if (static_cast<std::size_t>(y.size()) < options.min_samples_split) {
        return make_leaf(y, node_impurity);
    }

    const SplitCandidate split = find_best_split(
        X,
        y,
        options
    );

    if (!split.valid) {
        return make_leaf(y, node_impurity);
    }

    DatasetSplit dataset_split = split_dataset(
        X,
        y,
        split.feature_index,
        split.threshold
    );

    auto node = std::make_unique<TreeNode>();

    node->is_leaf = false;
    node->prediction = majority_class(y);
    node->feature_index = split.feature_index;
    node->threshold = split.threshold;
    node->impurity = node_impurity;
    node->impurity_decrease = split.impurity_decrease;
    node->num_samples = static_cast<std::size_t>(y.size());

    node->left = build_tree(
        dataset_split.X_left,
        dataset_split.y_left,
        options,
        depth + 1
    );
    
    node->right = build_tree(
        dataset_split.X_right,
        dataset_split.y_right,
        options,
        depth + 1
    );

    return node;
}

}  // namespace ml