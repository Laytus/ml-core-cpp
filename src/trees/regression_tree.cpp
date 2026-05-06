#include "ml/trees/regression_tree.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/common/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ml {

namespace {

void validate_finite_matrix_values(
    const Matrix& X,
    const std::string& context
) {
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            if (!std::isfinite(X(i, j))) {
                throw std::invalid_argument(
                    context + ": X values must be finite"
                );
            }
        }
    }
}

void validate_finite_vector_values(
    const Vector& y,
    const std::string& context
) {
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y(i))) {
            throw std::invalid_argument(
                context + ": y values must be finite"
            );
        }
    }
}

double vector_mean_value(
    const Vector& y
) {
    validate_non_empty_vector(y, "vector_mean_value");

    return y.sum() / static_cast<double>(y.size());
}

double sum_squared_error(
    const Vector& y
) {
    validate_non_empty_vector(y, "sum_squared_error");

    const double mean = vector_mean_value(y);

    double result = 0.0;

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double residual = y(i) - mean;
        result += residual * residual;
    }

    return result;
}

std::vector<double> sorted_unique_feature_values(
    const Matrix& X,
    Eigen::Index feature_index
) {
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(X.rows()));

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const double value = X(i, feature_index);

        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "sorted_unique_feature_values: feature value must be finite"
            );
        }

        values.push_back(value);
    }

    std::sort(values.begin(), values.end());

    values.erase(
        std::unique(values.begin(), values.end()),
        values.end()
    );

    return values;
}

std::vector<double> midpoint_thresholds(
    const std::vector<double>& sorted_unique_values
) {
    std::vector<double> thresholds;

    if (sorted_unique_values.size() < 2) {
        return thresholds;
    }

    thresholds.reserve(sorted_unique_values.size() - 1);

    for (std::size_t i = 0; i < sorted_unique_values.size(); ++i) {
        thresholds.push_back(
            0.5 * (sorted_unique_values[i] + sorted_unique_values[i + 1])
        );
    }

    return thresholds;
}

struct RegressionSplit {
    bool valid{false};
    Eigen::Index feature_index{-1};
    double threshold{0.0};

    std::size_t left_count{0};
    std::size_t right_count{0};

    double parent_error{0.0};
    double left_error{0.0};
    double right_error{0.0};
    double error_decrease{0.0};
};

struct RegressionDatasetSplit {
    Matrix X_left;
    Vector y_left;
    Matrix X_right;
    Vector y_right;
};

RegressionDatasetSplit split_regression_dataset(
    const Matrix& X,
    const Vector& y,
    Eigen::Index feature_index,
    double threshold
) {
    std::size_t left_count = 0;
    std::size_t right_count = 0;

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        if (X(i, feature_index) <= threshold) {
            ++left_count;
        } else {
            ++right_count;
        }
    }

    RegressionDatasetSplit split;

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

RegressionSplit evaluate_regression_threshold(
    const Matrix& X,
    const Vector& y,
    Eigen::Index feature_index,
    double threshold,
    const RegressionTreeOptions& options
) {
    RegressionSplit candidate;
    candidate.feature_index = feature_index;
    candidate.threshold = threshold;
    candidate.parent_error = sum_squared_error(y);

    std::size_t left_count = 0;
    std::size_t right_count = 0;

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        if (X(i, feature_index) <= threshold) {
            ++left_count;
        } else {
            ++right_count;
        }
    }

    candidate.left_count = left_count;
    candidate.right_count = right_count;

    if (left_count == 0 || right_count == 0) {
        return candidate;
    }

    if (
        left_count < options.min_samples_leaf ||
        right_count < options.min_samples_leaf
    ) {
        return candidate;
    }

    RegressionDatasetSplit split = split_regression_dataset(
        X,
        y,
        feature_index,
        threshold
    );

    candidate.left_error = sum_squared_error(split.y_left);
    candidate.right_error = sum_squared_error(split.y_right);

    candidate.error_decrease =
        candidate.parent_error -
        (candidate.left_error + candidate.right_error);

    if (candidate.error_decrease < options.min_error_decrease) {
        return candidate;
    }

    candidate.valid = true;
    return candidate;
}

bool is_better_regression_split(
    const RegressionSplit& candidate,
    const RegressionSplit& best
) {

    if (!candidate.valid) {
        return false;
    }

    if (!best.valid) {
        return true;
    }

    constexpr double epsilon = 1e-12;

    if (candidate.error_decrease > best.error_decrease + epsilon) {
        return true;
    }

    if (std::abs(candidate.error_decrease - best.error_decrease) <= epsilon) {
        if (candidate.feature_index < best.feature_index) {
            return true;
        }

        if (
            candidate.feature_index == best.feature_index &&
            candidate.threshold < best.threshold
        ) {
            return true;
        }
    }

    return false;
}

RegressionSplit find_best_regression_split(
    const Matrix& X,
    const Vector& y,
    const RegressionTreeOptions& options
) {
    RegressionSplit best;

    for (Eigen::Index feature_index = 0; feature_index < X.cols(); ++feature_index) {
        const std::vector<double> unique_values =
            sorted_unique_feature_values(
                X,
                feature_index
            );

        const std::vector<double> thresholds = midpoint_thresholds(unique_values);

        for (double threshold : thresholds) {
            const RegressionSplit candidate =
                evaluate_regression_threshold(
                    X,
                    y,
                    feature_index,
                    threshold,
                    options
                );

            if (is_better_regression_split(candidate, best)) {
                best = candidate;
            }
        }
    }

    return best;
}

bool target_values_are_constant(
    const Vector& y
) {
    if (y.size() <= 1) {
        return true;
    }

    constexpr double epsilon = 1e-12;

    for (Eigen::Index i = 1; i < y.size(); ++i) {
        if (std::abs(y(i) - y(0)) > epsilon) {
            return false;
        }
    }

    return true;
}

}  // namespace

void validate_regression_tree_options(
    const RegressionTreeOptions& options,
    const std::string& context
) {
    if (options.max_depth == 0) {
        throw std::invalid_argument(
            context + ": max_depth must be at least 1"
        );
    }

    if (options.min_samples_split < 2) {
        throw std::invalid_argument(
            context + ": min_samples_split must be at least 2"
        );
    }

    if (options.min_samples_leaf == 0) {
        throw std::invalid_argument(
            context + ": min_samples_leaf must be at least 1"
        );
    }

    if (
        !std::isfinite(options.min_error_decrease) ||
        options.min_error_decrease < 0.0
    ) {
        throw std::invalid_argument(
            context + ": min_error_decrease must be finite and non-negative"
        );
    }
}

DecisionTreeRegressor::DecisionTreeRegressor(
    RegressionTreeOptions options
)
    : options_{options} {
    validate_regression_tree_options(
        options_,
        "DecisionTreeRegressor"
    );
}

void DecisionTreeRegressor::fit(
    const Matrix& X,
    const Vector& y
) {
    validate_regression_tree_options(options_, "DecisionTreeRegressor::fit");
    
    validate_non_empty_matrix(X, "DecisionTreeRegressor::fit");
    validate_non_empty_vector(y, "DecisionTreeRegressor::fit");
    validate_same_number_of_rows(X, y, "DecisionTreeRegressor::fit");
    
    validate_finite_matrix_values(X, "DecisionTreeRegressor::fit");
    validate_finite_vector_values(y, "DecisionTreeRegressor::fit");

    root_ = build_node(X, y, 0);

    num_features_ = X.cols();
}

Vector DecisionTreeRegressor::predict(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "DecisionTreeRegressor::predict: model must be fitted before prediction"
        );
    }

    validate_non_empty_matrix(X, "DecisionTreeRegressor::predict");

    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "DecisionTreeRegressor::predict: X feature count must match training feature count"
        );
    }

    validate_finite_matrix_values(X, "DecisionTreeRegressor::predict");

    Vector predictions(X.rows());

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        predictions(i) = predict_one(
            X.row(i).transpose(),
            *root_
        );
    }

    return predictions;
}

bool DecisionTreeRegressor::is_fitted() const {
    return root_ != nullptr;
}

const RegressionTreeOptions& DecisionTreeRegressor::options() const {
    return options_;
}


std::unique_ptr<DecisionTreeRegressor::Node> DecisionTreeRegressor::build_node(
    const Matrix& X,
    const Vector& y,
    std::size_t depth
) const {
    auto node = std::make_unique<Node>();

    node->is_leaf = true;
    node->prediction = vector_mean_value(y);
    node->num_samples = static_cast<std::size_t>(y.size());
    node->squared_error = sum_squared_error(y);
    node->error_decrease = 0.0;

    if (target_values_are_constant(y)) {
        return node;
    }

    if (depth >= options_.max_depth) {
        return node;
    }

    if (static_cast<std::size_t>(y.size()) < options_.min_samples_split) {
        return node;
    }

    const RegressionSplit split = find_best_regression_split(
        X,
        y,
        options_
    );

    if (!split.valid) {
        return node;
    }

    RegressionDatasetSplit dataset_split = split_regression_dataset(
        X,
        y,
        split.feature_index,
        split.threshold
    );

    node->is_leaf = false;
    node->feature_index = split.feature_index;
    node->threshold = split.threshold;
    node->squared_error = split.parent_error;
    node->error_decrease = split.error_decrease;

    node->left = build_node(
        dataset_split.X_left,
        dataset_split.y_left,
        depth + 1
    );

    node->right = build_node(
        dataset_split.X_right,
        dataset_split.y_right,
        depth + 1
    );

    return node;
}

double DecisionTreeRegressor::predict_one(
    const Vector& sample,
    const Node& node
) const {
    if (node.is_leaf) {
        return node.prediction;
    }

    if (!node.left || !node.right) {
        throw std::runtime_error(
            "DecisionTreeRegressor::predict_one: internal node is missing children"
        );
    }

    if (node.feature_index < 0 || node.feature_index >= sample.size()) {
        throw std::runtime_error(
            "DecisionTreeRegressor::predict_one: node feature_index is invalid"
        );
    }

    if (sample(node.feature_index) <= node.threshold) {
        return predict_one(
            sample,
            *node.left
        );
    }

    return predict_one(
        sample,
        *node.right
    );
}


}  // namespace ml