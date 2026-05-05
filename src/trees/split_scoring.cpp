#include "ml/trees/split_scoring.hpp"

#include "ml/common/shape_validation.hpp"

#include <cmath>
#include <algorithm>
#include <map>
#include <vector>
#include <stdexcept>
#include <string>

namespace ml {

namespace {

std::map<int, std::size_t> class_counts(const Vector& y) {
    validate_non_empty_vector(y, "class_counts");

    std::map<int, std::size_t> counts;

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double value = y(i);
        const double rounded = std::round(value);

        if (std::abs(value - rounded) > 1e-12) {
            throw std::invalid_argument(
                "class_count: class labels must be integer valued"
            );
        }

        if (rounded < 0.0) {
            throw std::invalid_argument(
                "class_count: class labels must be non-negative"
            );
        }

        counts[static_cast<int>(rounded)] += 1;
    }

    return counts;
}

void validate_impurity_value(
    double impurity,
    const char* name
) {
    if (!std::isfinite(impurity)) {
        throw std::invalid_argument(
            std::string(name) + ": impurity must be finite"
        );;
    }
    
    if (impurity < 0.0) {
        throw std::invalid_argument(
            std::string(name) + ": impurity must be non-negative"
        );;
    }
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
        const double left = sorted_unique_values[i];
        const double right = sorted_unique_values[i + 1];

        // thresholds.push_back(left + 0.5 * (right - left));
        thresholds.push_back(0.5 * (left + right));
    }

    return thresholds;
}

bool is_better_split(
    const SplitCandidate& candidate,
    const SplitCandidate& best
) {
    if (!candidate.valid) {
        return false;
    }
    
    if (!best.valid) {
        return true;
    }

    constexpr double epsilon = 1e-12;

    if (candidate.impurity_decrease > best.impurity_decrease + epsilon) {
        return true;
    }

    if (std::abs(candidate.impurity_decrease - best.impurity_decrease) <= epsilon) {
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

}  // namespace

double gini_impurity(const Vector& y) {
    validate_non_empty_vector(y, "gini_impurity");

    const auto counts = class_counts(y);
    const double n = static_cast<double>(y.size());

    double squared_proportion_sum = 0.0;

    for (const auto& [label, count] : counts) {
        static_cast<void>(label);

        const double proportion = static_cast<double>(count) / n;

        squared_proportion_sum += proportion * proportion;
    }

    return 1.0 - squared_proportion_sum;
}

double entropy(const Vector& y) {
    validate_non_empty_vector(y, "entropy");

    const auto counts = class_counts(y);
    const double n = static_cast<double>(y.size());

    double result = 0.0;

    for (const auto& [label, count] : counts) {
        static_cast<void>(label);

        const double proportion = static_cast<double>(count) / n;

        if (proportion > 0.0) {
            result -= proportion * std::log2(proportion);
        }
    }

    return result;
}

double weighted_child_impurity(
    double left_impurity,
    double right_impurity,
    std::size_t left_count,
    std::size_t right_count
) {
    validate_impurity_value(left_impurity, "weighted_child_impurity left_impurity");
    validate_impurity_value(right_impurity, "weighted_child_impurity right_impurity");

    const std::size_t total_count = left_count + right_count;

    if (total_count == 0) {
        throw std::invalid_argument(
            "weighted_child_impurity: total child count must be positive"
        );
    }

    if (left_count == 0 || right_count == 0) {
        throw std::invalid_argument(
            "weighted_child_impurity: both child counts must be positive"
        );
    }

    const double left_weight = static_cast<double>(left_count) / static_cast<double>(total_count);
    const double right_weight = static_cast<double>(right_count) / static_cast<double>(total_count);

    return left_weight * left_impurity + right_weight * right_impurity;
}

double impurity_reduction(
    double parent_impurity,
    double weighted_child_impurity
) {
    validate_impurity_value(parent_impurity, "impurity_reduction parent_impurity");
    validate_impurity_value(weighted_child_impurity, "impurity_reduction weighted_child_impurity");

    return parent_impurity - weighted_child_impurity;
}

void validate_decision_tree_options(
    const DecisionTreeOptions& options,
    const std::string& context
) {
    if (options.max_depth == 0) {
        throw std::invalid_argument(
            context + ": max_depth must be strictly positive"
        );
    }
    
    if (options.min_samples_split < 2) {
        throw std::invalid_argument(
            context + ": min_samples_split must be at least 2"
        );
    }
    
    if (options.min_samples_leaf < 1) {
        throw std::invalid_argument(
            context + ": min_samples_leaf must be at least 1"
        );
    }
    
    if (!std::isfinite(options.min_impurity_decrease)) {
        throw std::invalid_argument(
            context + ": min_impurity_decrease must be finite"
        );
    }
    
    if (options.min_impurity_decrease < 0.0) {
        throw std::invalid_argument(
            context + ": min_impurity_decrease must be non-negative"
        );
    }
}

SplitCandidate evaluate_candidate_threshold(
    const Matrix& X,
    const Vector& y,
    Eigen::Index feature_index,
    double threshold,
    const DecisionTreeOptions& options
) {
    validate_decision_tree_options(
        options,
        "evaluate_candidate_threshold"
    );
    
    validate_non_empty_matrix(X, "evaluate_candidate_threshold");
    validate_non_empty_vector(y, "evaluate_candidate_threshold");
    validate_same_number_of_rows(X, y, "evaluate_candidate_threshold");

    if (feature_index < 0 || feature_index >= X.cols()) {
        throw std::invalid_argument(
            "evaluate_candidate_threshold: feature_index is out of range"
        );
    }

    if (!std::isfinite(threshold)) {
        throw std::invalid_argument(
            "evaluate_candidate_threshold: threshold must be finite"
        );
    }

    SplitCandidate candidate;
    candidate.feature_index = feature_index;
    candidate.threshold = threshold;
    candidate.parent_impurity = gini_impurity(y);

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

    Vector left_y(static_cast<Eigen::Index>(left_count));
    Vector right_y(static_cast<Eigen::Index>(right_count));

    Eigen::Index left_index = 0;
    Eigen::Index right_index = 0;

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        if (X(i, feature_index) <= threshold) {
            left_y(left_index) = y(i);
            ++left_index;
        } else {
            right_y(right_index) = y(i);
            ++right_index;
        }
    }

    candidate.left_impurity = gini_impurity(left_y);
    candidate.right_impurity = gini_impurity(right_y);

    candidate.weighted_child_impurity = weighted_child_impurity(
        candidate.left_impurity,
        candidate.right_impurity,
        candidate.left_count,
        candidate.right_count
    );

    candidate.impurity_decrease = impurity_reduction(
        candidate.parent_impurity,
        candidate.weighted_child_impurity
    );

    if (candidate.impurity_decrease < options.min_impurity_decrease) {
        return candidate;
    }

    candidate.valid = true;

    return candidate;
}

SplitCandidate find_best_split(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options
) {
    validate_decision_tree_options(options, "find_best_split");
    
    validate_non_empty_matrix(X, "find_best_split");
    validate_non_empty_vector(y, "find_best_split");
    validate_same_number_of_rows(X, y, "find_best_split");

    SplitCandidate best;

    for (Eigen::Index feature_index = 0; feature_index < X.cols(); ++feature_index) {
        const std::vector<double> unique_values = sorted_unique_feature_values(X, feature_index);

        const std::vector<double> thresholds = midpoint_thresholds(unique_values);

        for (double threshold : thresholds) {
            const SplitCandidate candidate = evaluate_candidate_threshold(
                X,
                y,
                feature_index,
                threshold,
                options
            );

            if (is_better_split(candidate, best)) {
                best = candidate;
            }
        }
    }

    return best;
}

}  // namespace ml