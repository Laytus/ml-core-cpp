#pragma once

#include "ml/common/types.hpp"

#include <cstddef>

namespace ml {

struct DecisionTreeOptions {
    std::size_t max_depth{3};
    std::size_t min_samples_split{2};
    std::size_t min_samples_leaf{1};
    double min_impurity_decrease{0.0};
};

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

void validate_decision_tree_options(
    const DecisionTreeOptions& options,
    const std::string& context
);

SplitCandidate evaluate_candidate_threshold(
    const Matrix& X,
    const Vector& y,
    Eigen::Index feature_index,
    double threshold,
    const DecisionTreeOptions& options = DecisionTreeOptions{}
);

SplitCandidate find_best_split(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options = DecisionTreeOptions{}
);

}  // namespace ml