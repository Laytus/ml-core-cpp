#pragma once

#include "ml/common/types.hpp"
#include "ml/trees/split_scoring.hpp"
#include "ml/trees/tree_node.hpp"

#include <cstddef>
#include <memory>

namespace ml {

struct DatasetSplit {
    Matrix X_left;
    Vector y_left;
    Vector sample_weight_left;
    
    Matrix X_right;
    Vector y_right;
    Vector sample_weight_right;
};

double majority_class(const Vector& y);

double majority_class(
    const Vector& y,
    const Vector& sample_weight
);

bool is_pure_node(const Vector& y);

DatasetSplit split_dataset(
    const Matrix& X,
    const Vector& y,
    Eigen::Index feature_index,
    double threshold
);

DatasetSplit split_dataset(
    const Matrix& X,
    const Vector& y,
    const Vector& sample_weight,
    Eigen::Index feature_index,
    double threshold
);

std::unique_ptr<TreeNode> build_tree(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options,
    std::size_t depth = 0
);

std::unique_ptr<TreeNode> build_tree(
    const Matrix& X,
    const Vector& y,
    const Vector& sample_weight,
    const DecisionTreeOptions& options,
    std::size_t depth = 0
);

}  // namespace ml