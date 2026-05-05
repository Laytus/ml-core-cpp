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
    
    Matrix X_right;
    Vector y_right;
};

double majority_class(const Vector& y);

bool is_pure_node(const Vector& y);

DatasetSplit split_dataset(
    const Matrix& X,
    const Vector& y,
    Eigen::Index feature_index,
    double threshold
);

std::unique_ptr<TreeNode> build_tree(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options,
    std::size_t depth = 0
);

}  // namespace ml