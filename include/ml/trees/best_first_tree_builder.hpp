#pragma once

#include "ml/common/types.hpp"
#include "ml/trees/split_scoring.hpp"
#include "ml/trees/tree_node.hpp"

#include <memory>

namespace ml {

std::unique_ptr<TreeNode> build_tree_best_first(
    const Matrix& X,
    const Vector& y,
    const DecisionTreeOptions& options
);

std::unique_ptr<TreeNode> build_tree_best_first(
    const Matrix& X,
    const Vector& y,
    const Vector& sample_weight,
    const DecisionTreeOptions& options
);

}  // namespace ml