#pragma once

#include "ml/common/shape_validation.hpp"

#include <cstddef>
#include <memory>

namespace ml {

struct TreeNode {
    bool is_leaf{true};

    double prediction{0.0};

    Eigen::Index feature_index{-1};
    double threshold{0.0};

    double impurity{0.0};
    double impurity_decrease{0.0};
    std::size_t num_samples{0};

    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
};

}  // namespace ml