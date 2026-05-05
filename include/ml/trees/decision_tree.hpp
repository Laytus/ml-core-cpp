#pragma once

#include "ml/common/types.hpp"
#include "ml/trees/split_scoring.hpp"
#include "ml/trees/tree_node.hpp"

#include <memory>

namespace ml {

class DecisionTreeClassifier {
public:
    DecisionTreeClassifier() = default;

    explicit DecisionTreeClassifier(DecisionTreeOptions options);

    void fit(
        const Matrix& X,
        const Vector& y
    );

    Vector predict(
        const Matrix& X
    ) const;

    bool is_fitted() const;

    const DecisionTreeOptions& options() const;

private: 
    double predict_one(
        const Vector& sample,
        const TreeNode& node
    ) const;

    DecisionTreeOptions options_{};
    std::unique_ptr<TreeNode> root_{nullptr};
    Eigen::Index num_features_{0};
};

}  // namespace ml