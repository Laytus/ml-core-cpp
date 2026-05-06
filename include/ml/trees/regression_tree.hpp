#pragma once

#include "ml/common/types.hpp"

#include <cstddef>
#include <memory>

namespace ml {

struct RegressionTreeOptions {
    std::size_t max_depth{2};
    std::size_t min_samples_split{2};
    std::size_t min_samples_leaf{1};
    double min_error_decrease{0.0};
};

void validate_regression_tree_options(
    const RegressionTreeOptions& options,
    const std::string& context
);

class DecisionTreeRegressor {
public:
    DecisionTreeRegressor() = default;

    explicit DecisionTreeRegressor(
        RegressionTreeOptions options
    );

    void fit(
        const Matrix& X,
        const Vector& y
    );
    
    Vector predict(
        const Matrix& X
    ) const;

    bool is_fitted() const;

    const RegressionTreeOptions& options() const;

private:
    struct Node {
        bool is_leaf{true};
        double prediction{0.0};
        Eigen::Index feature_index{-1};
        double threshold{0.0};
        std::size_t num_samples{0};
        double squared_error{0.0};
        double error_decrease{0.0};

        std::unique_ptr<Node> left{nullptr};
        std::unique_ptr<Node> right{nullptr};
    };

    std::unique_ptr<Node> build_node(
        const Matrix& X,
        const Vector& y,
        std::size_t depth
    ) const;

    double predict_one(
        const Vector& sample,
        const Node& node
    ) const;

    RegressionTreeOptions options_{};
    std::unique_ptr<Node> root_{nullptr};
    Eigen::Index num_features_{0};
};

}  // namespace ml