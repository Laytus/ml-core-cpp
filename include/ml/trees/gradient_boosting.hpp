#pragma once

#include "ml/common/types.hpp"
#include "ml/trees/regression_tree.hpp"

#include <cstddef>
#include <vector>

namespace ml {

struct GradientBoostingRegressorOptions {
    std::size_t n_estimators{100};
    double learning_rate{0.1};
    std::size_t max_depth{2};
    std::size_t min_samples_split{2};
    std::size_t min_samples_leaf{1};
    unsigned int random_seed{42};
};

void validate_gradient_boosting_regressor_options(
    const GradientBoostingRegressorOptions& options,
    const std::string& context
);

class GradientBoostingRegressor {
public:
    GradientBoostingRegressor() = default;

    explicit GradientBoostingRegressor(
        GradientBoostingRegressorOptions options
    );

    void fit(
        const Matrix& X,
        const Vector& y
    );

    Vector predict(
        const Matrix& X
    ) const;

    bool is_fitted() const;

    const GradientBoostingRegressorOptions& options() const;

    double initial_prediction() const;

    std::size_t num_trees() const;

    const std::vector<double>& training_loss_history() const;

private:
    GradientBoostingRegressorOptions options_{};
    double initial_prediction_{0.0};
    std::vector<DecisionTreeRegressor> trees_{};
    std::vector<double> training_loss_history_{};

    Eigen::Index num_features_{0};
    bool fitted_{false};
};

}  // namespace ml