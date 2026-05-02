#pragma once

#include "ml/common/types.hpp"
#include "ml/linear_models/regularization.hpp"

#include <cstddef>
#include <vector>

namespace ml {

struct LinearRegressionOptions {
    double learning_rate{0.01};
    std::size_t max_iterations{1000};
    double tolerance{1e-8};
    RegularizationConfig regularization{RegularizationConfig::none()};
    bool store_loss_history{true};
};

struct LinearRegressionTrainingHistory {
    std::vector<double> losses;
    std::size_t iterations_run{0};
    bool converged{false};
};

class LinearRegression {
public:
    LinearRegression() = default;

    explicit LinearRegression(LinearRegressionOptions options);

    void fit (
        const Matrix& X,
        const Vector& y
    );

    Vector predict(const Matrix& X) const;

    double score_mse(
        const Matrix& X,
        const Vector& y
    );

    const Vector& weights() const;

    double bias() const;

    bool is_fitted() const;

    const LinearRegressionTrainingHistory& training_history() const;

    const LinearRegressionOptions& options() const;

private:
    LinearRegressionOptions options_{};
    LinearRegressionTrainingHistory history_{};
    Vector weights_{};
    double bias_{0.0};
    bool fitted_{false};
};

}  // namespace ml