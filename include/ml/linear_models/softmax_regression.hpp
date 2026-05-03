#pragma once

#include "ml/common/types.hpp"
#include "ml/linear_models/regularization.hpp"

#include <cstddef>
#include <vector>

namespace ml {

struct SoftmaxRegressionOptions {
    double learning_rate{0.01};
    std::size_t max_iterations{1000};
    double tolerance{1e-8};
    RegularizationConfig regularization{RegularizationConfig::none()};
    bool store_loss_history{true};
};

struct SoftmaxRegressionTrainingHistory {
    std::vector<double> losses;
    std::size_t iterations_run{0};
    bool converged{false};
};

class SoftmaxRegression {
public:
    SoftmaxRegression() = default;

    explicit SoftmaxRegression(SoftmaxRegressionOptions options);

    void fit(
        const Matrix& X,
        const Vector& y,
        Eigen::Index num_classes
    );

    Matrix logits(const Matrix& X) const;

    Matrix predict_proba(const Matrix& X) const;

    Vector predict_classes(const Matrix& X) const;

    double categorical_cross_entropy(
        const Matrix& X,
        const Vector& y
    ) const;

    const Matrix& weights() const;

    const Vector& bias() const;

    Eigen::Index num_classes() const;

    bool is_fitted() const;

    const SoftmaxRegressionTrainingHistory& training_history() const;

    const SoftmaxRegressionOptions& options() const;

private:
    SoftmaxRegressionOptions options_{};
    SoftmaxRegressionTrainingHistory history_{};
    Matrix weights_{};
    Vector bias_{};
    Eigen::Index num_classes_{0};
    bool fitted_{false};
};

}  // namespace ml