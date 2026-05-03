#pragma once

#include "ml/common/types.hpp"
#include "ml/linear_models/regularization.hpp"

#include <cstddef>
#include <vector>

namespace ml {

struct LogisticRegressionOptions {
    double learning_rate{0.01};
    std::size_t max_iterations{1000};
    double tolerance{1e-8};
    RegularizationConfig regularization{RegularizationConfig::none()};
    bool store_loss_history{true};
};

struct LogisticRegressionTrainingHistory {
    std::vector<double> losses;
    std::size_t iterations_run{0};
    bool converged{false};
};

class LogisticRegression {
public:
    LogisticRegression() = default;

    explicit LogisticRegression(LogisticRegressionOptions options);

    void fit (
        const Matrix& X,
        const Vector& y
    );

    Vector logits(const Matrix& X) const;
    
    Vector predict_proba(const Matrix& X) const;

    Vector predict_classes(
        const Matrix& X,
        const double threshold = 0.5
    ) const;

    double binary_cross_entropy(
        const Matrix& X,
        const Vector& y
    ) const;

    const Vector& weights() const;

    double bias() const;

    bool is_fitted() const;

    const LogisticRegressionTrainingHistory& training_history() const;

    const LogisticRegressionOptions& options() const;

private:
    LogisticRegressionOptions options_{};
    LogisticRegressionTrainingHistory history_{};
    Vector weights_{};
    double bias_{0.0};
    bool fitted_{false};
};

}  // namespace ml