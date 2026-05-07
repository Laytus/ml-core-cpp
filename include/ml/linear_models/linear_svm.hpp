#pragma once

#include "ml/common/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ml {

struct LinearSVMOptions {
    double learning_rate{0.01};
    std::size_t max_epochs{100};
    double l2_lambda{0.01};
};

void validate_linear_svm_options(
    const LinearSVMOptions& options,
    const std::string& context
);

class LinearSVM {
public:
    LinearSVM() = default;

    explicit LinearSVM(
        LinearSVMOptions options
    );

    void fit(
        const Matrix& X,
        const Vector& y
    );

    Vector decision_function(
        const Matrix& X
    ) const;
    
    Vector predict(
        const Matrix& X
    ) const;

    bool is_fitted() const;

    const LinearSVMOptions& options() const;

    const Vector& weights() const;

    double bias() const;

    const std::vector<double>& training_loss_history() const;

private:
    Vector map_binary_targets_to_svm_targets(
        const Vector& y
    ) const;

    double compute_objective_loss(
        const Matrix& X,
        const Vector& y_svm
    ) const;

    LinearSVMOptions options_{};
    Vector weights_{};
    double bias_{0.0};
    std::vector<double> training_loss_history_{};

    Eigen::Index num_features_{0};
    bool fitted_{false};
};

}  // namespace ml