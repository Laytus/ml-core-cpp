#pragma once

#include "ml/common/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ml {

struct PerceptronOptions {
    double learning_rate{0.1};
    std::size_t max_epochs{100};
};

void validate_perceptron_options(
    const PerceptronOptions& options,
    const std::string& context
);

class Perceptron {
public:
    Perceptron() = default;

    explicit Perceptron(
        PerceptronOptions options
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

    const PerceptronOptions& options() const;

    const Vector& weights() const;

    double bias() const;

    const std::vector<double>& mistake_history() const;

    Eigen::Index num_features() const;

private:
    double map_binary_label_to_signed(
        double label,
        const std::string& context
    ) const;

    PerceptronOptions options_{};

    Vector weights_{};
    double bias_{0.0};

    std::vector<double> mistake_history_{};

    Eigen::Index num_features_{0};
    bool fitted_{false};
};

}  // namespace ml