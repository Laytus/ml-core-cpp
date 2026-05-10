#pragma once

#include "ml/common/types.hpp"

#include <stddef.h>
#include <string>

namespace ml {

struct GaussianNaiveBayesOptions {
    double variance_smoothing{1e-9};
};

void validate_gaussian_naive_bayes_options(
    const GaussianNaiveBayesOptions& options,
    const std::string& context
);

class GaussianNaiveBayes {
public:
    GaussianNaiveBayes() = default;

    explicit GaussianNaiveBayes(
        GaussianNaiveBayesOptions options
    );

    void fit(
        const Matrix& X,
        const Vector& y
    );

    Vector predict(
        const Matrix& X
    ) const;

    Matrix predict_proba(
        const Matrix& X
    ) const;

    Matrix predict_log_proba(
        const Matrix& X
    ) const;

    bool is_fitted() const;

    const GaussianNaiveBayesOptions& options() const;

    const Vector& classes() const;
    
    const Vector& class_priors() const;
    
    const Matrix& means() const;
    
    const Matrix& variances() const;

    Eigen::Index num_features() const;

private:
    Matrix compute_joint_log_likelihood(
        const Matrix& X
    ) const;

    GaussianNaiveBayesOptions options_{};

    Vector classes_{};
    Vector class_priors_{};
    Matrix means_{};
    Matrix variances_{};

    Eigen::Index num_features_{0};
    bool fitted_{false};
};

}  // namespace ml