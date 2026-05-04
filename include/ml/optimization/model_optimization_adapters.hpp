#pragma once

#include "ml/common/types.hpp"
#include "ml/linear_models/regularization.hpp"
#include "ml/optimization/optimization_problem.hpp"

#include <Eigen/Dense>

namespace ml {

class LinearRegressionOptimizationProblem : public OptimizationProblem {
public:
    explicit LinearRegressionOptimizationProblem(
        Eigen::Index num_features,
        RegularizationConfig regularization = RegularizationConfig::none()
    );

    double loss(
        const Matrix& X,
        const Matrix& y
    ) const override;

    ParameterGradient gradients(
        const Matrix& X,
        const Matrix& y
    ) const override;

    void set_parameters(
        const Matrix& weights,
        const Vector& bias
    ) override;
    
    Matrix weights() const override;

    Vector bias() const override;

private:
    Matrix weights_;
    Vector bias_;
    RegularizationConfig regularization_;
};

class LogisticRegressionOptimizationProblem : public OptimizationProblem {
public:
    explicit LogisticRegressionOptimizationProblem(
        Eigen::Index num_features,
        RegularizationConfig regularization = RegularizationConfig::none()
    );

    double loss(
        const Matrix& X,
        const Matrix& y
    ) const override;
    
    ParameterGradient gradients(
        const Matrix& X,
        const Matrix& y
    ) const override;
    
    void set_parameters(
        const Matrix& weights,
        const Vector& bias
    ) override;

    Matrix weights() const override;

    Vector bias() const override;

private:
    Matrix weights_;
    Vector bias_;
    RegularizationConfig regularization_;
};

class SoftmaxRegressionOptimizationProblem : public OptimizationProblem {
public:
    explicit SoftmaxRegressionOptimizationProblem(
        Eigen::Index num_features,
        Eigen::Index num_classes,
        RegularizationConfig regularization = RegularizationConfig::none()
    );

    double loss(
        const Matrix& X,
        const Matrix& y
    ) const override;

    ParameterGradient gradients(
        const Matrix& X,
        const Matrix& y
    ) const override;
    
    void set_parameters(
        const Matrix& weights,
        const Vector& bias
    ) override;

    Matrix weights() const override;

    Vector bias() const override;

    Eigen::Index num_classes() const;

private:
    Matrix weights_;
    Vector bias_;
    Eigen::Index num_classes_{0};
    RegularizationConfig regularization_;
};

}  // namespace ml