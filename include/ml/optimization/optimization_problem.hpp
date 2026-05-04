#pragma once

#include "ml/common/types.hpp"

namespace ml {

struct OptimizationBatch {
    Matrix X;
    Matrix y;
};

struct ParameterGradient {
    Matrix weights_gradient;
    Vector bias_gradient;
};

class OptimizationProblem {
public:
    virtual ~OptimizationProblem() = default;

    virtual Matrix weights() const = 0;

    virtual Vector bias() const = 0;

    virtual void set_parameters(
        const Matrix& weights,
        const Vector& bias
    ) = 0;

    virtual double loss(
        const Matrix& X,
        const Matrix& y
    ) const = 0;

    virtual ParameterGradient gradients(
        const Matrix& X,
        const Matrix& y
    ) const = 0;
};

}  // namespace ml