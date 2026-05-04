#pragma once

#include "ml/optimization/optimizer.hpp"

#include <cstddef>

namespace ml {

class MiniBatchGradientDescent : public Optimizer {
public:
    MiniBatchGradientDescent() = default;

    explicit MiniBatchGradientDescent(OptimizerOptions options);

    TrainingHistory optimize(
        OptimizationProblem& problem,
        const Matrix& X,
        const Matrix& y
    ) const override;

    const char* name() const override;

    const OptimizerOptions& options() const;

private:
    OptimizerOptions options_{};
};

}  // namespace ml