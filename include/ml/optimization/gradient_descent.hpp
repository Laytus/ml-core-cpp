#pragma once

#include "ml/optimization/optimizer.hpp"

namespace ml {

class BatchGradientDescent : public Optimizer {
public:
    BatchGradientDescent() = default;

    explicit BatchGradientDescent(OptimizerOptions options);

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