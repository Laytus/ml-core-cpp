#pragma once

#include "ml/optimization/optimizer.hpp"

namespace ml {

class StochasticGradientDescent : public Optimizer {
public:
    StochasticGradientDescent() = default;

    explicit StochasticGradientDescent(OptimizerOptions options);

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