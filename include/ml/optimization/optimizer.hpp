#pragma once

#include "ml/common/types.hpp"
#include "ml/optimization/optimization_problem.hpp"
#include "ml/optimization/training_history.hpp"

#include <cstddef>

namespace ml {

struct OptimizerOptions {
    double learning_rate{0.01};
    double momentum{0.0};

    std::size_t max_epochs{1000};
    std::size_t max_iterations{100000};
    std::size_t batch_size{32};

    unsigned int random_seed{42};
    bool shuffle{true};

    double loss_tolerance{1e-8};
    double gradient_tolerance{0.0};

    bool store_loss_history{true};
};

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual TrainingHistory optimize(
        OptimizationProblem& problem,
        const Matrix& X,
        const Matrix& y
    ) const = 0;

    virtual const char* name() const = 0;
};

}  // namespace ml