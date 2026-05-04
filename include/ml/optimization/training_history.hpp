#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ml {

enum class OptimizationStopReason {
    MaxIterationsReached,
    MaxEpochsReached,
    LossToleranceReached,
    GradientToleranceReached,
    NonFiniteLoss,
    Unknown
};

struct TrainingHistory {
    std::vector<double> losses;
    std::vector<double> gradient_norms;

    std::size_t iterations_run{0};
    std::size_t epochs_run{0};

    bool converged{false};
    OptimizationStopReason stop_reason{OptimizationStopReason::Unknown};

    std::string optimizer_name{};

    double learning_rate{0.0};
    double momentum{0.0};

    std::size_t batch_size{0};

    bool has_losses() const;
    double initial_loss() const;
    double final_loss() const;
    double best_loss() const;
    double loss_improvement() const;
};

const char* optimization_stop_reason_name(
    OptimizationStopReason reason
);

std::string training_history_summary(
    const TrainingHistory& history
);

}  // namespace ml