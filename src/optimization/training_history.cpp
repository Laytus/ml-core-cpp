#include "ml/optimization/training_history.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace ml {

bool TrainingHistory::has_losses() const {
    return !losses.empty();
}

double TrainingHistory::initial_loss() const {
    if (losses.empty()) {
        throw std::invalid_argument(
            "TrainingHistory::initial_loss: loss history is empty"
        );
    }

    return losses.front();
}

double TrainingHistory::final_loss() const {
    if (losses.empty()) {
        throw std::invalid_argument(
            "TrainingHistory::final_loss: loss history is empty"
        );
    }

    return losses.back();
}

double TrainingHistory::best_loss() const {
    if (losses.empty()) {
        throw std::invalid_argument(
            "TrainingHistory::best_loss: loss history is empty"
        );
    }

    return *std::min_element(losses.begin(), losses.end());
}

double TrainingHistory::loss_improvement() const {
    if (losses.empty()) {
        throw std::invalid_argument(
            "TrainingHistory::loss_improvement: loss history is empty"
        );
    }

    return initial_loss() - final_loss();
}

const char* optimization_stop_reason_name(
    OptimizationStopReason reason
) {
    switch (reason) {
        case OptimizationStopReason::MaxIterationsReached:
            return "MaxIterationsReached";

        case OptimizationStopReason::MaxEpochsReached:
            return "MaxEpochsReached";

        case OptimizationStopReason::LossToleranceReached:
            return "LossToleranceReached";

        case OptimizationStopReason::GradientToleranceReached:
            return "GradientToleranceReached";

        case OptimizationStopReason::NonFiniteLoss:
            return "NonFiniteLoss";

        case OptimizationStopReason::Unknown:
            return "Unknown";
    }

    return "Unknown";
}

std::string training_history_summary(
    const TrainingHistory& history
) {
    std::ostringstream oss;

    oss << "Optimizer: " << history.optimizer_name << "\n"
        << "Learning rate: " << history.learning_rate << "\n"
        << "Momentum: " << history.momentum << "\n"
        << "Batch size: " << history.batch_size << "\n"
        << "Iterations run: " << history.iterations_run << "\n"
        << "Epochs run: " << history.epochs_run << "\n"
        << "Converged: " << (history.converged ? "true" : "false") << "\n"
        << "Stop reason: " << optimization_stop_reason_name(history.stop_reason) << "\n";

    if (history.has_losses()) {
        oss << "Initial loss: " << history.initial_loss() << "\n"
            << "Final loss: " << history.final_loss() << "\n"
            << "Best loss: " << history.best_loss() << "\n"
            << "Loss improvement: " << history.loss_improvement() << "\n";
    } else {
        oss << "Loss history: empty\n";
    }

    if (!history.gradient_norms.empty()) {
        oss << "Initial gradient norm: " << history.gradient_norms.front() << "\n"
            << "Final gradient norm: " << history.gradient_norms.back() << "\n";
    } else {
        oss << "Gradient norm history: empty\n";
    }

    return oss.str();
}

}  // namespace ml