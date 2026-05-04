#include "ml/optimization/gradient_descent.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/optimization/optimizer.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

namespace {

double gradient_norm(const ParameterGradient& gradient) {
    return std::sqrt(
        gradient.weights_gradient.squaredNorm() +
        gradient.bias_gradient.squaredNorm()
    );
}

void validate_optimizer_options(const OptimizerOptions& options) {
    if (options.learning_rate <= 0.0) {
        throw std::invalid_argument(
            "BatchGradientDescent: learning_rate must be strictly greater than 0"
        );
    }

    if (options.max_epochs == 0) {
        throw std::invalid_argument(
            "BatchGradientDescent: max_epochs must be strictly greater than 0"
        );
    }

    if (options.max_iterations == 0) {
        throw std::invalid_argument(
            "BatchGradientDescent: max_iterations must be strictly greater than 0"
        );
    }

    if (options.loss_tolerance < 0.0) {
        throw std::invalid_argument(
            "BatchGradientDescent: loss_tolerance must be non-negative"
        );
    }

    if (options.gradient_tolerance < 0.0) {
        throw std::invalid_argument(
            "BatchGradientDescent: gradient_tolerance must be non-negative"
        );
    }

    if (options.momentum < 0.0 || options.momentum >= 1.0) {
        throw std::invalid_argument(
            "BatchGradientDescent: momentum must be in [0.0, 1.0)"
        );
    }
}

}  // namespace 

BatchGradientDescent::BatchGradientDescent(OptimizerOptions options)
    : options_{options} {}

TrainingHistory BatchGradientDescent::optimize(
    OptimizationProblem& problem,
    const Matrix& X,
    const Matrix& y
) const {
    validate_optimizer_options(options_);

    validate_non_empty_matrix(X, "BatchGradientDescent::optimize");
    validate_non_empty_matrix(y, "BatchGradientDescent::optimize");
    validate_same_number_of_rows(X, y, "BatchGradientDescent::optimize");

    TrainingHistory history;
    history.optimizer_name = name();
    history.learning_rate = options_.learning_rate;
    history.momentum = options_.momentum;
    history.batch_size = static_cast<std::size_t>(X.rows());

    bool has_previous_loss = false;
    double previous_loss = 0.0;

    Matrix velocity_weights = Matrix::Zero(
        problem.weights().rows(),
        problem.weights().cols()
    );
    
    Vector velocity_bias = Vector::Zero(problem.bias().size());

    const std::size_t max_steps = std::min(options_.max_epochs, options_.max_iterations);

    for (std::size_t step = 0; step < max_steps; ++step) {
        const double current_loss = problem.loss(X, y);

        history.iterations_run = step + 1;
        history.epochs_run = step + 1;

        if (!std::isfinite(current_loss)) {
            history.converged = false;
            history.stop_reason = OptimizationStopReason::NonFiniteLoss;
            return history;
        }

        if (options_.store_loss_history) {
            history.losses.push_back(current_loss);
        }

        if (
            has_previous_loss &&
            std::abs(previous_loss - current_loss) <= options_.loss_tolerance
        ) {
            history.converged = true;
            history.stop_reason = OptimizationStopReason::LossToleranceReached;
            return history;
        }

        const ParameterGradient gradient = problem.gradients(X, y);
        const double current_gradient_norm = gradient_norm(gradient);

        if (options_.store_loss_history) {
            history.gradient_norms.push_back(current_gradient_norm);
        }

        if (
            options_.gradient_tolerance > 0.0 &&
            current_gradient_norm <= options_.gradient_tolerance
        ) {
            history.converged = true;
            history.stop_reason = OptimizationStopReason::GradientToleranceReached;
            return history;
        }

        velocity_weights = 
            options_.momentum * velocity_weights +
            gradient.weights_gradient;

        velocity_bias = 
            options_.momentum * velocity_bias +
            gradient.bias_gradient;

        const Matrix new_weights = 
            problem.weights() - 
            options_.learning_rate * velocity_weights;

        const Vector new_bias = 
            problem.bias() - 
            options_.learning_rate * velocity_bias;

        problem.set_parameters(new_weights, new_bias);

        previous_loss = current_loss;
        has_previous_loss = true;
    }

    history.converged = false;

    if (history.epochs_run >= options_.max_epochs) {
        history.stop_reason = OptimizationStopReason::MaxEpochsReached;
    } else {
        history.stop_reason = OptimizationStopReason::MaxIterationsReached;
    }

    return history;
}

const char* BatchGradientDescent::name() const {
    return "BatchGradientDescent";
}

const OptimizerOptions& BatchGradientDescent::options() const {
    return options_;
}

}  // namespace ml