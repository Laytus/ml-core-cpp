#include "ml/optimization/sgd.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/optimization/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

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
            "StochasticGradientDescent: learning_rate must be strictly greater than 0"
        );
    }

    if (options.max_epochs == 0) {
        throw std::invalid_argument(
            "StochasticGradientDescent: max_epochs must be strictly greater than 0"
        );
    }

    if (options.max_iterations == 0) {
        throw std::invalid_argument(
            "StochasticGradientDescent: max_iterations must be strictly greater than 0"
        );
    }

    if (options.loss_tolerance < 0.0) {
        throw std::invalid_argument(
            "StochasticGradientDescent: loss_tolerance must be non-negative"
        );
    }

    if (options.gradient_tolerance < 0.0) {
        throw std::invalid_argument(
            "StochasticGradientDescent: gradient_tolerance must be non-negative"
        );
    }

    if (options.momentum < 0.0 || options.momentum >= 1.0) {
        throw std::invalid_argument(
            "StochasticGradientDescent: momentum must be in [0.0, 1.0)"
        );
    }
}

Matrix single_row_matrix(
    const Matrix& X,
    Eigen::Index row_index
) {
    Matrix row(1, X.cols());
    row.row(0) = X.row(row_index);
    return row;
}

Matrix single_target_row_matrix(
    const Matrix& y,
    Eigen::Index row_index
) {
    Matrix row(1, y.cols());
    row.row(0) = y.row(row_index);
    return row;
}

}  // namespace 

StochasticGradientDescent::StochasticGradientDescent(OptimizerOptions options)
    : options_{options} {}

TrainingHistory StochasticGradientDescent::optimize(
    OptimizationProblem& problem,
    const Matrix& X,
    const Matrix& y
) const {
    validate_optimizer_options(options_);

    validate_non_empty_matrix(X, "StochasticGradientDescent::optimize");
    validate_non_empty_matrix(y, "StochasticGradientDescent::optimize");
    validate_same_number_of_rows(X, y, "StochasticGradientDescent::optimize");

    TrainingHistory history;
    history.optimizer_name = name();
    history.learning_rate = options_.learning_rate;
    history.momentum = options_.momentum;
    history.batch_size = 1;

    std::vector<Eigen::Index> indices(static_cast<std::size_t>(X.rows()));
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 generator(options_.random_seed);

    bool has_previous_epoch_loss = false;
    double previous_epoch_loss = 0.0;

    Matrix velocity_weights = Matrix::Zero(
        problem.weights().rows(),
        problem.weights().cols()
    );

    Vector velocity_bias = Vector::Zero(problem.bias().size());

    std::size_t iteration = 0;

    for (std::size_t epoch = 0; epoch < options_.max_epochs; ++epoch) {
        if (options_.shuffle) {
            std::shuffle(indices.begin(), indices.end(), generator);
        }

        for (const Eigen::Index sample_index : indices) {
            if (iteration >= options_.max_iterations) {
                history.converged = false;
                history.stop_reason = OptimizationStopReason::MaxIterationsReached;
                return history;
            }

            const Matrix X_sample = single_row_matrix(X, sample_index);
            const Matrix y_sample = single_target_row_matrix(y, sample_index);

            const double sample_loss = problem.loss(X_sample, y_sample);
    
            if (!std::isfinite(sample_loss)) {
                history.converged = false;
                history.stop_reason = OptimizationStopReason::NonFiniteLoss;
                return history;
            }

            const ParameterGradient gradient = problem.gradients(X_sample, y_sample);
            const double current_gradient_norm = gradient_norm(gradient);  

            if (options_.store_loss_history) {
                history.gradient_norms.push_back(current_gradient_norm);
            }
    
            if (
                options_.gradient_tolerance > 0.0 &&
                current_gradient_norm <= options_.gradient_tolerance
            ) {
                history.iterations_run = iteration + 1;
                history.epochs_run= epoch + 1;
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

            ++iteration;
            history.iterations_run = iteration;
        }

        const double epoch_loss = problem.loss(X, y);

        if (!std::isfinite(epoch_loss)) {
            history.converged = false;
            history.stop_reason = OptimizationStopReason::NonFiniteLoss;
            return history;
        }

        if (options_.store_loss_history) {
            history.losses.push_back(epoch_loss);
        }

        history.epochs_run = epoch + 1;

        if (
            has_previous_epoch_loss &&
            std::abs(previous_epoch_loss - epoch_loss) <= options_.loss_tolerance
        ) {
            history.converged = true;
            history.stop_reason = OptimizationStopReason::LossToleranceReached;
            return history;
        }

        previous_epoch_loss = epoch_loss;
        has_previous_epoch_loss = true;
    }

    history.converged = false;
    history.stop_reason = OptimizationStopReason::MaxEpochsReached;

    return history;
}

const char* StochasticGradientDescent::name() const {
    return "StochasticGradientDescent";
}

const OptimizerOptions& StochasticGradientDescent::options() const {
    return options_;
}

}  // namespace ml