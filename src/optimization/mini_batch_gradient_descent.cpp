#include "ml/optimization/mini_batch_gradient_descent.hpp"

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
            "MiniBatchGradientDescent: learning_rate must be strictly greater than 0"
        );
    }

    if (options.max_epochs == 0) {
        throw std::invalid_argument(
            "MiniBatchGradientDescent: max_epochs must be strictly greater than 0"
        );
    }

    if (options.max_iterations == 0) {
        throw std::invalid_argument(
            "MiniBatchGradientDescent: max_iterations must be strictly greater than 0"
        );
    }

    if (options.batch_size == 0) {
        throw std::invalid_argument(
            "MiniBatchGradientDescent: batch_size must be strictly greater than 0"
        );
    }

    if (options.loss_tolerance < 0.0) {
        throw std::invalid_argument(
            "MiniBatchGradientDescent: loss_tolerance must be non-negative"
        );
    }

    if (options.gradient_tolerance < 0.0) {
        throw std::invalid_argument(
            "MiniBatchGradientDescent: gradient_tolerance must be non-negative"
        );
    }

    if (options.momentum < 0.0 || options.momentum >= 1.0) {
        throw std::invalid_argument(
            "MiniBatchGradientDescent: momentum must be in [0.0, 1.0)"
        );
    }
}

Matrix select_rows(
    const Matrix& matrix,
    const std::vector<Eigen::Index>& indices,
    std::size_t begin,
    std::size_t end
) {
    const std::size_t batch_size = end - begin;

    Matrix batch(
        static_cast<Eigen::Index>(batch_size),
        matrix.cols()
    );

    for (std::size_t i = 0; i < batch_size; ++i) {
        batch.row(static_cast<Eigen::Index>(i)) =
            matrix.row(indices[begin + i]);
    }

    return batch;
}

}  // namespace 

MiniBatchGradientDescent::MiniBatchGradientDescent(OptimizerOptions options)
    : options_{options} {}

TrainingHistory MiniBatchGradientDescent::optimize(
    OptimizationProblem& problem,
    const Matrix& X,
    const Matrix& y
) const {
    validate_optimizer_options(options_);

    validate_non_empty_matrix(X, "MiniBatchGradientDescent::optimize");
    validate_non_empty_matrix(y, "MiniBatchGradientDescent::optimize");
    validate_same_number_of_rows(X, y, "MiniBatchGradientDescent::optimize");

    TrainingHistory history;
    history.optimizer_name = name();
    history.learning_rate = options_.learning_rate;
    history.momentum = options_.momentum;
    history.batch_size = options_.batch_size;

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

        for (
            size_t batch_begin = 0;
            batch_begin < indices.size();
            batch_begin += options_.batch_size
        ) {
            if (iteration >= options_.max_iterations) {
                history.converged = false;
                history.stop_reason = OptimizationStopReason::MaxIterationsReached;
                return history;
            }

            const std::size_t batch_end = std::min(
                batch_begin + options_.batch_size,
                indices.size()
            );

            const Matrix X_batch = select_rows(
                X,
                indices,
                batch_begin,
                batch_end
            );

            const Matrix y_batch = select_rows(
                y,
                indices,
                batch_begin,
                batch_end
            );

            const double batch_loss = problem.loss(X_batch, y_batch);
    
            if (!std::isfinite(batch_loss)) {
                history.converged = false;
                history.stop_reason = OptimizationStopReason::NonFiniteLoss;
                return history;
            }

            const ParameterGradient gradient = problem.gradients(X_batch, y_batch);
            const double current_gradient_norm = gradient_norm(gradient);  
            
            if (options_.store_loss_history) {
                history.gradient_norms.push_back(current_gradient_norm);
            }
    
            if (
                options_.gradient_tolerance > 0.0 &&
                current_gradient_norm <= options_.gradient_tolerance
            ) {
                history.iterations_run = iteration + 1;
                history.epochs_run = epoch + 1;
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

const char* MiniBatchGradientDescent::name() const {
    return "MiniBatchGradientDescent";
}

const OptimizerOptions& MiniBatchGradientDescent::options() const {
    return options_;
}

}  // namespace ml