#include "ml/linear_models/linear_svm.hpp"

#include "ml/common/shape_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

namespace {

void validate_finite_matrix_values(
    const Matrix& X,
    const std::string& context
) {
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            if (!std::isfinite(X(i, j))) {
                throw std::invalid_argument(
                    context + ": X values must be finite"
                );
            }
        }
    }
}

void validate_binary_targets(
    const Vector& y,
    const std::string& context
) {
    validate_non_empty_vector(y, context);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double value = y(i);
        const double rounded = std::round(value);

        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                context + ": target values must be finite"
            );
        }

        if (std::abs(value - rounded) > 1e-12) {
            throw std::invalid_argument(
                context + ": target values must be integer-valued"
            );
        }

        if (!(rounded == 0.0 || rounded == 1.0)) {
            throw std::invalid_argument(
                context + ": target values must be binary labels 0 or 1"
            );
        }
    }
}

}  // namespace

void validate_linear_svm_options(
    const LinearSVMOptions& options,
    const std::string& context
) {
    if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0) {
        throw std::invalid_argument(
            context + ": learning_rate must be finite and positive"
        );
    }
    
    if (options.max_epochs == 0) {
        throw std::invalid_argument(
            context + ": max_epochs must be at least 1"
        );
    }
    
    if (!std::isfinite(options.l2_lambda) || options.l2_lambda < 0.0) {
        throw std::invalid_argument(
            context + ": l2_lambda must be finite and non-negative"
        );
    }
}

LinearSVM::LinearSVM(LinearSVMOptions options)
    : options_{options} {
    validate_linear_svm_options(options_, "LinearSVM");
}

void LinearSVM::fit(
    const Matrix& X,
    const Vector& y
) {
    validate_linear_svm_options(options_, "LinearSVM::fit");

    validate_non_empty_matrix(X, "LinearSVM::fit");
    validate_binary_targets(y, "LinearSVM::fit");
    validate_same_number_of_rows(X, y, "LinearSVM::fit");
    validate_finite_matrix_values(X, "LinearSVM::fit");

    num_features_ = X.cols();

    weights_ = Vector::Zero(num_features_);
    bias_ = 0.0;

    const Vector y_svm = map_binary_targets_to_svm_targets(y);

    training_loss_history_.clear();
    training_loss_history_.reserve(options_.max_epochs);

    for (std::size_t epoch = 0; epoch < options_.max_epochs; ++epoch) {
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            const Vector x_i = X.row(i).transpose();
            const double y_i = y_svm(i);

            const double score = weights_.dot(x_i) + bias_;

            const double margin = y_i * score;

            Vector grad_w = options_.l2_lambda * weights_;

            double grad_b = 0.0;

            if (margin < 1.0) {
                grad_w -= y_i * x_i;
                grad_b = -y_i;
            }

            weights_ -= options_.learning_rate * grad_w;
            bias_ -= options_.learning_rate * grad_b;
        }

        training_loss_history_.push_back(
            compute_objective_loss(X, y_svm)
        );
    }

    fitted_ = true;
}

Vector LinearSVM::decision_function(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "LinearSVM::decision_function: model must be fitted before prediction"
        );
    }
    
    validate_non_empty_matrix(X, "LinearSVM::decision_function");
    validate_finite_matrix_values(X, "LinearSVM::decision_function");
    
    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "LinearSVM::decision_function: X feature count must match training feature count"
        );
    }

    return X * weights_ + Vector::Constant(X.rows(), bias_);
}

Vector LinearSVM::predict(
    const Matrix& X
) const {
    const Vector scores = decision_function(X);

    Vector predictions(scores.size());

    for (Eigen::Index i = 0; i < scores.size(); ++i) {
        predictions(i) = scores(i) >= 0.0 ? 1.0 : 0.0;
    }

    return predictions;
}

bool LinearSVM::is_fitted() const {
    return fitted_;
}

const LinearSVMOptions& LinearSVM::options() const {
    return options_;
}

const Vector& LinearSVM::weights() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "LinearSVM::weights: model must be fitted"
        );
    }

    return weights_;
}

double LinearSVM::bias() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "LinearSVM::bias: model must be fitted"
        );
    }

    return bias_;
}

const std::vector<double>& LinearSVM::training_loss_history() const {
    return training_loss_history_;
}

Vector LinearSVM::map_binary_targets_to_svm_targets(
    const Vector& y
) const {
    validate_binary_targets(y, "LinearSVM::map_binary_targets_to_svm_targets");

    Vector mapped(y.size());

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        mapped(i) = y(i) == 1.0 ? 1.0 : -1.0;
    }

    return mapped;
}

double LinearSVM::compute_objective_loss(
    const Matrix& X,
    const Vector& y_svm
) const {
    validate_same_number_of_rows(X, y_svm, "LinearSVM::compute_objective_loss");

    double hinge_total = 0.0;

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const Vector x_i = X.row(i).transpose();

        const double score = weights_.dot(x_i) + bias_;

        const double margin = y_svm(i) * score;

        hinge_total += std::max(0.0, 1.0 - margin);
    }

    const double mean_hinge_loss = hinge_total / static_cast<double>(X.rows());
    
    const double regularization = 0.5 * options_.l2_lambda * weights_.squaredNorm();

    return mean_hinge_loss + regularization;
}

}  // namespace ml