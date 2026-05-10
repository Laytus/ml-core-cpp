#include "ml/dl_bridge/perceptron.hpp"

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

void validate_perceptron_options(
    const PerceptronOptions& options,
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
}

Perceptron::Perceptron(
    PerceptronOptions options
)
    : options_{options} {
    validate_perceptron_options(options_, "Perceptron");
}

void Perceptron::fit(
    const Matrix& X,
    const Vector& y
) {
    validate_perceptron_options(options_, "Perceptron::fit");

    validate_non_empty_matrix(X, "Perceptron::fit");
    validate_binary_targets(y, "Perceptron::fit");
    validate_same_number_of_rows(X, y, "Perceptron::fit");
    validate_finite_matrix_values(X, "Perceptron::fit");

    num_features_ = X.cols();
    weights_ = Vector::Zero(num_features_);
    bias_ = 0.0;

    mistake_history_.clear();
    mistake_history_.reserve(options_.max_epochs);

    for (std::size_t epoch = 0; epoch < options_.max_epochs; ++epoch) {
        std::size_t mistakes = 0;

        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            const Vector x_i = X.row(i).transpose();
            const double y_signed = map_binary_label_to_signed(y(i), "Perceptron::fit");

            const double score = weights_.dot(x_i) + bias_;

            if (y_signed * score <= 0.0) {
                weights_ += options_.learning_rate * y_signed * x_i;

                bias_ += options_.learning_rate * y_signed;

                ++mistakes;
            }
        }

        mistake_history_.push_back(static_cast<double>(mistakes));

        if (mistakes == 0) {
            break;
        }
    }

    fitted_ = true;
}

Vector Perceptron::decision_function(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "Perceptron::decision_function: model must be fitted before prediction"
        );
    }

    validate_non_empty_matrix(X, "Perceptron::decision_function");
    validate_finite_matrix_values(X, "Perceptron::decision_function");

    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "Perceptron::decision_function: X feature count must match training feature count"
        );
    }

    return X * weights_ + Vector::Constant(X.rows(), bias_);
}

Vector Perceptron::predict(
    const Matrix& X
) const {
    const Vector scores = decision_function(X);

    Vector predictions(scores.size());

    for (Eigen::Index i = 0; i < scores.size(); ++i) {
        predictions(i) = scores(i) >= 0.0 ? 1.0 : 0.0;
    }

    return predictions;
}

bool Perceptron::is_fitted() const {
    return fitted_;
}

const PerceptronOptions& Perceptron::options() const {
    return options_;
}

const Vector& Perceptron::weights() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "Perceptron::weights: model must be fitted"
        );
    }

    return weights_;
}

double Perceptron::bias() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "Perceptron::bias: model must be fitted"
        );
    }

    return bias_;
}

const std::vector<double>& Perceptron::mistake_history() const {
    return mistake_history_;
}

Eigen::Index Perceptron::num_features() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "Perceptron::num_features: model must be fitted"
        );
    }

    return num_features_;
}

double Perceptron::map_binary_label_to_signed(
    double label,
    const std::string& context
) const {
    const double rounded = std::round(label);

    if (!std::isfinite(label)) {
        throw std::invalid_argument(
            context + ": target values must be finite"
        );
    }

    if (std::abs(label - rounded) > 1e-12) {
        throw std::invalid_argument(
            context + ": target values must be integer-valued"
        );
    }
    
    if (rounded == 0.0) {
        return -1.0;
    }
    
    if (rounded == 1.0) {
        return 1.0;
    }
    
    throw std::invalid_argument(
        context + ": target values must be binary labels of 0 or 1"
    );
}


}  // namespace ml