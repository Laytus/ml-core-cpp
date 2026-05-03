#include "ml/linear_models/logistic_regression.hpp"

#include "ml/common/classification_utils.hpp"
#include "ml/common/shape_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

namespace {

void validate_options(const LogisticRegressionOptions& options) {
    if (options.learning_rate <= 0.0) {
        throw std::invalid_argument(
            "LogisticRegression::fit: learning_rate must be strictly greater than 0"
        );
    }
    
    if (options.max_iterations == 0) {
        throw std::invalid_argument(
            "LogisticRegression::fit: max_iteractions must be strictly greater than 0"
        );
    }
    
    if (options.tolerance < 0.0) {
        throw std::invalid_argument(
            "LogisticRegression::fit: tolerance must be non-negative"
        );
    }
    
    if (options.regularization.lambda < 0.0) {
        throw std::invalid_argument(
            "LogisticRegression::fit: regularization lambda must be non-negative"
        );
    }

    if (options.regularization.is_lasso()) {
        throw std::invalid_argument(
            "LogisticRegression::fit: Lasso regularization is not implemented yet"
        );
    }

    if (
        options.regularization.type != RegularizationType::None &&
        options.regularization.type != RegularizationType::Ridge 
    ) {
        throw std::invalid_argument(
            "LogisticRegression::fit: regularization type must be None or Ridge"
        );
    }
}

double logistic_loss_with_regularization(
    const Vector& probabilities,
    const Vector& targets,
    const Vector& weights,
    const RegularizationConfig& regularization
) {
    double loss = binary_cross_entropy(probabilities, targets);

    if (regularization.is_ridge()) {
        loss += regularization.lambda * weights.squaredNorm();
    }

    return loss;
}

}  // namespace

LogisticRegression::LogisticRegression(LogisticRegressionOptions options)
    : options_{options} {}

void LogisticRegression::fit (
    const Matrix& X,
    const Vector& y
) {
    validate_options(options_);

    validate_non_empty_matrix(X, "LogisticRegression::fit");
    validate_non_empty_vector(y, "LogisticRegression::fit");
    validate_same_number_of_rows(X, y, "LogisticRegression::fit");

    const auto num_samples = static_cast<double>(X.rows());

    weights_ = Vector::Zero(X.cols());
    bias_ = 0.0;
    fitted_ = false;
    history_ = LogisticRegressionTrainingHistory{};

    bool has_previous_loss = false;
    double previous_loss = 0.0;

    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        Vector current_logits = X * weights_;
        current_logits.array() += bias_;

        const Vector probabilities = sigmoid(current_logits);
        const Vector errors = probabilities - y;
    
        double loss = logistic_loss_with_regularization(
            probabilities,
            y,
            weights_,
            options_.regularization
        );
    
        Vector gradient_w = (1.0 / num_samples) * (X.transpose() * errors);
        const double gradient_b = (1.0 / num_samples) * errors.sum();
    
        if (options_.regularization.is_ridge()) {
            gradient_w += 2.0 * options_.regularization.lambda * weights_;
        }
    
        if (options_.store_loss_history) {
            history_.losses.push_back(loss);
        }

        history_.iterations_run = iteration + 1;

        if (
            has_previous_loss &&
            std::abs(previous_loss - loss) <= options_.tolerance
        ) {
            history_.converged = true;
            break;
        }

        previous_loss = loss;
        has_previous_loss = true;

        weights_ -= options_.learning_rate * gradient_w;
        bias_ -= options_.learning_rate * gradient_b;
    }

    fitted_ = true;
}

Vector LogisticRegression::logits(const Matrix& X) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "LogisticRegression::logits: model must be fitted before prediction"
        );
    }

    validate_non_empty_matrix(X, "LogisticRegression::logits");
    validate_feature_count(X, weights_, "LogisticRegression::logits");

    Vector result = X * weights_;
    result.array() += bias_;

    return result;
}

Vector LogisticRegression::predict_proba(const Matrix& X) const {
    return sigmoid(logits(X));
}

Vector LogisticRegression::predict_classes(
    const Matrix& X,
    const double threshold
) const {
    validate_probability_threshold(
        threshold,
        "LogisticRegression::predict_classes"
    );

    return threshold_probabilities(
        predict_proba(X),
        threshold
    );
}

double LogisticRegression::binary_cross_entropy(
    const Matrix& X,
    const Vector& y
) const {
    const Vector probabilities = predict_proba(X);

    validate_binary_targets(y, "LogisticRegression::binary_cross_entropy");
    validate_same_size(
        probabilities,
        y,
        "LogisticRegression::binary_cross_entropy"
    );

    return ml::binary_cross_entropy(probabilities, y);
}

const Vector& LogisticRegression::weights() const {
    return weights_;
}

double LogisticRegression::bias() const {
    return bias_;
}

bool LogisticRegression::is_fitted() const {
    return fitted_;
}

const LogisticRegressionTrainingHistory& LogisticRegression::training_history() const {
    return history_;
}

const LogisticRegressionOptions& LogisticRegression::options() const {
    return options_;
}

}  // namespace ml