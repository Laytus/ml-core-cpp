#include "ml/linear_models/softmax_regression.hpp"

#include "ml/common/classification_utils.hpp"
#include "ml/common/shape_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

namespace {

void validate_options(const SoftmaxRegressionOptions& options) {
    if (options.learning_rate <= 0.0) {
        throw std::invalid_argument(
            "SoftmaxRegression::fit: learning_rate must be strictly greater than 0"
        );
    }
    
    if (options.max_iterations == 0) {
        throw std::invalid_argument(
            "SoftmaxRegression::fit: max_iterations must be strictly greater than 0"
        );
    }
    
    if (options.tolerance < 0.0) {
        throw std::invalid_argument(
            "SoftmaxRegression::fit: tolerance must be non-negative"
        );
    }
    
    if (options.regularization.lambda < 0.0) {
        throw std::invalid_argument(
            "SoftmaxRegression::fit: regularization lambda must be non-negative"
        );
    }

    if (options.regularization.is_lasso()) {
        throw std::invalid_argument(
            "SoftmaxRegression::fit: Lasso regularization is not implemented yet"
        );
    }

    if (
        options.regularization.type != RegularizationType::None &&
        options.regularization.type != RegularizationType::Ridge 
    ) {
        throw std::invalid_argument(
            "SoftmaxRegression::fit: regularization type must be None or Ridge"
        );
    }
}

double softmax_loss_with_regularization(
    const Matrix& probabilities,
    const Matrix& targets,
    const Matrix& weights,
    const RegularizationConfig& regularization
) {
    double loss = categorical_cross_entropy(probabilities, targets);

    if (regularization.is_ridge()) {
        loss += regularization.lambda * weights.squaredNorm();
    }

    return loss;
}

}  // namespace

SoftmaxRegression::SoftmaxRegression(SoftmaxRegressionOptions options)
    : options_{options} {}

void SoftmaxRegression::fit (
    const Matrix& X,
    const Vector& y,
    Eigen::Index num_classes
) {
    validate_options(options_);

    validate_non_empty_matrix(X, "SoftmaxRegression::fit");
    validate_non_empty_vector(y, "SoftmaxRegression::fit");
    validate_same_number_of_rows(X, y, "SoftmaxRegression::fit");
    validate_class_indices(y, num_classes, "SoftmaxRegression::fit");

    const Matrix Y = one_hot_encode(y, num_classes);

    const auto num_samples = static_cast<double>(X.rows());

    weights_ = Matrix::Zero(X.cols(), num_classes);
    bias_ = Vector::Zero(num_classes);
    num_classes_ = num_classes;
    fitted_ = false;
    history_ = SoftmaxRegressionTrainingHistory{};

    bool has_previous_loss = false;
    double previous_loss = 0.0;

    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        Matrix current_logits = X * weights_;
        current_logits.rowwise() += bias_.transpose();

        const Matrix probabilities = softmax_rows(current_logits);
        const Matrix errors = probabilities - Y;
    
        double loss = softmax_loss_with_regularization(
            probabilities,
            Y,
            weights_,
            options_.regularization
        );
    
        Matrix gradient_w = (1.0 / num_samples) * (X.transpose() * errors);
        Vector gradient_b = (1.0 / num_samples) * errors.colwise().sum().transpose();
    
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

Matrix SoftmaxRegression::logits(const Matrix& X) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "SoftmaxRegression::logits: model must be fitted before prediction"
        );
    }

    validate_non_empty_matrix(X, "SoftmaxRegression::logits");
    
    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument(
            "SoftmaxRegression::logits: X feature count must match weights rows"
        );
    }

    Matrix result = X * weights_;
    result.rowwise() += bias_.transpose();

    return result;
}

Matrix SoftmaxRegression::predict_proba(const Matrix& X) const {
    return softmax_rows(logits(X));
}

Vector SoftmaxRegression::predict_classes(const Matrix& X) const {
    return argmax_rows(predict_proba(X));
}

double SoftmaxRegression::categorical_cross_entropy(
    const Matrix& X,
    const Vector& y
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "SoftmaxRegression::categorical_cross_entropy: model must be fitted before scoring"
        );
    }

    validate_non_empty_vector(y, "SoftmaxRegression::categorical_cross_entropy");
    validate_same_number_of_rows(X, y, "SoftmaxRegression::categorical_cross_entropy");
    validate_class_indices(
        y,
        num_classes_, 
        "SoftmaxRegression::categorical_cross_entropy"
    );

    const Matrix probabilities = predict_proba(X);
    const Matrix Y = one_hot_encode(y, num_classes_);

    return ml::categorical_cross_entropy(probabilities, Y);
}

const Matrix& SoftmaxRegression::weights() const {
    return weights_;
}

const Vector& SoftmaxRegression::bias() const {
    return bias_;
}

Eigen::Index SoftmaxRegression::num_classes() const {
    return num_classes_;
}

bool SoftmaxRegression::is_fitted() const {
    return fitted_;
}

const SoftmaxRegressionTrainingHistory& SoftmaxRegression::training_history() const {
    return history_;
}

const SoftmaxRegressionOptions& SoftmaxRegression::options() const {
    return options_;
}

}  // namespace ml