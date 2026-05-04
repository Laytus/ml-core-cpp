#include "ml/optimization/model_optimization_adapters.hpp"

#include "ml/common/classification_utils.hpp"
#include "ml/common/regression_metrics.hpp"
#include "ml/common/shape_validation.hpp"
#include "ml/common/math_ops.hpp"

#include <stdexcept>
#include <string>

namespace ml {

namespace {

void validate_regularization_for_gradient_training(
    const RegularizationConfig& regularization,
    const std::string& context
) {
    if (regularization.lambda < 0.0) {
        throw std::invalid_argument(
            context + ": regularization lambda must be non-negative"
        );
    }

    if (regularization.is_lasso()) {
        throw std::invalid_argument(
            context + ": Lasso regularization is not implemented for optimizer adapters yet"
        );
    }
}

void validate_target_matrix_single_column(
    const Matrix& y,
    const std::string& context
) {
    validate_non_empty_matrix(y, context);

    if (y.cols() != 1) {
        throw std::invalid_argument(
            context + ": target matrix must have exactly one column"
        );
    }
}

void validate_parameter_shapes(
    const Matrix& weights,
    const Vector& bias,
    Eigen::Index expected_weight_rows,
    Eigen::Index expected_weight_cols,
    Eigen::Index expected_bias_size,
    const std::string& context
) {
    if (weights.rows() != expected_weight_rows ||
        weights.cols() != expected_weight_cols) {
        throw std::invalid_argument(
            context + ": weights have invalid shape"
        );
    }

    if (bias.size() != expected_bias_size) {
        throw std::invalid_argument(
            context + ": bias has invalid shape"
        );
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// LinearRegressionOptimizationProblem
// -----------------------------------------------------------------------------

LinearRegressionOptimizationProblem::LinearRegressionOptimizationProblem(
    Eigen::Index num_features,
    RegularizationConfig regularization
)
    : weights_{Matrix::Zero(num_features, 1)},
      bias_{Vector::Zero(1)},
      regularization_{regularization} {
    if (num_features <= 0) {
        throw std::invalid_argument(
            "LinearRegressionOptimizationProblem: num_features must be positive"
        );
    }

    validate_regularization_for_gradient_training(
        regularization_,
        "LinearRegressionOptimizationProblem"
    );
}

Matrix LinearRegressionOptimizationProblem::weights() const {
    return weights_;
}

Vector LinearRegressionOptimizationProblem::bias() const {
    return bias_;
}

void LinearRegressionOptimizationProblem::set_parameters(
    const Matrix& weights,
    const Vector& bias
) {
    validate_parameter_shapes(
        weights,
        bias,
        weights_.rows(),
        weights_.cols(),
        bias_.size(),
        "LinearRegressionOptimizationProblem::set_parameters"
    );

    weights_ = weights;
    bias_ = bias;
}

double LinearRegressionOptimizationProblem::loss(
    const Matrix& X,
    const Matrix& y
) const {
    validate_non_empty_matrix(X, "LinearRegressionOptimizationProblem::loss");
    validate_target_matrix_single_column(y, "LinearRegressionOptimizationProblem::loss");
    validate_same_number_of_rows(X, y, "LinearRegressionOptimizationProblem::loss");

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument(
            "LinearRegressionOptimizationProblem::loss: X feature count must match weights rows"
        );
    }

    Vector predictions = X * weights_.col(0);
    predictions.array() += bias_(0);

    const Vector targets = y.col(0);

    double result = mean_squared_error(predictions, targets);

    if (regularization_.is_ridge()) {
        result += regularization_.lambda * weights_.squaredNorm();
    }

    return result;
}

ParameterGradient LinearRegressionOptimizationProblem::gradients(
    const Matrix& X,
    const Matrix& y
) const {
    validate_non_empty_matrix(X, "LinearRegressionOptimizationProblem::gradients");
    validate_target_matrix_single_column(y, "LinearRegressionOptimizationProblem::gradients");
    validate_same_number_of_rows(X, y, "LinearRegressionOptimizationProblem::gradients");

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument(
            "LinearRegressionOptimizationProblem::gradients: X feature count must match weights rows"
        );
    }

    const double num_samples = static_cast<double>(X.rows());

    Vector predictions = X * weights_.col(0);
    predictions.array() += bias_(0);

    const Vector errors = predictions - y.col(0);

    Matrix weights_gradient = (2.0 / num_samples) * (X.transpose() * errors);

    Vector bias_gradient = Vector::Zero(1);
    bias_gradient(0) = (2.0 / num_samples) * errors.sum();

    if (regularization_.is_ridge()) {
        weights_gradient += 2.0 * regularization_.lambda * weights_;
    }

    return ParameterGradient{
        weights_gradient,
        bias_gradient
    };
}

// -----------------------------------------------------------------------------
// LogisticRegressionOptimizationProblem
// -----------------------------------------------------------------------------

LogisticRegressionOptimizationProblem::LogisticRegressionOptimizationProblem(
    Eigen::Index num_features,
    RegularizationConfig regularization
)
    : weights_{Matrix::Zero(num_features, 1)},
      bias_{Vector::Zero(1)},
      regularization_{regularization} {
    if (num_features <= 0) {
        throw std::invalid_argument(
            "LogisticRegressionOptimizationProblem: num_features must be positive"
        );
    }

    validate_regularization_for_gradient_training(
        regularization_,
        "LogisticRegressionOptimizationProblem"
    );
}

Matrix LogisticRegressionOptimizationProblem::weights() const {
    return weights_;
}

Vector LogisticRegressionOptimizationProblem::bias() const {
    return bias_;
}

void LogisticRegressionOptimizationProblem::set_parameters(
    const Matrix& weights,
    const Vector& bias
) {
    validate_parameter_shapes(
        weights,
        bias,
        weights_.rows(),
        weights_.cols(),
        bias_.size(),
        "LogisticRegressionOptimizationProblem::set_parameters"
    );

    weights_ = weights;
    bias_ = bias;
}

double LogisticRegressionOptimizationProblem::loss(
    const Matrix& X,
    const Matrix& y
) const {
    validate_non_empty_matrix(X, "LogisticRegressionOptimizationProblem::loss");
    validate_target_matrix_single_column(y, "LogisticRegressionOptimizationProblem::loss");
    validate_same_number_of_rows(X, y, "LogisticRegressionOptimizationProblem::loss");

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument(
            "LogisticRegressionOptimizationProblem::loss: X feature count must match weights rows"
        );
    }

    const Vector targets = y.col(0);
    validate_binary_targets(targets, "LogisticRegressionOptimizationProblem::loss");

    Vector logits = X * weights_.col(0);
    logits.array() += bias_(0);

    const Vector probabilities = sigmoid(logits);

    double result = binary_cross_entropy(probabilities, targets);

    if (regularization_.is_ridge()) {
        result += regularization_.lambda * weights_.squaredNorm();
    }

    return result;
}

ParameterGradient LogisticRegressionOptimizationProblem::gradients(
    const Matrix& X,
    const Matrix& y
) const {
    validate_non_empty_matrix(X, "LogisticRegressionOptimizationProblem::gradients");
    validate_target_matrix_single_column(y, "LogisticRegressionOptimizationProblem::gradients");
    validate_same_number_of_rows(X, y, "LogisticRegressionOptimizationProblem::gradients");

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument(
            "LogisticRegressionOptimizationProblem::gradients: X feature count must match weights rows"
        );
    }

    const Vector targets = y.col(0);
    validate_binary_targets(targets, "LogisticRegressionOptimizationProblem::gradients");

    const double num_samples = static_cast<double>(X.rows());

    Vector logits = X * weights_.col(0);
    logits.array() += bias_(0);

    const Vector probabilities = sigmoid(logits);
    const Vector errors = probabilities - targets;

    Matrix weights_gradient = (1.0 / num_samples) * (X.transpose() * errors);

    Vector bias_gradient = Vector::Zero(1);
    bias_gradient(0) = (1.0 / num_samples) * errors.sum();

    if (regularization_.is_ridge()) {
        weights_gradient += 2.0 * regularization_.lambda * weights_;
    }

    return ParameterGradient{
        weights_gradient,
        bias_gradient
    };
}

// -----------------------------------------------------------------------------
// SoftmaxRegressionOptimizationProblem
// -----------------------------------------------------------------------------

SoftmaxRegressionOptimizationProblem::SoftmaxRegressionOptimizationProblem(
    Eigen::Index num_features,
    Eigen::Index num_classes,
    RegularizationConfig regularization
)
    : weights_{Matrix::Zero(num_features, num_classes)},
      bias_{Vector::Zero(num_classes)},
      num_classes_{num_classes},
      regularization_{regularization} {
    if (num_features <= 0) {
        throw std::invalid_argument(
            "SoftmaxRegressionOptimizationProblem: num_features must be positive"
        );
    }

    if (num_classes <= 1) {
        throw std::invalid_argument(
            "SoftmaxRegressionOptimizationProblem: num_classes must be greater than 1"
        );
    }

    validate_regularization_for_gradient_training(
        regularization_,
        "SoftmaxRegressionOptimizationProblem"
    );
}

Matrix SoftmaxRegressionOptimizationProblem::weights() const {
    return weights_;
}

Vector SoftmaxRegressionOptimizationProblem::bias() const {
    return bias_;
}

void SoftmaxRegressionOptimizationProblem::set_parameters(
    const Matrix& weights,
    const Vector& bias
) {
    validate_parameter_shapes(
        weights,
        bias,
        weights_.rows(),
        weights_.cols(),
        bias_.size(),
        "SoftmaxRegressionOptimizationProblem::set_parameters"
    );

    weights_ = weights;
    bias_ = bias;
}

double SoftmaxRegressionOptimizationProblem::loss(
    const Matrix& X,
    const Matrix& y
) const {
    validate_non_empty_matrix(X, "SoftmaxRegressionOptimizationProblem::loss");
    validate_non_empty_matrix(y, "SoftmaxRegressionOptimizationProblem::loss");
    validate_same_number_of_rows(X, y, "SoftmaxRegressionOptimizationProblem::loss");
    
    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument(
            "SoftmaxRegressionOptimizationProblem::loss: X feature count must match weights rows"
        );
    }
    
    if (y.cols() != num_classes_) {
        throw std::invalid_argument(
            "SoftmaxRegressionOptimizationProblem::loss: target matrix columns must match num_classes"
        );
    }

    Matrix logits = X * weights_;
    logits.rowwise() += bias_.transpose();
    
    const Matrix probabilities = softmax_rows(logits);
    
    double result = categorical_cross_entropy(probabilities, y);
    
    if (regularization_.is_ridge()) {
        result += regularization_.lambda * weights_.squaredNorm();
    }

    return result;
}

ParameterGradient SoftmaxRegressionOptimizationProblem::gradients(
    const Matrix& X,
    const Matrix& y
) const {
    validate_non_empty_matrix(X, "SoftmaxRegressionOptimizationProblem::gradients");
    validate_non_empty_matrix(y, "SoftmaxRegressionOptimizationProblem::gradients");
    validate_same_number_of_rows(X, y, "SoftmaxRegressionOptimizationProblem::gradients");

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument(
            "SoftmaxRegressionOptimizationProblem::gradients: X feature count must match weights rows"
        );
    }

    if (y.cols() != num_classes_) {
        throw std::invalid_argument(
            "SoftmaxRegressionOptimizationProblem::gradients: target matrix columns must match num_classes"
        );
    }

    const double num_samples = static_cast<double>(X.rows());

    Matrix logits = X * weights_;
    logits.rowwise() += bias_.transpose();

    const Matrix probabilities = softmax_rows(logits);
    const Matrix errors = probabilities - y;

    Matrix weights_gradient = (1.0 / num_samples) * (X.transpose() * errors);

    Vector bias_gradient = Vector::Zero(num_classes_);

    for (Eigen::Index class_index = 0; class_index < num_classes_; ++class_index) {
        bias_gradient(class_index) =
            (1.0 / num_samples) * errors.col(class_index).sum();
    }

    if (regularization_.is_ridge()) {
        weights_gradient += 2.0 * regularization_.lambda * weights_;
    }

    return ParameterGradient{
        weights_gradient,
        bias_gradient
    };
}

Eigen::Index SoftmaxRegressionOptimizationProblem::num_classes() const {
    return num_classes_;
}

}  // namespace ml