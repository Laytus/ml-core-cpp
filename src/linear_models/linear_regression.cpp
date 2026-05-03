#include "ml/linear_models/linear_regression.hpp"

#include "ml/common/math_ops.hpp"
#include "ml/common/shape_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

namespace {

void validate_options(const LinearRegressionOptions& options) {
    if (options.learning_rate <= 0.0) {
        throw std::invalid_argument(
            "LinearRegression::fit: learning_rate must be strictly greater than 0"
        );
    }
    
    if (options.max_iterations == 0) {
        throw std::invalid_argument(
            "LinearRegression::fit: max_iteractions must be strictly greater than 0"
        );
    }
    
    if (options.tolerance < 0.0) {
        throw std::invalid_argument(
            "LinearRegression::fit: tolerance must be non-negative"
        );
    }
    
    if (options.regularization.lambda < 0.0) {
        throw std::invalid_argument(
            "LinearRegression::fit: regularization lambda must be non-negative"
        );
    }

    if (options.regularization.is_lasso()) {
        throw std::invalid_argument(
            "LinearRegression::fit: Lasso regularization is not implemented yet"
        );
    }

    if (
        options.regularization.type != RegularizationType::None &&
        options.regularization.type != RegularizationType::Ridge 
    ) {
        throw std::invalid_argument(
            "LinearRegression::fit: regularization type must be None or Ridge"
        );
    }
}

double mse_with_regularization(
    const Vector& predictions,
    const Vector& targets,
    const Vector& weights,
    const RegularizationConfig& regularization
) {
    double loss = mean_squared_error(predictions, targets);
            
    if (regularization.is_ridge()) {
        loss += regularization.lambda * weights.squaredNorm();
    }

    return loss;
}

}  // namespace

LinearRegression::LinearRegression(LinearRegressionOptions options)
    : options_{options} {}

void LinearRegression::fit (
    const Matrix& X,
    const Vector& y
) {
    validate_options(options_);

    validate_non_empty_matrix(X, "LinearRegression::fit");
    validate_non_empty_vector(y, "LinearRegression::fit");
    validate_same_number_of_rows(X, y, "LinearRegression::fit");

    const auto num_samples = static_cast<double>(X.rows());

    weights_ = Vector::Zero(X.cols());
    bias_ = 0.0;
    fitted_ = false;
    history_ = LinearRegressionTrainingHistory{};

    bool has_previous_loss = false;
    double previous_loss = 0.0;

    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        const Vector predictions = linear_prediction(X, weights_, bias_);
        const Vector residual_vector = residuals(predictions, y);

        double loss = mse_with_regularization(
            predictions,
            y,
            weights_,
            options_.regularization
        );
    
        Vector gradient_w = (2.0 / num_samples) * (X.transpose() * residual_vector);
        const double gradient_b = (2.0 / num_samples) * residual_vector.sum();
    
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

Vector LinearRegression::predict(const Matrix& X) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "LinearRegression::predict: model must be fitted before prediction"
        );
    }

    validate_non_empty_matrix(X, "LinearRegression::predict");
    validate_feature_count(X, weights_, "LinearRegression::predict");

    return linear_prediction(X, weights_, bias_);
}

double LinearRegression::score_mse(
    const Matrix& X,
    const Vector& y
) {
    const Vector predictions = predict(X);

    validate_non_empty_vector(y, "LinearRegression::score_mse");
    validate_same_size(predictions, y, "LinearRegression::score_mse");

    return mean_squared_error(predictions, y);
}

const Vector& LinearRegression::weights() const {
    return weights_;
}

double LinearRegression::bias() const {
    return bias_;
}

bool LinearRegression::is_fitted() const {
    return fitted_;
}

const LinearRegressionTrainingHistory& LinearRegression::training_history() const {
    return history_;
}

const LinearRegressionOptions& LinearRegression::options() const {
    return options_;
}

}  // namespace ml