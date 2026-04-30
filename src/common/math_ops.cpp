#include "ml/common/math_ops.hpp"
#include "ml/common/shape_validation.hpp"

namespace ml {

double dot_product(const Vector& a, const Vector& b) {
    validate_non_empty_vector(a, "dot_product");
    validate_non_empty_vector(b, "dot_product");
    validate_same_size(a, b, "dot_product");

    return a.dot(b);
}

Vector matvec(const Matrix& X, const Vector& weights) {
    validate_non_empty_matrix(X, "matvec");
    validate_non_empty_vector(weights, "matvec");
    validate_feature_count(X, weights, "matvec");

    return X * weights;
}

Vector linear_prediction(const Matrix& X, const Vector& weights, double bias) {
    validate_non_empty_matrix(X, "linear_prediction");
    validate_non_empty_vector(weights, "linear_prediction");
    validate_feature_count(X, weights, "linear_prediction");

    Vector predictions = matvec(X, weights);
    predictions.array() += bias;

    return predictions;
}

Vector residuals(const Vector& predictions, const Vector& targets) {
    validate_non_empty_vector(predictions, "residuals");
    validate_non_empty_vector(targets, "residuals");
    validate_same_size(predictions, targets, "residuals");

    return predictions - targets;
}

double mean_squared_error(const Vector& predictions, const Vector& targets) {
    validate_non_empty_vector(predictions, "mean_squared_error");
    validate_non_empty_vector(targets, "mean_squared_error");
    validate_same_size(predictions, targets, "mean_squared_error");

    Vector r = residuals(predictions, targets);
    return r.squaredNorm() / static_cast<double>(r.size());
}

}  // namespace ml