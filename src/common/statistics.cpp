#include "ml/common/statistics.hpp"
#include "ml/common/shape_validation.hpp"

#include <cmath>

namespace ml {

double mean(const Vector& values) {
    validate_non_empty_vector(values, "mean");

    return values.mean();
}

double variance_population(const Vector& values) {
    validate_non_empty_vector(values, "variance_population");

    Vector c = values.array() - mean(values);
    return c.squaredNorm() / static_cast<double>(values.size());
}

double variance_sample(const Vector& values) {
    validate_min_vector_size(values, 2, "variance_sample");
    
    Vector c = values.array() - mean(values);
    return c.squaredNorm() / (static_cast<double>(values.size()) - 1);
}

double standard_deviation_population(const Vector& values) {
    validate_non_empty_vector(values, "standard_deviation_population");

    return std::sqrt(variance_population(values));
}

double standard_deviation_sample(const Vector& values) {
    validate_min_vector_size(values, 2, "standard_deviation_sample");
    
    return std::sqrt(variance_sample(values));
}

Vector column_means(const Matrix& X) {
    validate_non_empty_matrix(X, "column_means");

    return X.colwise().mean();
}

Vector column_variance_population(const Matrix& X) {
    validate_non_empty_matrix(X, "column_variance_population");

    Vector variances(X.cols());

    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        variances(j) = variance_population(X.col(j));
    }

    return variances;
}

Vector column_variance_sample(const Matrix& X) {
    validate_non_empty_matrix(X, "column_variance_sample");
    validate_min_matrix_rows(X, 2 ,"column_variance_sample");
    
    Vector variances(X.cols());
    
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        variances(j) = variance_sample(X.col(j));
    }
    
    return variances;
}

}  // namespace ml