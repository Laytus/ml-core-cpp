#include "ml/common/preprocessing.hpp"
#include "ml/common/shape_validation.hpp"
#include "ml/common/statistics.hpp"

#include <cmath>

namespace ml {

Matrix standardize_columns(const Matrix& X) {
    validate_non_empty_matrix(X, "standardize_columns");

    const Vector means = column_means(X);

    Vector stds = column_standard_deviation_population(X);
    stds = (stds.array() < 1e-10).select(1.0, stds.array()).matrix();

    const Matrix centered = X.rowwise() - means.transpose();

    return (centered.array().rowwise() / stds.transpose().array()).matrix();
}

Matrix normalize_min_max_columns(const Matrix& X) {
    validate_non_empty_matrix(X, "normalize_min_max_columns");

    Vector mins = X.colwise().minCoeff().transpose();
    Vector maxs = X.colwise().maxCoeff().transpose();
    Vector ranges = maxs - mins;

    ranges = (ranges.array() < 1e-10).select(1.0, ranges.array());

    const Matrix centered = X.rowwise() - mins.transpose();

    return (centered.array().rowwise() / ranges.transpose().array()).matrix();
}

}  // namespace ml