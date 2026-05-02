#include "ml/common/preprocessing_pipeline.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/common/statistics.hpp"
#include "ml/common/numeric_utils.hpp"

#include <stdexcept>

namespace ml {

StandardScaler StandardScaler::fit(const Matrix& X_train) {
    validate_non_empty_matrix(X_train, "StandardScaler::fit");

    StandardScaler scaler;
    scaler.means = column_means(X_train);
    scaler.standard_deviations = replace_near_zero_with_one(column_standard_deviation_population(X_train));

    return scaler;
}

Matrix StandardScaler::transform(const Matrix& X) const {
    validate_non_empty_vector(means, "StandardScaler::transform");
    validate_non_empty_vector(standard_deviations, "StandardScaler::transform");
    validate_same_size(means, standard_deviations, "StandardScaler::transform");

    validate_feature_count_matches(
        X,
        means.size(),
        "StandardScaler::transform"
    );

    Matrix result(X.rows(), X.cols());

    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        result.col(j) = X.col(j);
        result.col(j).array() -= means(j);
        result.col(j) /= standard_deviations(j);
    }

    return result;
}

Matrix StandardScaler::fit_transform(const Matrix& X_train) {
    *this = StandardScaler::fit(X_train);
    return transform(X_train);
};

MinMaxScaler MinMaxScaler::fit(const Matrix& X_train) {
    validate_non_empty_matrix(X_train, "MinMaxScaler::fit");

    MinMaxScaler scaler;
    scaler.mins = column_mins(X_train);
    
    const Vector maxs = column_maxs(X_train);
    scaler.ranges = replace_near_zero_with_one(maxs - scaler.mins);

    return scaler;
}

Matrix MinMaxScaler::transform(const Matrix& X) const {
    validate_non_empty_vector(mins, "MinMaxScaler::transform");
    validate_non_empty_vector(ranges, "MinMaxScaler::transform");
    validate_same_size(mins, ranges, "MinMaxScaler::transform");

    validate_feature_count_matches(
        X,
        mins.size(),
        "MinMaxScaler::transform"
    );

    Matrix result(X.rows(), X.cols());

    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        result.col(j) = X.col(j);
        result.col(j).array() -= mins(j);
        result.col(j) /= ranges(j);
    }

    return result;
}

Matrix MinMaxScaler::fit_transform(const Matrix& X_train) {
    *this = MinMaxScaler::fit(X_train);
    return transform(X_train);
}

}  // namespace ml