#include "ml/common/numeric_utils.hpp"
#include "ml/common/shape_validation.hpp"

#include <stdexcept>
#include <cmath>

namespace ml {

void validate_feature_count_matches(
    const Matrix& X,
    Eigen::Index expected_num_features,
    const std::string& context
) {
    validate_non_empty_matrix(X, context);

    if (X.cols() != expected_num_features) {
        throw std::invalid_argument(
            context + ": feature count mismatch. Expected " +
            std::to_string(expected_num_features) +
            " features, got : " +
            std::to_string(X.cols()) +
            "."
        );
    }
}

Vector replace_near_zero_with_one(const Vector& values, double tolerance) {
    Vector result = values;

    for (Eigen::Index i = 0; i < result.size(); ++i) {
        if (std::abs(result(i)) < 1e-10) {
            result(i) = 1.0;
        }
    }

    return result;
}

}  // namespace ml