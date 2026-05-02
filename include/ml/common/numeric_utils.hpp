#pragma once

#include "ml/common/types.hpp"

namespace ml {

Vector replace_near_zero_with_one(
    const Vector& values,
    double tolerance = 1e-10
);

void validate_feature_count_matches(
    const Matrix& X,
    Eigen::Index expected_num_features,
    const std::string& context
);

}  // namespace ml