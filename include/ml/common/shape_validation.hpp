#pragma once

#include "ml/common/types.hpp"

#include <string>

namespace ml {

void validate_same_number_of_rows(
    const Matrix& X,
    const Vector& y,
    const std::string& context
);

void validate_same_size(
    const Vector& a,
    const Vector& b,
    const std::string& context
);

void validate_feature_count(
    const Matrix& X,
    const Vector& weights,
    const std::string& context
);

void validate_non_empty_matrix(
    const Matrix& X,
    const std::string& context
);

void validate_non_empty_vector(
    const Vector& v,
    const std::string& context
);

}  // namespace ml