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

void validate_min_vector_size(
    const Vector& v,
    Eigen::Index min_size,
    const std::string& context
);

void validate_min_matrix_rows(
    const Matrix& X,
    Eigen::Index min_rows,
    const std::string& context
);

}  // namespace ml