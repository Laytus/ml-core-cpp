#pragma once

#include "ml/common/types.hpp"

namespace ml {

Matrix standardize_columns(const Matrix& X);

Matrix normalize_min_max_columns(const Matrix& X);

}  // namespace ml