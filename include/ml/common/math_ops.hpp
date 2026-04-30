#pragma once

#include "ml/common/types.hpp"

namespace ml {

double dot_product(
    const Vector& a,
    const Vector& b
);

Vector matvec(
    const Matrix& X,
    const Vector& weights
);

Vector linear_prediction(
    const Matrix& X,
    const Vector& weights,
    double bias
);

Vector residuals(
    const Vector& predictions,
    const Vector& targets
);

double mean_squared_error(
    const Vector& predictions,
    const Vector& targets
);

}  // namespace ml