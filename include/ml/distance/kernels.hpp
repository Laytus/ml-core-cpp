#pragma once

#include "ml/common/types.hpp"

namespace ml {

double linear_kernel(
    const Vector& a,
    const Vector& b
);

double polynomial_kernel(
    const Vector& a,
    const Vector& b,
    double degree,
    double coef0 = 1.0
);

double rbf_kernel(
    const Vector& a,
    const Vector& b,
    double gamma
);

}  // namespace ml