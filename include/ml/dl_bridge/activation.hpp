#pragma once

#include "ml/common/types.hpp"

namespace ml {

Matrix relu(
    const Matrix& Z
);

Matrix relu_derivative(
    const Matrix& Z
);

Matrix sigmoid(
    const Matrix& Z
);

// Vector sigmoid(
//     const Vector& Z
// );

}  // namespace ml