#pragma once

#include "ml/common/types.hpp"

namespace ml {

double mean_absolute_error(
    const Vector& predictions,
    const Vector& targets
);

double root_mean_squared_error(
    const Vector& predictions,
    const Vector& targets
);

double r2_score(
    const Vector& predictions,
    const Vector& targets
);

}  // namespace ml