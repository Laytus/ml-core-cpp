#pragma once

#include "ml/common/types.hpp"

namespace ml {
    
double euclidean_distance(
    const Vector& a,
    const Vector& b
);
    
double squared_euclidean_distance(
    const Vector& a,
    const Vector& b
);
    
double manhattan_distance(
    const Vector& a,
    const Vector& b
);

} // namespace ml