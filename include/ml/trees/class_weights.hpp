#pragma once

#include "ml/common/types.hpp"

#include <map>

namespace ml {

std::map<int, double> balanced_class_weights(
    const Vector& y
);

Vector sample_weights_from_class_weights(
    const Vector& y,
    const std::map<int, double>& class_weights
);

}  // namespace ml