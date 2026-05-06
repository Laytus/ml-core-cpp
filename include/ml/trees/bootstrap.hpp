#pragma once

#include "ml/common/types.hpp"

#include <vector>

namespace ml {

struct BootstrapSample {
    Matrix X;
    Vector y;

    std::vector<Eigen::Index> sampled_indices;
    std::vector<Eigen::Index> out_of_bag_indices;
};

BootstrapSample make_bootstrap_sample(
    const Matrix& X,
    const Vector& y,
    unsigned int random_seed
);

BootstrapSample make_full_sample(
    const Matrix& X,
    const Vector& y
);

}  // namespace ml