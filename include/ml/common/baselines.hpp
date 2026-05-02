#pragma once

#include "ml/common/types.hpp"

namespace ml {

class MeanRegressor {
public:
    MeanRegressor() = default;

    void fit(const Vector& targets);

    Vector predict(Eigen::Index num_samples) const;

    double value() const;

    bool is_fitted() const;

private:
    double mean_target_{0.0};
    bool fitted_{false};
};

}  // namespace ml