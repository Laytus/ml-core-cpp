#include "ml/common/baselines.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/common/statistics.hpp"

#include <stdexcept>

namespace ml {

void MeanRegressor::fit(const Vector& targets) {
    validate_non_empty_vector(targets, "MeanRegressor::fit");

    mean_target_ = mean(targets);
    fitted_ = true;
}

Vector MeanRegressor::predict(Eigen::Index num_samples) const {
    if (!fitted_) {
        throw std::invalid_argument(
            "MeanRegressor::predict: model must be fitted before prediction"
        );
    }
    
    if (num_samples <= 0) {
        throw std::invalid_argument(
            "MeanRegressor::predict: num_samples must be positive"
        );
    }

    return Vector::Constant(num_samples, mean_target_);
}

double MeanRegressor::value() const {
    if (!fitted_) {
        throw std::invalid_argument(
            "MeanRegressor::predict: model must be fitted before reading the baseline value"
        );
    }

    return mean_target_;
}

bool MeanRegressor::is_fitted() const {
    return fitted_;
}

}  // namespace ml