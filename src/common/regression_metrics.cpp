#include "ml/common/regression_metrics.hpp"

#include "ml/common/math_ops.hpp"
#include "ml/common/shape_validation.hpp"
#include "ml/common/statistics.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

double mean_absolute_error(
    const Vector& predictions,
    const Vector& targets
) {
    validate_non_empty_vector(predictions, "mean_absolute_error");
    validate_non_empty_vector(targets, "mean_absolute_error");
    validate_same_size(predictions, targets, "mean_absolute_error");

    const Vector errors = predictions - targets;

    return errors.array().abs().sum() / static_cast<double>(errors.size());
}

double root_mean_squared_error(
    const Vector& predictions,
    const Vector& targets
) {
    validate_non_empty_vector(predictions, "root_mean_squared_error");
    validate_non_empty_vector(targets, "root_mean_squared_error");
    validate_same_size(predictions, targets, "root_mean_squared_error");

    return std::sqrt(mean_squared_error(predictions, targets));
}

double r2_score(
    const Vector& predictions,
    const Vector& targets
) {
    validate_non_empty_vector(predictions, "r2_score");
    validate_non_empty_vector(targets, "r2_score");
    validate_same_size(predictions, targets, "r2_score");

    const double target_mean = mean(targets);

    const Vector residual_errors = predictions - targets;
    const double ss_res = residual_errors.squaredNorm();

    const Vector centered_targets = targets.array() - target_mean;
    const double ss_tot = centered_targets.squaredNorm();

    if (ss_tot == 0.0) {
        throw std::invalid_argument(
            "r2_score: targets have zero variance, so R2 is undefined"
        );
    }

    return 1.0 - (ss_res / ss_tot);
}

}  // namespace ml