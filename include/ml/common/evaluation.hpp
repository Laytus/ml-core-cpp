#pragma once

#include "ml/common/types.hpp"

namespace ml {

struct RegressionEvaluation {
    double mse;
    double rmse;
    double mae;
    double r2;
};

RegressionEvaluation evaluate_regression(
    const Vector& predictions,
    const Vector& targets
);

struct BaselineComparison {
    RegressionEvaluation baseline;
    RegressionEvaluation model;
};

BaselineComparison compare_regression_to_baseline(
    const Vector& baseline_predictions,
    const Vector& model_predictions,
    const Vector& targets
);

}  // namespace ml