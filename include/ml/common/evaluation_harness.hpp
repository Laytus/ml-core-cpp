#pragma once

#include "ml/common/evaluation.hpp"
#include "ml/common/types.hpp"

#include <string>

namespace ml {

struct RegressionEvaluationInput {
    Vector targets;
    Vector baseline_predictions;
    Vector model_predictions;
    std::string model_name;
    std::string baseline_name;
};

struct RegressionEvaluationReport {
    std::string model_name;
    std::string baseline_name;
    BaselineComparison comparison;

    bool model_beats_baseline_mse() const;
    bool model_beats_baseline_rmse() const;
    bool model_beats_baseline_mae() const;
    bool model_beats_baseline_r2() const;
};

RegressionEvaluationReport run_regression_evaluation(
    const RegressionEvaluationInput& input
);

}  // namespace ml