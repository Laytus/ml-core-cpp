#include "ml/common/evaluation_harness.hpp"

#include "ml/common/evaluation.hpp"
#include "ml/common/shape_validation.hpp"

#include <stdexcept>

namespace ml {

RegressionEvaluationReport run_regression_evaluation(
    const RegressionEvaluationInput& input
) {

    validate_non_empty_vector(input.targets, "run_regression_evaluation");
    validate_non_empty_vector(input.baseline_predictions, "run_regression_evaluation");
    validate_non_empty_vector(input.model_predictions, "run_regression_evaluation");

    validate_same_size(
        input.baseline_predictions,
        input.targets,
        "run_regression_evaluation baseline_predictions/targets"
    );

    validate_same_size(
        input.model_predictions,
        input.targets,
        "run_regression_evaluation model_predictions/targets"
    );

    if (input.model_name.empty()) {
        throw std::invalid_argument(
            "run_regression_evaluation: model_name must not be empty"
        );
    }

    if (input.baseline_name.empty()) {
        throw std::invalid_argument(
            "run_regression_evaluation: baseline_name must not be empty"
        );
    }

    BaselineComparison comparison = compare_regression_to_baseline(
        input.baseline_predictions,
        input.model_predictions,
        input.targets
    );

    return RegressionEvaluationReport {
        input.model_name,
        input.baseline_name,
        comparison
    };
}

bool RegressionEvaluationReport::model_beats_baseline_mse() const {
    return comparison.model.mse < comparison.baseline.mse;
}

bool RegressionEvaluationReport::model_beats_baseline_rmse() const {
    return comparison.model.rmse < comparison.baseline.rmse;
}

bool RegressionEvaluationReport::model_beats_baseline_mae() const {
    return comparison.model.mae < comparison.baseline.mae;
}

bool RegressionEvaluationReport::model_beats_baseline_r2() const {
    return comparison.model.r2 > comparison.baseline.r2;
}

}  // namespace ml