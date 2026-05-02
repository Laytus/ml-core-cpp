#include "ml/common/evaluation.hpp"

#include "ml/common/math_ops.hpp"
#include "ml/common/regression_metrics.hpp"
#include "ml/common/shape_validation.hpp"

namespace ml {

RegressionEvaluation evaluate_regression(
    const Vector& predictions,
    const Vector& targets
) {
    validate_non_empty_vector(predictions, "evaluate_regression");
    validate_non_empty_vector(targets, "evaluate_regression");
    validate_same_size(predictions, targets, "evaluate_regression");

    return RegressionEvaluation{
        mean_squared_error(predictions, targets),
        root_mean_squared_error(predictions, targets),
        mean_absolute_error(predictions, targets),
        r2_score(predictions, targets),
    };
}

BaselineComparison compare_regression_to_baseline(
    const Vector& baseline_predictions,
    const Vector& model_predictions,
    const Vector& targets
) {
    validate_non_empty_vector(baseline_predictions, "compare_regression_to_baseline");
    validate_non_empty_vector(model_predictions, "compare_regression_to_baseline");
    validate_non_empty_vector(targets, "compare_regression_to_baseline");
    
    validate_same_size(
        baseline_predictions,
        targets,
        "compare_regression_to_baseline baseline_predictions/targets"
    );

    validate_same_size(
        model_predictions,
        targets,
        "compare_regression_to_baseline model_predictions/targets"
    );

    return BaselineComparison{
        evaluate_regression(baseline_predictions, targets),
        evaluate_regression(model_predictions, targets)
    };
}

}  // namespace ml