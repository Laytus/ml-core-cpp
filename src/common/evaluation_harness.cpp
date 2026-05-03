#include "ml/common/evaluation_harness.hpp"

#include "ml/common/classification_evaluation.hpp"
#include "ml/common/classification_utils.hpp"
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

    const BaselineComparison comparison = compare_regression_to_baseline(
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

BinaryClassificationEvaluationReport run_binary_classification_evaluation(
    const BinaryClassificationEvaluationInput& input
) {
    validate_non_empty_vector(input.targets, "run_binary_classification_evaluation");
    validate_non_empty_vector(input.probabilities, "run_binary_classification_evaluation");
    validate_binary_targets(input.predicted_classes, "run_binary_classification_evaluation");
    validate_binary_targets(input.targets, "run_binary_classification_evaluation");

    validate_same_size(
        input.probabilities,
        input.targets,
        "run_binary_classification_evaluation probabilities/targets"
    );

    validate_same_size(
        input.predicted_classes,
        input.targets,
        "run_binary_classification_evaluation predicted_classes/targets"
    );

    validate_probability_threshold(
        input.threshold,
        "run_binary_classification_evaluation"
    );

    if (input.model_name.empty()) {
        throw std::invalid_argument(
            "run_binary_classification_evaluation: model_name must not be empty"
        );
    }

    const ClassificationEvaluation evaluation = evaluate_binary_classification(
        input.probabilities,
        input.predicted_classes,
        input.targets
    );

    return BinaryClassificationEvaluationReport{
        input.model_name,
        input.threshold,
        evaluation
    };
}

bool BinaryClassificationEvaluationReport::has_perfect_accuracy() const {
    return evaluation.accuracy == 1.0;
}

}  // namespace ml