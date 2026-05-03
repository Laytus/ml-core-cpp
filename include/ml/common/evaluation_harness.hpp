#pragma once

#include "ml/common/evaluation.hpp"
#include "ml/common/types.hpp"
#include "ml/common/classification_evaluation.hpp"

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

struct BinaryClassificationEvaluationInput {
    Vector targets;
    Vector probabilities;
    Vector predicted_classes;
    std::string model_name;
    double threshold{0.5};
};

struct BinaryClassificationEvaluationReport {
    std::string model_name;
    double threshold;
    ClassificationEvaluation evaluation;

    bool has_perfect_accuracy() const;
};

BinaryClassificationEvaluationReport run_binary_classification_evaluation(
    const BinaryClassificationEvaluationInput& input
);

}  // namespace ml