#pragma once

#include "ml/common/classification_metrics.hpp"

namespace ml {

struct ClassificationEvaluation {
    double accuracy;
    double precision;
    double recall;
    double f1;
    double bce;
    ConfusionMatrix confusion;
};

ClassificationEvaluation evaluate_binary_classification(
    const Vector& probabilities,
    const Vector& predicted_classes,
    const Vector& targets
);

}  // namespace ml