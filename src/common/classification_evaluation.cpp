#include "ml/common/classification_evaluation.hpp"

#include "ml/common/classification_metrics.hpp"
#include "ml/common/classification_utils.hpp"
#include "ml/common/shape_validation.hpp"

namespace ml {

ClassificationEvaluation evaluate_binary_classification(
    const Vector& probabilities,
    const Vector& predicted_classes,
    const Vector& targets
) {
    validate_non_empty_vector(probabilities, "evaluate_binary_classification");
    validate_binary_targets(predicted_classes, "evaluate_binary_classification");
    validate_binary_targets(targets, "evaluate_binary_classification");

    validate_same_size(
        probabilities,
        targets,
        "evaluate_binary_classification probabilities/targets"
    );

    validate_same_size(
        predicted_classes,
        targets,
        "evaluate_binary_classification predicted_classes/targets"
    );

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        if (probabilities(i) < 0.0 || probabilities(i) > 1.0) {
            throw std::invalid_argument(
                "evaluate_binary_classification: probabilities must be between 0.0 and 1.0"
            );
        }
    }

    return ClassificationEvaluation{
        accuracy_score(predicted_classes, targets),
        precision_score(predicted_classes, targets),
        recall_score(predicted_classes, targets),
        f1_score(predicted_classes, targets),
        binary_cross_entropy(probabilities, targets),
        confusion_matrix(predicted_classes, targets)
    };
}

}  // namespace ml