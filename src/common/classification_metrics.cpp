#include "ml/common/classification_metrics.hpp"

#include "ml/common/classification_utils.hpp"
#include "ml/common/shape_validation.hpp"

#include <string>

namespace ml {

namespace {

void validate_binary_prediction_targets(
    const Vector& predicted_classes,
    const Vector& targets,
    const std::string& context
) {
    validate_binary_targets(predicted_classes, context);
    validate_binary_targets(targets, context);
    validate_same_size(predicted_classes, targets, context);
}

double safe_divide(double numerator, double denominator) {
    if (denominator == 0.0) {
        return 0.0;
    }

    return numerator / denominator;
}

}  // namespace

ConfusionMatrix confusion_matrix(
    const Vector& predicted_classes,
    const Vector& targets
) {
    validate_binary_prediction_targets(
        predicted_classes,
        targets,
        "confusion_matrix"
    );

    ConfusionMatrix matrix{};

    for (Eigen::Index i = 0; i < predicted_classes.size(); ++i) {
        const double predicted = predicted_classes(i);
        const double target = targets(i);

        if (predicted == 1.0 && target == 1.0) {
            ++matrix.true_positive;
        } else if (predicted == 0.0 && target == 0.0) {
            ++matrix.true_negative;
        } else if (predicted == 1.0 && target == 0.0) {
            ++matrix.false_positive;
        } else if (predicted == 0.0 && target == 1.0) {
            ++matrix.false_negative;
        }
    }

    return matrix;
}

double accuracy_score(
    const Vector& predicted_classes,
    const Vector& targets
) {
    const ConfusionMatrix matrix = confusion_matrix(predicted_classes, targets);

    const double correct = static_cast<double>(
        matrix.true_positive + matrix.true_negative
    );

    const double total = static_cast<double>(
        matrix.true_positive +
        matrix.true_negative +
        matrix.false_positive +
        matrix.false_negative
    );

    return safe_divide(correct, total);
}

double precision_score(
    const Vector& predicted_classes,
    const Vector& targets
) {
    const ConfusionMatrix matrix = confusion_matrix(predicted_classes, targets);

    const double true_positive = static_cast<double>(matrix.true_positive);
    const double predicted_positive = static_cast<double>(
        matrix.true_positive + matrix.false_positive
    );

    return safe_divide(true_positive, predicted_positive);
}

double recall_score(
    const Vector& predicted_classes,
    const Vector& targets
) {
    const ConfusionMatrix matrix = confusion_matrix(predicted_classes, targets);

    const double true_positive = static_cast<double>(matrix.true_positive);
    const double actual_positive = static_cast<double>(
        matrix.true_positive + matrix.false_negative
    );

    return safe_divide(true_positive, actual_positive);
}

double f1_score(
    const Vector& predicted_classes,
    const Vector& targets
) {
    const double precision = precision_score(predicted_classes, targets);
    const double recall = recall_score(predicted_classes, targets);

    return safe_divide(
        2.0 * precision * recall,
        precision + recall
    );
}

} // namespace ml