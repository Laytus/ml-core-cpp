#include "ml/common/multiclass_metrics.hpp"

#include "ml/common/classification_utils.hpp"
#include "ml/common/shape_validation.hpp"

#include <stdexcept>

namespace ml {

namespace {

double safe_divide(double numerator, double denominator) {
    if (denominator == 0.0) {
        return 0.0;
    }

    return numerator / denominator;
}

void validate_multiclass_metric_inputs(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes,
    const std::string& context
) {
    validate_class_indices(predicted_classes, num_classes, context);
    validate_class_indices(targets, num_classes, context);
    validate_same_size(predicted_classes, targets, context);
}

}  // namespace

Matrix multiclass_confusion_matrix(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
) {
    validate_multiclass_metric_inputs(
        predicted_classes,
        targets,
        num_classes,
        "multiclass_confusion_matrix"
    );

    Matrix matrix = Matrix::Zero(num_classes, num_classes);

    for (Eigen::Index i = 0; i < targets.size(); ++i) {
        const auto actual_class = static_cast<Eigen::Index>(targets(i));
        const auto predicted_class = static_cast<Eigen::Index>(predicted_classes(i));

        matrix(actual_class, predicted_class) += 1.0;
    }

    return matrix;
}

double multiclass_accuracy_score(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
) {
    const Matrix matrix = multiclass_confusion_matrix(
        predicted_classes,
        targets,
        num_classes
    );

    const double correct = matrix.diagonal().sum();
    const double total = matrix.sum();

    return safe_divide(correct, total);
}

double macro_precision(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
) {
    const Matrix matrix = multiclass_confusion_matrix(
        predicted_classes,
        targets,
        num_classes
    );

    double precision_sum = 0.0;

    for (Eigen::Index class_index = 0; class_index < num_classes; ++class_index) {
        const double true_positive = matrix(class_index, class_index);
        const double predicted_positive = matrix.col(class_index).sum();

        precision_sum += safe_divide(true_positive, predicted_positive);
    }

    return precision_sum / static_cast<double>(num_classes);
}

double macro_recall(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
) {
    const Matrix matrix = multiclass_confusion_matrix(
        predicted_classes,
        targets,
        num_classes
    );

    double recall_sum = 0.0;

    for (Eigen::Index class_index = 0; class_index < num_classes; ++class_index) {
        const double true_positive = matrix(class_index, class_index);
        const double actual_positive = matrix.row(class_index).sum();

        recall_sum += safe_divide(true_positive, actual_positive);
    }

    return recall_sum / static_cast<double>(num_classes);
}

double macro_f1(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
) {
    const Matrix matrix = multiclass_confusion_matrix(
        predicted_classes,
        targets,
        num_classes
    );

    double f1_sum = 0.0;

    for (Eigen::Index class_index = 0; class_index < num_classes; ++class_index) {
        const double true_positive = matrix(class_index, class_index);
        const double predicted_positive = matrix.col(class_index).sum();
        const double actual_positive = matrix.row(class_index).sum();

        const double precision = safe_divide(true_positive, predicted_positive);
        const double recall = safe_divide(true_positive, actual_positive);

        f1_sum += safe_divide(
            2.0 * precision * recall,
            precision + recall
        );
    }

    return f1_sum / static_cast<double>(num_classes);
}

}  // namespace ml