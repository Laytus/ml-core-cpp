#pragma once

#include "ml/common/types.hpp"

namespace ml {

Matrix multiclass_confusion_matrix(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
);

double multiclass_accuracy_score(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
);

double macro_precision(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
);

double macro_recall(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
);

double macro_f1(
    const Vector& predicted_classes,
    const Vector& targets,
    Eigen::Index num_classes
);

}  // namespace ml