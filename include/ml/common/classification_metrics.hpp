#pragma once

#include "ml/common/types.hpp"

namespace ml {

struct ConfusionMatrix {
    Eigen::Index true_positive{0};
    Eigen::Index true_negative{0};
    Eigen::Index false_positive{0};
    Eigen::Index false_negative{0};
};

ConfusionMatrix confusion_matrix(
    const Vector& predicted_classes,
    const Vector& targets
);

double accuracy_score(
    const Vector& predicted_classes,
    const Vector& targets
);

double precision_score(
    const Vector& predicted_classes,
    const Vector& targets
);

double recall_score(
    const Vector& predicted_classes,
    const Vector& targets
);

double f1_score(
    const Vector& predicted_classes,
    const Vector& targets
);

}  // namespace ml