#pragma once

#include "ml/common/types.hpp"

#include <string>

namespace ml {

double sigmoid(double z);

Vector sigmoid(const Vector& logits);

void validate_binary_targets(
    const Vector& targets,
    const std::string& context
);

void validate_probability_threshold(
    double threshold,
    const std::string& context
);

Vector threshold_probabilities(
    const Vector& probabilities,
    double threshold = 0.5
);

double binary_cross_entropy(
    const Vector& probabilities,
    const Vector& targets,
    double epsilon = 1e-15
);

Matrix softmax_rows(const Matrix& logits);

void validate_class_indices(
    const Vector& targets,
    Eigen::Index num_classes,
    const std::string& context
);

Matrix one_hot_encode(
    const Vector& targets,
    Eigen::Index num_classes
);

Vector argmax_rows(const Matrix& probabilities);

double categorical_cross_entropy(
    const Matrix& probabilities,
    const Matrix& one_hot_targets,
    double epsilon = 1e-15
);

}  // namespace ml