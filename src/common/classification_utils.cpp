#include "ml/common/classification_utils.hpp"

#include "ml/common/shape_validation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ml {

double sigmoid(double z) {
    if (z >= 0.0) {
        const double exp_neg_z = std::exp(-z);
        return 1.0 / (1.0 + exp_neg_z);
    }

    const double exp_z = std::exp(z);
    return exp_z / (1.0 + exp_z);
}

Vector sigmoid(const Vector& logits) {
    validate_non_empty_vector(logits, "sigmoid");

    Vector probabilities(logits.size());

    for (Eigen::Index i = 0; i < logits.size(); ++i) {
        probabilities(i) = sigmoid(logits(i));
    }

    return probabilities;
}

void validate_binary_targets(
    const Vector& targets,
    const std::string& context
) {
    validate_non_empty_vector(targets, context);

    for (Eigen::Index i = 0; i < targets.size(); ++i) {
        const double value = targets(i);

        if (value != 0.0 && value != 1.0) {
            throw std::invalid_argument(
                context + ": targets must contain only 0 or 1"
            );
        }
    }
}

void validate_probability_threshold(
    double threshold,
    const std::string& context
) {
    if (threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument(
            context + ": threshold must be between 0.0 and 1.0"
        );
    }
}

Vector threshold_probabilities(
    const Vector& probabilities,
    double threshold
) {
    validate_non_empty_vector(probabilities, "threshold_probabilities");
    validate_probability_threshold(threshold, "threshold_probabilities");

    Vector classes(probabilities.size());

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        const double probability = probabilities(i);

        if (probability < 0.0 || probability > 1.0) {
            throw std::invalid_argument(
                "threshold_probabilities: probabilities must be between 0.0 and 1.0"
            );
        }

        classes(i) = probability >= threshold ? 1.0 : 0.0;
    }

    return classes;
}

double binary_cross_entropy(
    const Vector& probabilities,
    const Vector& targets,
    double epsilon
) {
    validate_non_empty_vector(probabilities, "binary_cross_entropy");
    validate_binary_targets(targets, "binary_cross_entropy");
    validate_same_size(probabilities, targets, "binary_cross_entropy");

    if (epsilon <= 0.0 || epsilon >= 0.5) {
        throw std::invalid_argument(
            "binary_cross_entropy: epsilon must be in the interval (0.0, 0.5)"
        );
    }

    double total_loss = 0.0;

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        const double clipped_probability = std::clamp(
            probabilities(i),
            epsilon,
            1.0 - epsilon
        );

        total_loss += -(
            targets(i) * std::log(clipped_probability) +
            (1.0 - targets(i)) * std::log(1.0 - clipped_probability)
        );
    }

    return total_loss / static_cast<double>(probabilities.size());
}

Matrix softmax_rows(const Matrix& logits) {
    validate_non_empty_matrix(logits, "softmax_rows");

    const Vector row_max_values = logits.rowwise().maxCoeff();

    Matrix shifted_logits = logits;
    shifted_logits.colwise() -= row_max_values;

    const Matrix exponentials = shifted_logits.array().exp().matrix();
    const Vector denominators = exponentials.rowwise().sum();

    return (exponentials.array().colwise() / denominators.array()).matrix();
}

void validate_class_indices(
    const Vector& targets,
    Eigen::Index num_classes,
    const std::string& context
) {
    validate_non_empty_vector(targets, context);

    if (num_classes <= 1) {
        throw std::invalid_argument(
            context + ": num_classes must be greater than 1"
        );
    }

    for (Eigen::Index i = 0; i < targets.size(); ++i) {
        const double value = targets(i);
        const double rounded_value = std::round(value);

        if (std::abs(value - rounded_value) > 1e-12) {
            throw std::invalid_argument(
                context + ": class indices must be integer-valued"
            );
        }

        const auto class_index = static_cast<Eigen::Index>(rounded_value);

        if (class_index < 0 || class_index >= num_classes) {
            throw std::invalid_argument(
                context + ": class indices must be in [0, num_classes]"
            );
        }
    }
}

Matrix one_hot_encode(
    const Vector& targets,
    Eigen::Index num_classes
) {
    validate_class_indices(targets, num_classes, "one_hot_encode");

    Matrix encoded = Matrix::Zero(targets.size(), num_classes);

    for (Eigen::Index i = 0; i < targets.size(); ++i) {
        const auto class_index = static_cast<Eigen::Index>(std::round(targets(i)));
        encoded(i, class_index) = 1.0;
    }

    return encoded;
}

Vector argmax_rows(const Matrix& probabilities) {
    validate_non_empty_matrix(probabilities, "argmax_rows");

    Vector classes(probabilities.rows());

    for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
        Eigen::Index best_class = 0;
        double best_probability = probabilities(i, 0);

        if (best_probability < 0.0 || best_probability > 1.0) {
            throw std::invalid_argument(
                "argmax_rows: probabilities must be between 0.0 and 1.0"
            );
        }

        double row_sum = best_probability;

        for (Eigen::Index j = 1; j < probabilities.cols(); ++j) {
            const double probability = probabilities(i, j);

            if (probability < 0.0 || probability > 1.0) {
                throw std::invalid_argument(
                    "argmax_rows: probabilities must be between 0.0 and 1.0"
                );
            }

            row_sum += probability;

            if (probability > best_probability) {
                best_probability = probability;
                best_class = j;
            }
        }

        if (std::abs(row_sum - 1.0) > 1e-8) {
            throw std::invalid_argument(
                "argmax_rows: each probability row must sum to 1.0"
            );
        }

        classes(i) = static_cast<double>(best_class);
    }

    return classes;
}

double categorical_cross_entropy(
    const Matrix& probabilities,
    const Matrix& one_hot_targets,
    double epsilon
) {
    validate_non_empty_matrix(probabilities, "categorical_cross_entropy");
    validate_non_empty_matrix(one_hot_targets, "categorical_cross_entropy");

    if (probabilities.rows() != one_hot_targets.rows()) {
        throw std::invalid_argument(
            "categorical_cross_entropy: probabilities and targets must have the same number of rows"
        );
    }

    if (probabilities.cols() != one_hot_targets.cols()) {
        throw std::invalid_argument(
            "categorical_cross_entropy: probabilities and targets must have the same number of columns"
        );
    }

    if (epsilon <= 0.0 || epsilon >= 0.5) {
        throw std::invalid_argument(
            "categorical_cross_entropy: epsilon must be in the interval (0.0, 0.5)"
        );
    }

    double total_loss = 0.0;

    for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
        double target_row_sum = 0.0;

        for (Eigen::Index j = 0; j < probabilities.cols(); ++j) {
            const double probability = probabilities(i, j);
            const double target = one_hot_targets(i, j);

            if (probability < 0.0 || probability > 1.0) {
                throw std::invalid_argument(
                    "categorical_cross_entropy: probabilities must be between 0.0 and 1.0"
                );
            }

            if (target != 0.0 && target != 1.0) {
                throw std::invalid_argument(
                    "categorical_cross_entropy: one-hot targets must contain only 0 or 1"
                );
            }

            target_row_sum += target;

            const double clipped_probability = std::clamp(
                probability,
                epsilon,
                1.0 - epsilon
            );

            total_loss += -(target * std::log(clipped_probability));
        }

        if (target_row_sum != 1.0) {
            throw std::invalid_argument(
                "categorical_cross_entropy: each one-hot target row must sum to 1"
            );
        }
    }

    return total_loss / static_cast<double>(probabilities.rows());
}

}  // namespace ml