#include "ml/trees/class_weights.hpp"

#include "ml/common/shape_validation.hpp"

#include <cmath>
#include <map>
#include <stdexcept>

namespace ml {

namespace {

std::map<int, std::size_t> class_counts_for_weights(
    const Vector& y
) {
    validate_non_empty_vector(y, "class_counts_for_weights");

    std::map<int, std::size_t> counts;

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double value = y(i);
        const double rounded = std::round(value);

        if (std::abs(value - rounded) > 1e-12) {
            throw std::invalid_argument(
                "class_counts_for_weights: class labels must be integer-valued"
            );
        }

        if (rounded < 0.0) {
            throw std::invalid_argument(
                "class_counts_for_weights: class labels must be non-negative"
            );
        }

        counts[static_cast<int>(rounded)] += 1;
    }

    return counts;
}

}  // namespace

std::map<int, double> balanced_class_weights(
    const Vector& y
) {
    const std::map<int, std::size_t> counts = class_counts_for_weights(y);

    const double n_samples = static_cast<double>(y.size());

    const double n_classes = static_cast<double>(counts.size());

    std::map<int, double> weights;

    for (const auto& [label, count] : counts) {
        if (count == 0) {
            throw std::invalid_argument(
                "balanced_class_weights: class count must be positive"
            );
        }

        weights[label] = n_samples / (n_classes * static_cast<double>(count));
    }

    return weights;
}

Vector sample_weights_from_class_weights(
    const Vector& y,
    const std::map<int, double>& class_weights
) {
    validate_non_empty_vector(y, "sample_weights_from_class_weights");

    if (class_weights.empty()) {
        throw std::invalid_argument(
            "sample_weights_from_class_weights: class_weights must not be empty"
        );
    }

    Vector sample_weights(y.size());

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double value = y(i);
        const double rounded = std::round(value);

        if (std::abs(value - rounded) > 1e-12) {
            throw std::invalid_argument(
                "sample_weights_from_class_weights: class labels must be integer-valued"
            );
        }

        if (rounded < 0.0) {
            throw std::invalid_argument(
                "sample_weights_from_class_weights: class labels must be non-negative"
            );
        }

        const int label = static_cast<int>(rounded);
        const auto it = class_weights.find(label);

        if (it == class_weights.end()) {
            throw std::invalid_argument(
                "sample_weights_from_class_weights: missing class weight"
            );
        }

        if (!std::isfinite(it->second) || it->second < 0.0) {
            throw std::invalid_argument(
                "sample_weights_from_class_weights: class weights must be finite and non-negative"
            );
        }

        sample_weights(i) = it->second;
    }

    return sample_weights;
}

}  // namespace ml