#include "ml/distance/kernels.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/distance/distance_metrics.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

namespace {

void validate_kernel_vectors(
    const Vector& a,
    const Vector& b,
    const std::string& context
) {
    validate_non_empty_vector(a, context + "a");
    validate_non_empty_vector(b, context + "b");
    validate_same_size(a, b, context);

    for (Eigen::Index i = 0; i < a.size(); ++i) {
        if(!std::isfinite(a(i)) || !std::isfinite(b(i))) {
            throw std::invalid_argument(
                context + ": vector values must be finite"
            );
        }
    }
}

}  // namespace

double linear_kernel(
    const Vector& a,
    const Vector& b
) {
    validate_kernel_vectors(a, b, "linear_kernel");
    
    return a.dot(b);
}

double polynomial_kernel(
    const Vector& a,
    const Vector& b,
    double degree,
    double coef0
) {
    validate_kernel_vectors(a, b, "polynomial_kernel");

    if (!std::isfinite(degree) || degree <= 0.0) {
        throw std::invalid_argument(
            "polynomial_kernel: degree must be finite and positive"
        );
    }

    if (!std::isfinite(coef0)) {
        throw std::invalid_argument(
            "polynomial_kernel: coef0 must be finite"
        );
    }

    return std::pow(
        a.dot(b) + coef0,
        degree
    );
}

double rbf_kernel(
    const Vector& a,
    const Vector& b,
    double gamma
) {
    validate_kernel_vectors(a, b, "rbf_kernel");

    if (!std::isfinite(gamma) || gamma <= 0.0) {
        throw std::invalid_argument(
            "rbf_kernel: gamma must be finite and positive"
        );
    }

    return std::exp(
        -gamma * squared_euclidean_distance(a, b)
    );
}

}  // namespace ml