#include "ml/distance/distance_metrics.hpp"

#include "ml/common/shape_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

namespace {

void validate_distance_vectors(
    const Vector& a,
    const Vector& b,
    const std::string& context
) {
    validate_non_empty_vector(a, context + "a");
    validate_non_empty_vector(b, context + "b");
    validate_same_size(a, b, context);

    for (Eigen::Index i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a(i)) || !std::isfinite(b(i))) {
            throw std::invalid_argument(
                context + ": vector values must be finite"
            );
        }
    }
}

}  // namespace
    
double squared_euclidean_distance(
    const Vector& a,
    const Vector& b
) {
    validate_distance_vectors(a, b, "squared_euclidean_distance");
    
    double result = 0.0;

    for (Eigen::Index i = 0; i < a.size(); ++i) {
        const double difference = a(i) - b(i);
        result += difference * difference;
    }

    return result;
}

double euclidean_distance(
    const Vector& a,
    const Vector& b
) {
    validate_distance_vectors(a, b, "euclidean_distance");
    
    return std::sqrt(
        squared_euclidean_distance(a, b)
    );
}    

double manhattan_distance(
    const Vector& a,
    const Vector& b
) {
    validate_distance_vectors(a, b, "manhattan_distance");

    double result = 0.0;

    for (Eigen::Index i = 0; i < a.size(); ++i) {
        result += std::abs(a(i) - b(i));
    }

    return result;
}

} // namespace ml