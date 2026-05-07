#pragma once

#include "ml/common/types.hpp"

#include <cstddef>
#include <string>

namespace ml {

enum class DistanceMetric {
    Euclidean,
    SquaredEuclidean,
    Manhattan
};

std::string distance_metric_name (
    DistanceMetric metric
);

struct KNNClassifierOptions {
    std::size_t k{3};
    DistanceMetric distance_metric{DistanceMetric::Euclidean};
};

void validate_knn_classifier_options(
    const KNNClassifierOptions& options,
    const std::string& context
);

class KNNClassifier {
public:
    KNNClassifier() = default;

    explicit KNNClassifier(
        KNNClassifierOptions options
    );

    void fit(
        const Matrix& X,
        const Vector& y
    );

    Vector predict(
        const Matrix& X
    ) const;

    bool is_fitted() const;

    const KNNClassifierOptions& options() const;

    std::size_t num_train_samples() const;

    Eigen::Index num_features() const;

private:
    double distance_between(
        const Vector& a,
        const Vector& b
    ) const;
    
    double predict_one(
        const Vector& sample        
    ) const;

    KNNClassifierOptions options_{};
    Matrix X_train_{};
    Vector y_train_{};

    bool fitted_{false};
};

}  // namespace ml