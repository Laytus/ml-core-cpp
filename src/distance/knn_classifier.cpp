#include "ml/distance/knn_classifier.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/distance/distance_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {

namespace {

void validate_finite_matrix_values(
    const Matrix& X,
    const std::string& context
) {
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            if (!std::isfinite(X(i, j))) {
                throw std::invalid_argument(
                    context + ": X values must be finite"
                );
            }
        }
    }
}

void validate_class_targets(
    const Vector& y,
    const std::string& context
) {
    validate_non_empty_vector(y, context);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double value = y(i);
        const double rounded = std::round(value);

        if (std::abs(value - rounded) > 1e-12) {
            throw std::invalid_argument(
                context + ": class labels must be integer-valued"
            );
        }
        
        if (rounded < 0.0) {
            throw std::invalid_argument(
                context + ": class labels must be non-negative"
            );
        }
    }
}

Vector row_as_vector(
    const Matrix& X,
    Eigen::Index row
) {
    return X.row(row).transpose();
}

double majority_vote_smallest_label_tie_break(
    const std::vector<double>& labels
) {
    if (labels.empty()) {
        throw std::invalid_argument(
            "majority_vote_smallest_label_tie_break: labels must not be empty"
        );
    }

    std::map<int, std::size_t> counts;

    for (double label_value : labels) {
        const double rounded = std::round(label_value);

        if (std::abs(label_value - rounded) > 1e-12 || rounded < 0.0) {
            throw std::invalid_argument(
                "majority_vote_smallest_label_tie_break: labels must be non-negative integer values"
            );
        }

        counts[static_cast<int>(rounded)] += 1;
    }

    int best_label = -1;
    std::size_t best_count = 0;

    for (const auto& [label, count] : counts) {
        if (count > best_count) {
            best_label = label;
            best_count = count;
        }
    }

    return static_cast<double>(best_label);
}

} // namespace

std::string distance_metric_name (
    DistanceMetric metric
 ) {
    switch (metric) {
        case DistanceMetric::Euclidean:
            return "euclidean";
        
        case DistanceMetric::SquaredEuclidean:
            return "squared_euclidean";
        
        case DistanceMetric::Manhattan:
            return "manhattan";
    }

    throw std::invalid_argument(
        "distance_metric_name: unknown distance metric"
    );
}

void validate_knn_classifier_options(
    const KNNClassifierOptions& options,
    const std::string& context
) {
    if (options.k == 0) {
        throw std::invalid_argument(
            context + ": k must be at least 1"
        );
    }

    static_cast<void>(
        distance_metric_name(options.distance_metric)
    );
}

KNNClassifier::KNNClassifier(KNNClassifierOptions options)
    : options_{options} {
    validate_knn_classifier_options(
        options_,
        "KNNClassifier"
    );
}

void KNNClassifier::fit(
    const Matrix& X,
    const Vector& y
) {
    validate_knn_classifier_options(options_, "KNNClassifier::fit");
    
    validate_non_empty_matrix(X, "KNNClassifier::fit");
    validate_class_targets(y, "KNNClassifier::fit");
    validate_same_number_of_rows(X, y, "KNNClassifier::fit");
    validate_finite_matrix_values(X, "KNNClassifier::fit");
    
    if (options_.k > static_cast<std::size_t>(X.rows())) {
        throw std::invalid_argument(
            "KNNClassifier::fit: k must be less than or equal to the number of training samples"
        );
    }

    X_train_ = X;
    y_train_ = y;
    fitted_ = true;
}

Vector KNNClassifier::predict(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "KNNClassifier::predict: model must be fitted before prediction"
        );
    }
    
    validate_non_empty_matrix(X, "KNNClassifier::predict");
    validate_finite_matrix_values(X, "KNNClassifier::predict");
    
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument(
            "KNNClassifier::predict: X feature count must match training feature count"
        );
    }

    Vector predictions(X.rows());

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        predictions(i) = predict_one(
            row_as_vector(X, i)
        );
    }

    return predictions;
}

bool KNNClassifier::is_fitted() const {
    return fitted_;
}

const KNNClassifierOptions& KNNClassifier::options() const {
    return options_;
}

std::size_t KNNClassifier::num_train_samples() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "KNNClassifier::num_train_samples: model must be fitted"
        );
    }
    
    return static_cast<std::size_t>(X_train_.rows());
}

Eigen::Index KNNClassifier::num_features() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "KNNClassifier::num_features: model must be fitted"
        );
    }
    
    return X_train_.cols();
}

double KNNClassifier::distance_between(
    const Vector& a,
    const Vector& b
) const {
    switch (options_.distance_metric) {
        case DistanceMetric::Euclidean:
            return euclidean_distance(a, b);
        
        case DistanceMetric::SquaredEuclidean:
            return squared_euclidean_distance(a, b);
        
        case DistanceMetric::Manhattan:
            return manhattan_distance(a, b);
    }

    throw std::invalid_argument(
        "KNNClassifier::distance_between: unknown distance metric"
    );
}

double KNNClassifier::predict_one(
    const Vector& sample        
) const {
    std::vector<std::pair<double, double>> distance_label_pairs;
    distance_label_pairs.reserve(static_cast<std::size_t>(X_train_.rows()));

    for (Eigen::Index i = 0; i < X_train_.rows(); ++i) {
        const Vector train_sample = row_as_vector(X_train_, i);

        distance_label_pairs.push_back({
            distance_between(sample, train_sample),
            y_train_(i)
        });
    }

    std::sort(
        distance_label_pairs.begin(),
        distance_label_pairs.end(),
        [](const auto& left, const auto& right) {
            constexpr double epsilon = 1e-12;

            if (std::abs(left.first - right.first) > epsilon) {
                return left.first < right.first;
            }

            return left.second < right.second;
        }
    );

    std::vector<double> nearest_labels;
    nearest_labels.reserve(options_.k);

    for (std::size_t i = 0; i < options_.k; ++i) {
        nearest_labels.push_back(
            distance_label_pairs[i].second
        );
    }

    return majority_vote_smallest_label_tie_break(nearest_labels);
}


}  // namespace ml