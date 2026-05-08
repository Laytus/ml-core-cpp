#include "ml/unsupervised/kmeans.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/distance/distance_metrics.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

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

Vector row_as_vector(
    const Matrix& X,
    Eigen::Index row_index
) {
    return X.row(row_index).transpose();
}

}  // namespace

void validate_kmeans_options(
    const KMeansOptions& options,
    const std::string& context
) {
    if (options.num_clusters == 0) {
        throw std::invalid_argument(
            context + ": num_clusters must be at least 1"
        );
    }
    
    if (options.max_iterations == 0) {
        throw std::invalid_argument(
            context + ": max_iterations must be at least 1"
        );
    }
    
    if (!std::isfinite(options.tolerance) || options.tolerance < 0.0) {
        throw std::invalid_argument(
            context + ": tolerance must be finite and non-negative"
        );
    }
}

KMeans::KMeans(KMeansOptions options)
    : options_{options} {
    validate_kmeans_options(options, "KMeans");
}

void KMeans::fit(
    const Matrix& X
) {
    validate_kmeans_options(options_, "KMeans::fit");
    
    validate_non_empty_matrix(X, "KMeans::fit");
    validate_finite_matrix_values(X, "KMeans::fit");

    if (options_.num_clusters > static_cast<std::size_t>(X.rows())) {
        throw std::invalid_argument(
            "KMeans::fit: num_clusters must be less than or equal to the number of samples"
        );
    }

    num_features_ = X.cols();

    Matrix current_centroids = initialize_centroids(X);

    inertia_history_.clear();
    labels_ = Vector::Zero(X.rows());

    num_iterations_ = 0;

    for (
        std::size_t iteration = 0;
        iteration < options_.max_iterations;
        ++iteration
    ) {
        const Vector current_labels = assign_clusters(X, current_centroids);
        
        const Matrix updated_centroids =
            update_centroids(X, current_labels, current_centroids);
        
        const double current_inertia =
            compute_inertia(X, current_labels, updated_centroids);

        inertia_history_.push_back(current_inertia);

        const double centroid_shift = (updated_centroids - current_centroids).norm();

        labels_ = current_labels;
        current_centroids = updated_centroids;
        num_iterations_ = iteration + 1;

        if (centroid_shift <= options_.tolerance) {
            break;
        }
    }

    centroids_ = current_centroids;
    fitted_ = true;
}

Vector KMeans::predict(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "KMeans::predict: model must be fitted before prediction"
        );
    }
    
    validate_non_empty_matrix(X, "KMeans::predict");
    validate_finite_matrix_values(X, "KMeans::predict");
    
    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "KMeans::predict: X feature count must match training feature count"
        );
    }
    
    return assign_clusters(X, centroids_);
}

Vector KMeans::fit_predict(
    const Matrix& X
) {
    fit(X);
    return labels_;
}

bool KMeans::is_fitted() const {
    return fitted_;
}

const KMeansOptions& KMeans::options() const {
    return options_;
}

const Matrix& KMeans::centroids() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "KMeans::centroids: model must be fitted"
        );
    }

    return centroids_;
}

const Vector& KMeans::labels() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "KMeans::labels: model must be fitted"
        );
    }

    return labels_;
}

const std::vector<double>& KMeans::inertia_history() const {
    return inertia_history_;
}

double KMeans::inertia() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "KMeans::inertia: model must be fitted"
        );
    }
    
    if (inertia_history_.empty()) {
        throw std::invalid_argument(
            "KMeans::inertia: inertia history is empty"
        );
    }

    return inertia_history_.back();
}

std::size_t KMeans::num_iterations() const {
    return num_iterations_;
}

Matrix KMeans::initialize_centroids(
    const Matrix& X
) const {
    Matrix centroids(
        static_cast<Eigen::Index>(options_.num_clusters),
        X.cols()
    );

    for (
        std::size_t cluster_index = 0;
        cluster_index < options_.num_clusters;
        ++cluster_index 
    ) {
        centroids.row(static_cast<Eigen::Index>(cluster_index)) =
            X.row(static_cast<Eigen::Index>(cluster_index));
    }

    return centroids;
}

Vector KMeans::assign_clusters(
    const Matrix& X,
    const Matrix& centroids
) const {
    Vector labels(X.rows());

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const Vector sample = row_as_vector(X, i);

        double best_distance = std::numeric_limits<double>::infinity();

        Eigen::Index best_cluster = 0;

        for (
            Eigen::Index cluster_index = 0;
            cluster_index < centroids.rows();
            ++cluster_index
        ) {
            const Vector centroid = 
                row_as_vector(centroids, cluster_index);

            const double distance =
                squared_euclidean_distance(sample, centroid);
            
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = cluster_index;
            }
        }

        labels(i) = static_cast<double>(best_cluster);
    }

    return labels;
}

Matrix KMeans::update_centroids(
    const Matrix& X,
    const Vector& labels,
    const Matrix& previous_centroids
) const {
    Matrix updated_centroids =
        Matrix::Zero(
            previous_centroids.rows(),
            previous_centroids.cols()
        );

    Vector counts =
        Vector::Zero(previous_centroids.rows());

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const Eigen::Index cluster_index = static_cast<Eigen::Index>(labels(i));

        updated_centroids.row(cluster_index) += X.row(i);
        counts(cluster_index) += 1.0;
    }

    for (
        Eigen::Index cluster_index = 0;
        cluster_index < updated_centroids.rows();
        ++cluster_index
    ) {
        if (counts(cluster_index) > 0.0) {
            updated_centroids.row(cluster_index) /= counts(cluster_index);
        } else {
            updated_centroids.row(cluster_index) =
                previous_centroids.row(cluster_index);
        }
    }

    return updated_centroids;
}

double KMeans::compute_inertia(
    const Matrix& X,
    const Vector& labels,
    const Matrix& centroids
) const {
    double total = 0.0;

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const Eigen::Index cluster_index = static_cast<Eigen::Index>(labels(i));

        const Vector sample = row_as_vector(X, i);

        const Vector centroid = row_as_vector(centroids, cluster_index);

        total += squared_euclidean_distance(sample, centroid);
    }

    return total;
}


}  // namespace ml