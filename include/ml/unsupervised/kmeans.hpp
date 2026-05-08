#pragma once

#include "ml/common/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ml {

struct KMeansOptions {
    std::size_t num_clusters{2};
    std::size_t max_iterations{100};
    double tolerance{1e-6};
};

void validate_kmeans_options(
    const KMeansOptions& options,
    const std::string& context
);

class KMeans {
public:
    KMeans() = default;

    explicit KMeans(
        KMeansOptions options
    );

    void fit(
        const Matrix& X
    );

    Vector predict(
        const Matrix& X
    ) const;

    Vector fit_predict(
        const Matrix& X
    );

    bool is_fitted() const;

    const KMeansOptions& options() const;

    const Matrix& centroids() const;

    const Vector& labels() const;

    const std::vector<double>& inertia_history() const;

    double inertia() const;

    std::size_t num_iterations() const;

private:
    Matrix initialize_centroids(
        const Matrix& X
    ) const;

    Vector assign_clusters(
        const Matrix& X,
        const Matrix& centroids
    ) const;

    Matrix update_centroids(
        const Matrix& X,
        const Vector& labels,
        const Matrix& previous_centroids
    ) const;

    double compute_inertia(
        const Matrix& X,
        const Vector& labels,
        const Matrix& centroids
    ) const;

    KMeansOptions options_{};

    Matrix centroids_{};
    Vector labels_{};
    std::vector<double> inertia_history_{};

    Eigen::Index num_features_{0};
    std::size_t num_iterations_{0};
    bool fitted_{false};
};

}  // namespace ml