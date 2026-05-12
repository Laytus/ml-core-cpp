#include "ml/workflows/unsupervised_comparison.hpp"

#include "ml/common/csv_dataset_loader.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml::workflows {
namespace {

struct UnsupervisedData {
    Matrix X;
    Vector label_reference;
    bool has_label_reference{false};
};

struct StandardizationStats {
    Vector means;
    Vector stddevs;
};

void validate_config(const UnsupervisedComparisonConfig& config) {
    if (config.dataset_path.empty()) {
        throw std::invalid_argument("UnsupervisedComparisonConfig.dataset_path must not be empty");
    }

    if (config.output_root.empty()) {
        throw std::invalid_argument("UnsupervisedComparisonConfig.output_root must not be empty");
    }

    if (config.workflow_folder.empty()) {
        throw std::invalid_argument("UnsupervisedComparisonConfig.workflow_folder must not be empty");
    }

    if (config.workflow_name.empty()) {
        throw std::invalid_argument("UnsupervisedComparisonConfig.workflow_name must not be empty");
    }

    if (config.dataset_name.empty()) {
        throw std::invalid_argument("UnsupervisedComparisonConfig.dataset_name must not be empty");
    }

    if (config.feature_columns.empty()) {
        throw std::invalid_argument("UnsupervisedComparisonConfig.feature_columns must not be empty");
    }

    if (config.pca_options.num_components < 2) {
        throw std::invalid_argument("UnsupervisedComparisonConfig requires at least 2 PCA components");
    }
}

Matrix take_rows(
    const Matrix& X,
    const std::vector<Eigen::Index>& indices
) {
    Matrix result;
    result.resize(static_cast<Eigen::Index>(indices.size()), X.cols());

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(indices.size()); ++i) {
        result.row(i) = X.row(indices[static_cast<std::size_t>(i)]);
    }

    return result;
}

Vector take_values(
    const Vector& y,
    const std::vector<Eigen::Index>& indices
) {
    Vector result;
    result.resize(static_cast<Eigen::Index>(indices.size()));

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(indices.size()); ++i) {
        result(i) = y(indices[static_cast<std::size_t>(i)]);
    }

    return result;
}

UnsupervisedData maybe_limit_rows(
    const UnsupervisedData& data,
    std::size_t max_rows,
    std::uint32_t seed
) {
    const std::size_t n_rows = static_cast<std::size_t>(data.X.rows());

    if (max_rows == 0 || n_rows <= max_rows) {
        return data;
    }

    std::vector<Eigen::Index> indices(n_rows);
    std::iota(indices.begin(), indices.end(), static_cast<Eigen::Index>(0));

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    indices.resize(max_rows);
    std::sort(indices.begin(), indices.end());

    UnsupervisedData limited;
    limited.X = take_rows(data.X, indices);
    limited.has_label_reference = data.has_label_reference;

    if (data.has_label_reference) {
        limited.label_reference = take_values(data.label_reference, indices);
    }

    return limited;
}

StandardizationStats fit_standardizer(const Matrix& X) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("fit_standardizer: X must be non-empty");
    }

    StandardizationStats stats;
    stats.means = X.colwise().mean();
    stats.stddevs.resize(X.cols());

    for (Eigen::Index col = 0; col < X.cols(); ++col) {
        const Vector centered = X.col(col).array() - stats.means(col);
        const double variance = centered.array().square().mean();
        const double stddev = std::sqrt(variance);

        stats.stddevs(col) = stddev < 1e-12 ? 1.0 : stddev;
    }

    return stats;
}

Matrix apply_standardizer(
    const Matrix& X,
    const StandardizationStats& stats
) {
    if (X.cols() != stats.means.size() || X.cols() != stats.stddevs.size()) {
        throw std::invalid_argument("apply_standardizer: incompatible feature count");
    }

    Matrix result = X;

    for (Eigen::Index col = 0; col < X.cols(); ++col) {
        result.col(col).array() -= stats.means(col);
        result.col(col).array() /= stats.stddevs(col);
    }

    return result;
}

MetricsRow make_metric_row(
    const UnsupervisedComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const std::string& metric,
    double value
) {
    return {
        run_id,
        config.workflow_name,
        config.dataset_name,
        model,
        config.split_name,
        metric,
        value
    };
}

std::string label_reference_at(
    const UnsupervisedData& data,
    Eigen::Index row
) {
    if (!data.has_label_reference) {
        return "";
    }

    return std::to_string(static_cast<int>(data.label_reference(row)));
}

std::vector<ProjectionRow> build_projection_rows(
    const UnsupervisedComparisonConfig& config,
    const std::string& run_id,
    const Matrix& Z,
    const UnsupervisedData& data
) {
    if (Z.cols() < 2) {
        throw std::invalid_argument("build_projection_rows: expected at least 2 projected components");
    }

    if (Z.rows() != data.X.rows()) {
        throw std::invalid_argument("build_projection_rows: projection row count mismatch");
    }

    std::vector<ProjectionRow> rows;
    rows.reserve(static_cast<std::size_t>(Z.rows()));

    for (Eigen::Index i = 0; i < Z.rows(); ++i) {
        rows.push_back({
            run_id,
            static_cast<std::size_t>(i),
            config.workflow_name,
            config.dataset_name,
            "PCA",
            config.split_name,
            Z(i, 0),
            Z(i, 1),
            label_reference_at(data, i)
        });
    }

    return rows;
}

std::vector<ClusteringAssignmentRow> build_clustering_rows(
    const UnsupervisedComparisonConfig& config,
    const std::string& run_id,
    const std::string& method,
    const Vector& clusters,
    const UnsupervisedData& data
) {
    if (clusters.size() != data.X.rows()) {
        throw std::invalid_argument("build_clustering_rows: cluster row count mismatch");
    }

    std::vector<ClusteringAssignmentRow> rows;
    rows.reserve(static_cast<std::size_t>(clusters.size()));

    for (Eigen::Index i = 0; i < clusters.size(); ++i) {
        rows.push_back({
            run_id,
            static_cast<std::size_t>(i),
            config.workflow_name,
            config.dataset_name,
            method,
            config.split_name,
            static_cast<int>(clusters(i)),
            label_reference_at(data, i)
        });
    }

    return rows;
}

}  // namespace

UnsupervisedComparisonSummary run_unsupervised_comparison(
    const UnsupervisedComparisonConfig& config
) {
    validate_config(config);

    common::CsvDatasetLoader loader;

    UnsupervisedData data;

    if (config.label_reference_column.empty()) {
        const auto loaded = loader.load_unsupervised(
            config.dataset_path.string(),
            config.feature_columns
        );

        data.X = loaded.X;
        data.has_label_reference = false;
    } else {
        const auto loaded = loader.load_supervised(
            config.dataset_path.string(),
            config.feature_columns,
            config.label_reference_column
        );

        data.X = loaded.X;
        data.label_reference = loaded.y;
        data.has_label_reference = true;
    }

    data = maybe_limit_rows(data, config.max_rows, config.seed);

    if (config.standardize_features) {
        const StandardizationStats stats = fit_standardizer(data.X);
        data.X = apply_standardizer(data.X, stats);
    }

    OutputWriter writer(config.output_root);

    std::vector<MetricsRow> all_metrics;

    PCA pca(config.pca_options);
    const Matrix Z = pca.fit_transform(data.X);

    for (Eigen::Index i = 0; i < pca.explained_variance_ratio().size(); ++i) {
        all_metrics.push_back(
            make_metric_row(
                config,
                "pca_2d_baseline",
                "PCA",
                "explained_variance_ratio_" + std::to_string(i + 1),
                pca.explained_variance_ratio()(i)
            )
        );
    }

    KMeans kmeans(config.kmeans_options);
    const Vector clusters = kmeans.fit_predict(data.X);

    all_metrics.push_back(
        make_metric_row(
            config,
            "kmeans_baseline",
            "KMeans",
            "inertia",
            kmeans.inertia()
        )
    );

    all_metrics.push_back(
        make_metric_row(
            config,
            "kmeans_baseline",
            "KMeans",
            "iterations",
            static_cast<double>(kmeans.num_iterations())
        )
    );

    KMeans pca_kmeans(config.pca_kmeans_options);
    const Vector pca_clusters = pca_kmeans.fit_predict(Z.leftCols(2));

    all_metrics.push_back(
        make_metric_row(
            config,
            "pca_kmeans_baseline",
            "PCA+KMeans",
            "inertia",
            pca_kmeans.inertia()
        )
    );

    all_metrics.push_back(
        make_metric_row(
            config,
            "pca_kmeans_baseline",
            "PCA+KMeans",
            "iterations",
            static_cast<double>(pca_kmeans.num_iterations())
        )
    );

    writer.write_metrics(config.workflow_folder, all_metrics, false);

    if (config.export_projections) {
        writer.write_projections_2d(
            config.workflow_folder,
            build_projection_rows(config, "pca_2d_baseline", Z, data),
            false
        );
    }

    if (config.export_clustering_assignments) {
        writer.write_clustering_assignments(
            config.workflow_folder,
            build_clustering_rows(config, "kmeans_baseline", "KMeans", clusters, data),
            false
        );

        writer.write_clustering_assignments(
            config.workflow_folder,
            build_clustering_rows(config, "pca_kmeans_baseline", "PCA+KMeans", pca_clusters, data),
            true
        );
    }

    std::cout << "\n[Unsupervised comparison workflow]\n";
    std::cout << "Dataset: " << config.dataset_name << "\n";
    std::cout << "Rows used: " << static_cast<std::size_t>(data.X.rows()) << "\n";
    std::cout << "Features: " << static_cast<std::size_t>(data.X.cols()) << "\n";
    std::cout << "PCA components: " << config.pca_options.num_components << "\n";
    std::cout << "KMeans clusters: " << config.kmeans_options.num_clusters << "\n";
    std::cout << "PCA+KMeans clusters: " << config.pca_kmeans_options.num_clusters << "\n";
    std::cout << "Output folder: " << (config.output_root / config.workflow_folder) << "\n";

    return {
        static_cast<std::size_t>(data.X.rows()),
        static_cast<std::size_t>(data.X.cols()),
        config.pca_options.num_components,
        config.kmeans_options.num_clusters,
        config.pca_kmeans_options.num_clusters,
        all_metrics
    };
}

}  // namespace ml::workflows