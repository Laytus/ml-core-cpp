#pragma once

#include "ml/unsupervised/kmeans.hpp"
#include "ml/unsupervised/pca.hpp"
#include "ml/workflows/output_writer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ml::workflows {

struct UnsupervisedComparisonConfig {
    std::filesystem::path dataset_path;
    std::filesystem::path output_root{"outputs/practical-exercises"};

    std::string workflow_folder{"unsupervised"};
    std::string workflow_name{"unsupervised"};
    std::string dataset_name;
    std::string split_name{"full"};

    std::vector<std::string> feature_columns;

    // Optional column kept only for qualitative visualization labels.
    // Leave empty for datasets without labels.
    std::string label_reference_column;

    std::size_t max_rows{1000};
    std::uint32_t seed{42};

    bool standardize_features{true};
    bool export_projections{true};
    bool export_clustering_assignments{true};

    PCAOptions pca_options{};
    KMeansOptions kmeans_options{};
    KMeansOptions pca_kmeans_options{};
};

struct UnsupervisedComparisonSummary {
    std::size_t total_rows{0};
    std::size_t feature_count{0};
    std::size_t projection_components{0};
    std::size_t kmeans_clusters{0};
    std::size_t pca_kmeans_clusters{0};

    std::vector<MetricsRow> metrics;
};

UnsupervisedComparisonSummary run_unsupervised_comparison(
    const UnsupervisedComparisonConfig& config
);

}  // namespace ml::workflows