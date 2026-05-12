#pragma once

#include "ml/workflows/output_writer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ml::workflows {

struct HyperparameterSweepConfig {
    std::filesystem::path output_root{"outputs/practical-exercises"};

    std::filesystem::path regression_dataset_path;
    std::filesystem::path binary_dataset_path;
    std::filesystem::path multiclass_dataset_path;
    std::filesystem::path unsupervised_dataset_path;

    std::string regression_dataset_name{"stock_ohlcv_engineered"};
    std::string binary_dataset_name{"nasa_kc1_software_defects"};
    std::string multiclass_dataset_name{"wine"};
    std::string unsupervised_dataset_name{"stock_ohlcv_engineered"};

    std::vector<std::string> regression_feature_columns;
    std::string regression_target_column;

    std::vector<std::string> binary_feature_columns;
    std::string binary_target_column;

    std::vector<std::string> multiclass_feature_columns;
    std::string multiclass_target_column;
    std::size_t multiclass_num_classes{3};

    std::vector<std::string> unsupervised_feature_columns;

    double test_ratio{0.2};
    bool shuffle{true};
    std::uint32_t seed{42};

    bool standardize_features{true};

    // Keep sanity runs fast. Set to 0 to use the full dataset.
    std::size_t regression_max_rows{500};
    std::size_t binary_max_rows{800};
    std::size_t multiclass_max_rows{0};
    std::size_t unsupervised_max_rows{1000};

    std::vector<std::size_t> gradient_boosting_n_estimators_values{5, 10, 25};
    std::vector<double> gradient_boosting_learning_rate_values{0.03, 0.05, 0.1};

    std::vector<double> logistic_regression_learning_rate_values{0.005, 0.01, 0.05};
    std::vector<double> logistic_regression_ridge_lambda_values{0.0, 0.01, 0.1};

    std::vector<std::size_t> decision_tree_max_depth_values{2, 4, 6};
    std::vector<std::size_t> decision_tree_min_samples_leaf_values{1, 5, 10};

    std::vector<std::size_t> random_forest_n_estimators_values{5, 10, 25};
    std::vector<std::size_t> random_forest_max_features_values{4, 8, 12};

    std::vector<std::size_t> tiny_mlp_hidden_units_values{4, 8, 16};
    std::vector<double> tiny_mlp_learning_rate_values{0.01, 0.05};
    std::vector<std::size_t> tiny_mlp_max_epochs_values{50, 100};

    std::vector<std::size_t> knn_k_values{1, 3, 5, 9};

    std::vector<std::size_t> pca_num_components_values{2, 3, 5};
    std::vector<std::size_t> kmeans_num_clusters_values{2, 3, 4, 6};
};

struct HyperparameterSweepSummary {
    std::size_t regression_rows{0};
    std::size_t binary_rows{0};
    std::size_t multiclass_rows{0};
    std::size_t unsupervised_rows{0};

    std::size_t regression_sweep_rows{0};
    std::size_t binary_sweep_rows{0};
    std::size_t multiclass_sweep_rows{0};
    std::size_t unsupervised_sweep_rows{0};
};

HyperparameterSweepSummary run_hyperparameter_sweeps(
    const HyperparameterSweepConfig& config
);

}  // namespace ml::workflows