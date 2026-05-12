#pragma once

#include "ml/linear_models/linear_regression.hpp"
#include "ml/trees/gradient_boosting.hpp"
#include "ml/trees/regression_tree.hpp"
#include "ml/workflows/output_writer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ml::workflows {

struct RegressionComparisonConfig {
    std::filesystem::path dataset_path;
    std::filesystem::path output_root{"outputs/practical-exercises"};

    std::string workflow_folder{"regression"};
    std::string workflow_name{"regression"};
    std::string dataset_name;
    std::string split_name{"test"};

    std::vector<std::string> feature_columns;
    std::string target_column;

    double test_ratio{0.2};
    bool shuffle{true};
    std::uint32_t seed{42};

    // Keeps practical workflow runs fast and avoids huge prediction files.
    // Set to 0 to use the full dataset.
    std::size_t max_rows{5000};

    bool standardize_features{true};
    bool export_predictions{true};
    bool export_loss_history{true};

    LinearRegressionOptions linear_regression_options{};
    RegressionTreeOptions decision_tree_options{};
    GradientBoostingRegressorOptions gradient_boosting_options{};
};

struct RegressionComparisonSummary {
    std::size_t total_rows{0};
    std::size_t train_rows{0};
    std::size_t test_rows{0};
    std::size_t feature_count{0};

    std::vector<MetricsRow> metrics;
};

RegressionComparisonSummary run_regression_comparison(
    const RegressionComparisonConfig& config
);

}  // namespace ml::workflows