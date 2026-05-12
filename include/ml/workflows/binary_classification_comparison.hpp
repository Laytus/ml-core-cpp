#pragma once

#include "ml/linear_models/linear_svm.hpp"
#include "ml/linear_models/logistic_regression.hpp"
#include "ml/probabilistic/naive_bayes.hpp"
#include "ml/trees/decision_tree.hpp"
#include "ml/trees/random_forest.hpp"
#include "ml/workflows/output_writer.hpp"
#include "ml/dl_bridge/mlp.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ml::workflows {

struct BinaryClassificationComparisonConfig {
    std::filesystem::path dataset_path;
    std::filesystem::path output_root{"outputs/practical-exercises"};

    std::string workflow_folder{"binary-classification"};
    std::string workflow_name{"binary_classification"};
    std::string dataset_name;
    std::string split_name{"test"};

    std::vector<std::string> feature_columns;
    std::string target_column;

    double test_ratio{0.2};
    bool shuffle{true};
    std::uint32_t seed{42};

    // Keeps practical workflow sanity runs fast.
    // Set to 0 to use the full dataset.
    std::size_t max_rows{0};

    bool standardize_features{true};
    bool export_predictions{true};
    bool export_probabilities{true};
    bool export_decision_scores{true};
    bool export_loss_history{true};

    LogisticRegressionOptions logistic_regression_options{};
    LinearSVMOptions linear_svm_options{};
    GaussianNaiveBayesOptions gaussian_naive_bayes_options{};
    DecisionTreeOptions decision_tree_options{};
    RandomForestOptions random_forest_options{};
    TinyMLPBinaryClassifierOptions tiny_mlp_options{};
};

struct BinaryClassificationComparisonSummary {
    std::size_t total_rows{0};
    std::size_t train_rows{0};
    std::size_t test_rows{0};
    std::size_t feature_count{0};

    std::vector<MetricsRow> metrics;
};

BinaryClassificationComparisonSummary run_binary_classification_comparison(
    const BinaryClassificationComparisonConfig& config
);

}  // namespace ml::workflows