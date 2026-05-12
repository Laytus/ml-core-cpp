#pragma once

#include "ml/distance/knn_classifier.hpp"
#include "ml/linear_models/softmax_regression.hpp"
#include "ml/probabilistic/naive_bayes.hpp"
#include "ml/trees/decision_tree.hpp"
#include "ml/trees/random_forest.hpp"
#include "ml/workflows/output_writer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ml::workflows {

struct MulticlassClassificationComparisonConfig {
    std::filesystem::path dataset_path;
    std::filesystem::path output_root{"outputs/practical-exercises"};

    std::string workflow_folder{"multiclass-classification"};
    std::string workflow_name{"multiclass_classification"};
    std::string dataset_name;
    std::string split_name{"test"};

    std::vector<std::string> feature_columns;
    std::string target_column;

    Eigen::Index num_classes{0};

    double test_ratio{0.2};
    bool shuffle{true};
    std::uint32_t seed{42};

    // Keeps sanity runs fast. Set to 0 to use the full dataset.
    std::size_t max_rows{0};

    bool standardize_features{true};
    bool export_predictions{true};
    bool export_probabilities{true};
    bool export_loss_history{true};

    SoftmaxRegressionOptions softmax_regression_options{};
    KNNClassifierOptions knn_options{};
    GaussianNaiveBayesOptions gaussian_naive_bayes_options{};
    DecisionTreeOptions decision_tree_options{};
    RandomForestOptions random_forest_options{};
};

struct MulticlassClassificationComparisonSummary {
    std::size_t total_rows{0};
    std::size_t train_rows{0};
    std::size_t test_rows{0};
    std::size_t feature_count{0};
    Eigen::Index num_classes{0};

    std::vector<MetricsRow> metrics;
};

MulticlassClassificationComparisonSummary run_multiclass_classification_comparison(
    const MulticlassClassificationComparisonConfig& config
);

}  // namespace ml::workflows