#include "phase11_practical_workflows_sanity.hpp"

#include "ml/common/csv_dataset_loader.hpp"
#include "ml/workflows/output_writer.hpp"
#include "ml/workflows/regression_comparison.hpp"
#include "ml/workflows/binary_classification_comparison.hpp"
#include "ml/workflows/multiclass_classification_comparison.hpp"
#include "ml/workflows/unsupervised_comparison.hpp"
#include "ml/workflows/hyperparameter_sweeps.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_true(const std::string& label, bool condition) {
    if (!condition) {
        throw std::runtime_error("FAILED: " + label);
    }

    std::cout << "[OK] " << label << "\n";
}

void expect_size_equal(
    const std::string& label,
    std::size_t actual,
    std::size_t expected
) {
    if (actual != expected) {
        throw std::runtime_error(
            "FAILED: " + label
            + " | expected " + std::to_string(expected)
            + ", got " + std::to_string(actual)
        );
    }

    std::cout << "[OK] " << label << "\n";
}

void expect_file_exists_and_non_empty(
    const std::string& label,
    const std::filesystem::path& path
) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("FAILED: " + label + " | file does not exist: " + path.string());
    }

    if (std::filesystem::file_size(path) == 0) {
        throw std::runtime_error("FAILED: " + label + " | file is empty: " + path.string());
    }

    std::cout << "[OK] " << label << "\n";
}

void run_test(const std::string& label, void (*test_fn)()) {
    try {
        test_fn();
        std::cout << "[PASS] " << label << "\n\n";
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << label << "\n";
        std::cerr << "Reason: " << error.what() << "\n";
        throw;
    }
}

std::vector<std::string> stock_feature_columns() {
    return {
        "return_1d",
        "return_5d",
        "volatility_5d",
        "range_pct",
        "volume_change_1d"
    };
}

std::vector<std::string> kc1_feature_columns() {
    return {
        "loc",
        "v_g",
        "ev_g",
        "iv_g",
        "n",
        "v",
        "l",
        "d",
        "i",
        "e",
        "b",
        "t",
        "lOCode",
        "lOComment",
        "lOBlank",
        "locCodeAndComment",
        "uniq_Op",
        "uniq_Opnd",
        "total_Op",
        "total_Opnd",
        "branchCount"
    };
}

std::vector<std::string> wine_feature_columns() {
    return {
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280_od315_of_diluted_wines",
        "proline"
    };
}

bool vector_contains_only_binary_labels(const Eigen::VectorXd& y) {
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double label = y(i);

        if (label != 0.0 && label != 1.0) {
            return false;
        }
    }

    return true;
}

bool vector_contains_only_wine_multiclass_labels(const Eigen::VectorXd& y) {
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double label = y(i);

        if (label != 0.0 && label != 1.0 && label != 2.0) {
            return false;
        }
    }

    return true;
}

void test_regression_workflow_runner() {
    ml::common::CsvDatasetLoader loader;

    const auto dataset = loader.load_supervised(
        "data/processed/stock_ohlcv_engineered.csv",
        stock_feature_columns(),
        "target_next_return"
    );

    expect_true("Regression workflow row count is positive", dataset.rows() > 0);
    expect_size_equal("Regression workflow feature count", dataset.features(), static_cast<std::size_t>(5));
    expect_size_equal("Regression workflow target size", static_cast<std::size_t>(dataset.y.size()), dataset.rows());
    expect_true("Regression workflow has non-empty feature names", !dataset.feature_names.empty());
    expect_true("Regression workflow target name is set", dataset.target_name == "target_next_return");

    std::cout << "Regression dataset rows: " << dataset.rows() << "\n";
    std::cout << "Regression dataset features: " << dataset.features() << "\n";
}

void test_binary_classification_workflow_runner() {
    ml::common::CsvDatasetLoader loader;

    const auto dataset = loader.load_supervised(
        "data/processed/nasa_kc1_software_defects.csv",
        kc1_feature_columns(),
        "defects"
    );

    expect_size_equal("Binary classification row count", dataset.rows(), static_cast<std::size_t>(2109));
    expect_size_equal("Binary classification feature count", dataset.features(), static_cast<std::size_t>(21));
    expect_size_equal("Binary classification target size", static_cast<std::size_t>(dataset.y.size()), dataset.rows());
    expect_true("Binary classification labels are encoded as 0/1", vector_contains_only_binary_labels(dataset.y));
    expect_true("Binary classification target name is set", dataset.target_name == "defects");

    std::cout << "Binary classification dataset rows: " << dataset.rows() << "\n";
    std::cout << "Binary classification dataset features: " << dataset.features() << "\n";
}

void test_multiclass_classification_workflow_runner() {
    ml::common::CsvDatasetLoader loader;

    const auto dataset = loader.load_supervised(
        "data/processed/wine.csv",
        wine_feature_columns(),
        "class"
    );

    expect_size_equal("Multiclass classification row count", dataset.rows(), static_cast<std::size_t>(178));
    expect_size_equal("Multiclass classification feature count", dataset.features(), static_cast<std::size_t>(13));
    expect_size_equal("Multiclass classification target size", static_cast<std::size_t>(dataset.y.size()), dataset.rows());
    expect_true("Multiclass classification labels are encoded as 0/1/2", vector_contains_only_wine_multiclass_labels(dataset.y));
    expect_true("Multiclass classification target name is set", dataset.target_name == "class");

    std::cout << "Multiclass classification dataset rows: " << dataset.rows() << "\n";
    std::cout << "Multiclass classification dataset features: " << dataset.features() << "\n";
}

void test_unsupervised_workflow_runner() {
    ml::common::CsvDatasetLoader loader;

    const auto dataset = loader.load_unsupervised(
        "data/processed/stock_ohlcv_engineered.csv",
        stock_feature_columns()
    );

    expect_true("Unsupervised workflow row count is positive", dataset.rows() > 0);
    expect_size_equal("Unsupervised workflow feature count", dataset.features(), static_cast<std::size_t>(5));
    expect_true("Unsupervised workflow has non-empty feature names", !dataset.feature_names.empty());

    std::cout << "Unsupervised dataset rows: " << dataset.rows() << "\n";
    std::cout << "Unsupervised dataset features: " << dataset.features() << "\n";
}

void test_output_writer_regression_outputs() {
    const ml::workflows::OutputWriter writer("outputs/practical-exercises");

    writer.write_metrics(
        "regression",
        {
            {
                "sanity_regression",
                "regression",
                "stock_ohlcv_engineered",
                "DummyRegressionModel",
                "test",
                "mse",
                0.125
            }
        },
        false
    );

    writer.write_regression_predictions(
        "regression",
        {
            {
                "sanity_regression",
                0,
                "regression",
                "stock_ohlcv_engineered",
                "DummyRegressionModel",
                "test",
                1.0,
                0.8,
                -0.2
            }
        },
        false
    );

    expect_file_exists_and_non_empty(
        "Regression metrics output exists",
        "outputs/practical-exercises/regression/metrics.csv"
    );
    expect_file_exists_and_non_empty(
        "Regression predictions output exists",
        "outputs/practical-exercises/regression/predictions.csv"
    );
}

void test_output_writer_binary_classification_outputs() {
    const ml::workflows::OutputWriter writer("outputs/practical-exercises");

    writer.write_metrics(
        "binary-classification",
        {
            {
                "sanity_binary_classification",
                "binary_classification",
                "nasa_kc1_software_defects",
                "DummyBinaryClassifier",
                "test",
                "accuracy",
                0.75
            }
        },
        false
    );

    writer.write_classification_predictions(
        "binary-classification",
        {
            {
                "sanity_binary_classification",
                0,
                "binary_classification",
                "nasa_kc1_software_defects",
                "DummyBinaryClassifier",
                "test",
                1.0,
                1.0,
                1
            }
        },
        false
    );

    writer.write_binary_probabilities(
        "binary-classification",
        {
            {
                "sanity_binary_classification",
                0,
                "binary_classification",
                "nasa_kc1_software_defects",
                "DummyBinaryClassifier",
                "test",
                1.0,
                0.2,
                0.8
            }
        },
        false
    );

    writer.write_decision_scores(
        "binary-classification",
        {
            {
                "sanity_binary_classification",
                0,
                "binary_classification",
                "nasa_kc1_software_defects",
                "DummyBinaryClassifier",
                "test",
                1.0,
                1.5
            }
        },
        false
    );

    expect_file_exists_and_non_empty(
        "Binary metrics output exists",
        "outputs/practical-exercises/binary-classification/metrics.csv"
    );
    expect_file_exists_and_non_empty(
        "Binary predictions output exists",
        "outputs/practical-exercises/binary-classification/predictions.csv"
    );
    expect_file_exists_and_non_empty(
        "Binary probabilities output exists",
        "outputs/practical-exercises/binary-classification/probabilities.csv"
    );
    expect_file_exists_and_non_empty(
        "Binary decision scores output exists",
        "outputs/practical-exercises/binary-classification/decision_scores.csv"
    );
}

void test_output_writer_multiclass_classification_outputs() {
    const ml::workflows::OutputWriter writer("outputs/practical-exercises");

    writer.write_metrics(
        "multiclass-classification",
        {
            {
                "sanity_multiclass_classification",
                "multiclass_classification",
                "wine",
                "DummyMulticlassClassifier",
                "test",
                "accuracy",
                0.80
            }
        },
        false
    );

    writer.write_classification_predictions(
        "multiclass-classification",
        {
            {
                "sanity_multiclass_classification",
                0,
                "multiclass_classification",
                "wine",
                "DummyMulticlassClassifier",
                "test",
                2.0,
                2.0,
                1
            }
        },
        false
    );

    writer.write_multiclass_probabilities(
        "multiclass-classification",
        {
            {
                "sanity_multiclass_classification",
                0,
                "multiclass_classification",
                "wine",
                "DummyMulticlassClassifier",
                "test",
                2.0,
                {0.1, 0.2, 0.7}
            }
        },
        3,
        false
    );

    expect_file_exists_and_non_empty(
        "Multiclass metrics output exists",
        "outputs/practical-exercises/multiclass-classification/metrics.csv"
    );
    expect_file_exists_and_non_empty(
        "Multiclass predictions output exists",
        "outputs/practical-exercises/multiclass-classification/predictions.csv"
    );
    expect_file_exists_and_non_empty(
        "Multiclass probabilities output exists",
        "outputs/practical-exercises/multiclass-classification/probabilities.csv"
    );
}

void test_output_writer_unsupervised_outputs() {
    const ml::workflows::OutputWriter writer("outputs/practical-exercises");

    writer.write_metrics(
        "unsupervised",
        {
            {
                "sanity_unsupervised",
                "unsupervised",
                "stock_ohlcv_engineered",
                "KMeans",
                "full",
                "inertia",
                12.5
            }
        },
        false
    );

    writer.write_projections_2d(
        "unsupervised",
        {
            {
                "sanity_unsupervised",
                0,
                "unsupervised",
                "stock_ohlcv_engineered",
                "PCA",
                "full",
                -1.0,
                0.5,
                ""
            }
        },
        false
    );

    writer.write_clustering_assignments(
        "unsupervised",
        {
            {
                "sanity_unsupervised",
                0,
                "unsupervised",
                "stock_ohlcv_engineered",
                "KMeans",
                "full",
                1,
                ""
            }
        },
        false
    );

    expect_file_exists_and_non_empty(
        "Unsupervised metrics output exists",
        "outputs/practical-exercises/unsupervised/metrics.csv"
    );
    expect_file_exists_and_non_empty(
        "Unsupervised projections output exists",
        "outputs/practical-exercises/unsupervised/projections.csv"
    );
    expect_file_exists_and_non_empty(
        "Unsupervised clustering assignments output exists",
        "outputs/practical-exercises/unsupervised/clustering_assignments.csv"
    );
}

void test_output_writer_training_outputs() {
    const ml::workflows::OutputWriter writer("outputs/practical-exercises");

    writer.write_loss_history(
        "binary-classification",
        {
            {
                "sanity_binary_classification",
                "binary_classification",
                "nasa_kc1_software_defects",
                "DummyBinaryClassifier",
                "train",
                0,
                0.693
            }
        },
        false
    );

    writer.write_hyperparameter_sweep(
        "binary-classification",
        {
            {
                "sanity_binary_classification_lr_0_01",
                "binary_classification",
                "nasa_kc1_software_defects",
                "DummyBinaryClassifier",
                "test",
                "learning_rate",
                "0.01",
                "accuracy",
                0.75
            }
        },
        false
    );

    expect_file_exists_and_non_empty(
        "Loss history output exists",
        "outputs/practical-exercises/binary-classification/loss_history.csv"
    );
    expect_file_exists_and_non_empty(
        "Hyperparameter sweep output exists",
        "outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv"
    );
}

void test_regression_model_comparison_workflow() {
    ml::workflows::RegressionComparisonConfig config;

    config.dataset_path = "data/processed/stock_ohlcv_engineered.csv";
    config.output_root = "outputs/practical-exercises";
    config.workflow_folder = "regression";
    config.workflow_name = "regression";
    config.dataset_name = "stock_ohlcv_engineered";

    config.feature_columns = stock_feature_columns();
    config.target_column = "target_next_return";

    config.test_ratio = 0.2;
    config.shuffle = true;
    config.seed = 42;

    // Keep the sanity test fast.
    // Full-dataset regression comparisons should be run as dedicated practical exercises,
    // not as part of the global test runner.
    config.max_rows = 500;

    config.standardize_features = true;
    config.export_predictions = true;
    config.export_loss_history = true;

    config.linear_regression_options.learning_rate = 0.01;
    config.linear_regression_options.max_iterations = 300;
    config.linear_regression_options.tolerance = 1e-8;
    config.linear_regression_options.store_loss_history = true;

    config.decision_tree_options.max_depth = 3;
    config.decision_tree_options.min_samples_split = 20;
    config.decision_tree_options.min_samples_leaf = 10;
    config.decision_tree_options.min_error_decrease = 0.0;

    config.gradient_boosting_options.n_estimators = 5;
    config.gradient_boosting_options.learning_rate = 0.05;
    config.gradient_boosting_options.max_depth = 2;
    config.gradient_boosting_options.min_samples_split = 20;
    config.gradient_boosting_options.min_samples_leaf = 10;
    config.gradient_boosting_options.random_seed = 42;

    std::cout << "Running regression comparison sanity workflow...\n";
    const auto summary = ml::workflows::run_regression_comparison(config);

    expect_true("Regression comparison total rows is positive", summary.total_rows > 0);
    expect_true("Regression comparison train rows is positive", summary.train_rows > 0);
    expect_true("Regression comparison test rows is positive", summary.test_rows > 0);
    expect_size_equal("Regression comparison feature count", summary.feature_count, static_cast<std::size_t>(5));

    // 4 models x 4 metrics: mse, rmse, mae, r2
    expect_size_equal("Regression comparison metric row count", summary.metrics.size(), static_cast<std::size_t>(16));

    expect_file_exists_and_non_empty(
        "Regression comparison metrics output exists",
        "outputs/practical-exercises/regression/metrics.csv"
    );

    expect_file_exists_and_non_empty(
        "Regression comparison predictions output exists",
        "outputs/practical-exercises/regression/predictions.csv"
    );

    expect_file_exists_and_non_empty(
        "Regression comparison loss history output exists",
        "outputs/practical-exercises/regression/loss_history.csv"
    );
}

void test_binary_classification_model_comparison_workflow() {
    ml::workflows::BinaryClassificationComparisonConfig config;

    config.dataset_path = "data/processed/nasa_kc1_software_defects.csv";
    config.output_root = "outputs/practical-exercises";
    config.workflow_folder = "binary-classification";
    config.workflow_name = "binary_classification";
    config.dataset_name = "nasa_kc1_software_defects";

    config.feature_columns = kc1_feature_columns();
    config.target_column = "defects";

    config.test_ratio = 0.2;
    config.shuffle = true;
    config.seed = 42;

    // Keep the sanity test fast.
    // Full-dataset binary comparisons should be run as dedicated practical exercises.
    config.max_rows = 800;

    config.standardize_features = true;
    config.export_predictions = true;
    config.export_probabilities = true;
    config.export_decision_scores = true;
    config.export_loss_history = true;

    config.logistic_regression_options.learning_rate = 0.01;
    config.logistic_regression_options.max_iterations = 300;
    config.logistic_regression_options.tolerance = 1e-8;
    config.logistic_regression_options.store_loss_history = true;

    config.linear_svm_options.learning_rate = 0.01;
    config.linear_svm_options.max_epochs = 50;
    config.linear_svm_options.l2_lambda = 0.01;

    config.gaussian_naive_bayes_options.variance_smoothing = 1e-9;

    config.decision_tree_options.max_depth = 4;
    config.decision_tree_options.min_samples_split = 20;
    config.decision_tree_options.min_samples_leaf = 10;
    config.decision_tree_options.min_impurity_decrease = 0.0;
    config.decision_tree_options.use_balanced_class_weight = true;
    config.decision_tree_options.random_seed = 42;

    config.random_forest_options.n_estimators = 10;
    config.random_forest_options.bootstrap = true;
    config.random_forest_options.max_features = std::size_t{5};
    config.random_forest_options.random_seed = 42;
    config.random_forest_options.tree_options = config.decision_tree_options;

    config.tiny_mlp_options.hidden_units = 8;
    config.tiny_mlp_options.learning_rate = 0.05;
    config.tiny_mlp_options.max_epochs = 100;
    config.tiny_mlp_options.batch_size = 32;
    config.tiny_mlp_options.random_seed = 42;

    std::cout << "Running binary classification comparison sanity workflow...\n";
    const auto summary = ml::workflows::run_binary_classification_comparison(config);

    expect_true("Binary comparison total rows is positive", summary.total_rows > 0);
    expect_true("Binary comparison train rows is positive", summary.train_rows > 0);
    expect_true("Binary comparison test rows is positive", summary.test_rows > 0);
    expect_size_equal("Binary comparison feature count", summary.feature_count, static_cast<std::size_t>(21));

    // 6 models x 4 metrics: accuracy, precision, recall, f1
    expect_size_equal("Binary comparison metric row count", summary.metrics.size(), static_cast<std::size_t>(24));

    expect_file_exists_and_non_empty(
        "Binary comparison metrics output exists",
        "outputs/practical-exercises/binary-classification/metrics.csv"
    );

    expect_file_exists_and_non_empty(
        "Binary comparison predictions output exists",
        "outputs/practical-exercises/binary-classification/predictions.csv"
    );

    expect_file_exists_and_non_empty(
        "Binary comparison probabilities output exists",
        "outputs/practical-exercises/binary-classification/probabilities.csv"
    );

    expect_file_exists_and_non_empty(
        "Binary comparison decision scores output exists",
        "outputs/practical-exercises/binary-classification/decision_scores.csv"
    );

    expect_file_exists_and_non_empty(
        "Binary comparison loss history output exists",
        "outputs/practical-exercises/binary-classification/loss_history.csv"
    );
}

void test_multiclass_classification_model_comparison_workflow() {
    ml::workflows::MulticlassClassificationComparisonConfig config;

    config.dataset_path = "data/processed/wine.csv";
    config.output_root = "outputs/practical-exercises";
    config.workflow_folder = "multiclass-classification";
    config.workflow_name = "multiclass_classification";
    config.dataset_name = "wine";

    config.feature_columns = wine_feature_columns();
    config.target_column = "class";
    config.num_classes = 3;

    config.test_ratio = 0.2;
    config.shuffle = true;
    config.seed = 42;

    config.max_rows = 0;

    config.standardize_features = true;
    config.export_predictions = true;
    config.export_probabilities = true;
    config.export_loss_history = true;

    config.softmax_regression_options.learning_rate = 0.01;
    config.softmax_regression_options.max_iterations = 300;
    config.softmax_regression_options.tolerance = 1e-8;
    config.softmax_regression_options.store_loss_history = true;

    config.knn_options.k = 5;
    config.knn_options.distance_metric = ml::DistanceMetric::Euclidean;

    config.gaussian_naive_bayes_options.variance_smoothing = 1e-9;

    config.decision_tree_options.max_depth = 4;
    config.decision_tree_options.min_samples_split = 4;
    config.decision_tree_options.min_samples_leaf = 2;
    config.decision_tree_options.min_impurity_decrease = 0.0;
    config.decision_tree_options.use_balanced_class_weight = false;
    config.decision_tree_options.random_seed = 42;

    config.random_forest_options.n_estimators = 10;
    config.random_forest_options.bootstrap = true;
    config.random_forest_options.max_features = std::size_t{4};
    config.random_forest_options.random_seed = 42;
    config.random_forest_options.tree_options = config.decision_tree_options;

    std::cout << "Running multiclass classification comparison sanity workflow...\n";
    const auto summary = ml::workflows::run_multiclass_classification_comparison(config);

    expect_true("Multiclass comparison total rows is positive", summary.total_rows > 0);
    expect_true("Multiclass comparison train rows is positive", summary.train_rows > 0);
    expect_true("Multiclass comparison test rows is positive", summary.test_rows > 0);
    expect_size_equal("Multiclass comparison feature count", summary.feature_count, static_cast<std::size_t>(13));
    expect_size_equal("Multiclass comparison class count", static_cast<std::size_t>(summary.num_classes), static_cast<std::size_t>(3));

    // 5 models x 4 metrics: accuracy, macro_precision, macro_recall, macro_f1
    expect_size_equal("Multiclass comparison metric row count", summary.metrics.size(), static_cast<std::size_t>(20));

    expect_file_exists_and_non_empty(
        "Multiclass comparison metrics output exists",
        "outputs/practical-exercises/multiclass-classification/metrics.csv"
    );

    expect_file_exists_and_non_empty(
        "Multiclass comparison predictions output exists",
        "outputs/practical-exercises/multiclass-classification/predictions.csv"
    );

    expect_file_exists_and_non_empty(
        "Multiclass comparison probabilities output exists",
        "outputs/practical-exercises/multiclass-classification/probabilities.csv"
    );

    expect_file_exists_and_non_empty(
        "Multiclass comparison loss history output exists",
        "outputs/practical-exercises/multiclass-classification/loss_history.csv"
    );
}

void test_unsupervised_model_comparison_workflow() {
    ml::workflows::UnsupervisedComparisonConfig config;

    config.dataset_path = "data/processed/stock_ohlcv_engineered.csv";
    config.output_root = "outputs/practical-exercises";
    config.workflow_folder = "unsupervised";
    config.workflow_name = "unsupervised";
    config.dataset_name = "stock_ohlcv_engineered";

    config.feature_columns = stock_feature_columns();

    config.max_rows = 1000;
    config.seed = 42;

    config.standardize_features = true;
    config.export_projections = true;
    config.export_clustering_assignments = true;

    config.pca_options.num_components = 2;

    config.kmeans_options.num_clusters = 4;
    config.kmeans_options.max_iterations = 50;
    config.kmeans_options.tolerance = 1e-6;

    config.pca_kmeans_options.num_clusters = 4;
    config.pca_kmeans_options.max_iterations = 50;
    config.pca_kmeans_options.tolerance = 1e-6;

    std::cout << "Running unsupervised comparison sanity workflow...\n";
    const auto summary = ml::workflows::run_unsupervised_comparison(config);

    expect_true("Unsupervised comparison total rows is positive", summary.total_rows > 0);
    expect_size_equal("Unsupervised comparison feature count", summary.feature_count, static_cast<std::size_t>(5));
    expect_size_equal("Unsupervised comparison PCA component count", summary.projection_components, static_cast<std::size_t>(2));
    expect_size_equal("Unsupervised comparison KMeans cluster count", summary.kmeans_clusters, static_cast<std::size_t>(4));
    expect_size_equal("Unsupervised comparison PCA+KMeans cluster count", summary.pca_kmeans_clusters, static_cast<std::size_t>(4));

    expect_file_exists_and_non_empty(
        "Unsupervised comparison metrics output exists",
        "outputs/practical-exercises/unsupervised/metrics.csv"
    );

    expect_file_exists_and_non_empty(
        "Unsupervised comparison projections output exists",
        "outputs/practical-exercises/unsupervised/projections.csv"
    );

    expect_file_exists_and_non_empty(
        "Unsupervised comparison clustering assignments output exists",
        "outputs/practical-exercises/unsupervised/clustering_assignments.csv"
    );
}

void test_hyperparameter_sweep_workflows() {
    ml::workflows::HyperparameterSweepConfig config;

    config.output_root = "outputs/practical-exercises";

    config.regression_dataset_path = "data/processed/stock_ohlcv_engineered.csv";
    config.binary_dataset_path = "data/processed/nasa_kc1_software_defects.csv";
    config.multiclass_dataset_path = "data/processed/wine.csv";
    config.unsupervised_dataset_path = "data/processed/stock_ohlcv_engineered.csv";

    config.regression_dataset_name = "stock_ohlcv_engineered";
    config.binary_dataset_name = "nasa_kc1_software_defects";
    config.multiclass_dataset_name = "wine";
    config.unsupervised_dataset_name = "stock_ohlcv_engineered";

    config.regression_feature_columns = stock_feature_columns();
    config.regression_target_column = "target_next_return";

    config.binary_feature_columns = kc1_feature_columns();
    config.binary_target_column = "defects";

    config.multiclass_feature_columns = wine_feature_columns();
    config.multiclass_target_column = "class";
    config.multiclass_num_classes = 3;

    config.unsupervised_feature_columns = stock_feature_columns();

    config.test_ratio = 0.2;
    config.shuffle = true;
    config.seed = 42;
    config.standardize_features = true;

    // Keep sanity sweeps fast.
    config.regression_max_rows = 400;
    config.binary_max_rows = 600;
    config.multiclass_max_rows = 0;
    config.unsupervised_max_rows = 800;

    config.gradient_boosting_n_estimators_values = {5, 10};
    config.gradient_boosting_learning_rate_values = {0.03, 0.1};

    config.logistic_regression_learning_rate_values = {0.005, 0.01};
    config.logistic_regression_ridge_lambda_values = {0.0, 0.01};

    config.decision_tree_max_depth_values = {2, 4};
    config.decision_tree_min_samples_leaf_values = {1, 10};

    config.random_forest_n_estimators_values = {5, 10};
    config.random_forest_max_features_values = {4, 8};

    config.tiny_mlp_hidden_units_values = {4, 8};
    config.tiny_mlp_learning_rate_values = {0.01, 0.05};
    config.tiny_mlp_max_epochs_values = {50};

    config.knn_k_values = {1, 3, 5};

    config.pca_num_components_values = {2, 3};
    config.kmeans_num_clusters_values = {2, 4};

    std::cout << "Running hyperparameter sweep sanity workflows...\n";
    const auto summary = ml::workflows::run_hyperparameter_sweeps(config);

    expect_true("Regression sweep used rows", summary.regression_rows > 0);
    expect_true("Binary sweep used rows", summary.binary_rows > 0);
    expect_true("Multiclass sweep used rows", summary.multiclass_rows > 0);
    expect_true("Unsupervised sweep used rows", summary.unsupervised_rows > 0);

    expect_true("Regression sweep rows exported", summary.regression_sweep_rows > 0);
    expect_true("Binary sweep rows exported", summary.binary_sweep_rows > 0);
    expect_true("Multiclass sweep rows exported", summary.multiclass_sweep_rows > 0);
    expect_true("Unsupervised sweep rows exported", summary.unsupervised_sweep_rows > 0);

    expect_file_exists_and_non_empty(
        "Regression hyperparameter sweep output exists",
        "outputs/practical-exercises/regression/hyperparameter_sweep.csv"
    );

    expect_file_exists_and_non_empty(
        "Binary hyperparameter sweep output exists",
        "outputs/practical-exercises/binary-classification/hyperparameter_sweep.csv"
    );

    expect_file_exists_and_non_empty(
        "Multiclass hyperparameter sweep output exists",
        "outputs/practical-exercises/multiclass-classification/hyperparameter_sweep.csv"
    );

    expect_file_exists_and_non_empty(
        "Unsupervised hyperparameter sweep output exists",
        "outputs/practical-exercises/unsupervised/hyperparameter_sweep.csv"
    );
}

// -----------------------------------------------------------------------------
// Test runners
// -----------------------------------------------------------------------------

void run_practical_workflow_tests() {
    std::cout << "\n[Phase 11.1] Practical workflow runner sanity tests\n\n";

    run_test(
        "Regression workflow runner loads stock OHLCV dataset",
        test_regression_workflow_runner
    );

    run_test(
        "Binary classification workflow runner loads NASA KC1 dataset",
        test_binary_classification_workflow_runner
    );

    run_test(
        "Multiclass classification workflow runner loads Wine dataset",
        test_multiclass_classification_workflow_runner
    );

    run_test(
        "Unsupervised workflow runner loads stock OHLCV dataset",
        test_unsupervised_workflow_runner
    );

    run_test(
        "OutputWriter writes regression CSV outputs",
        test_output_writer_regression_outputs
    );

    run_test(
        "OutputWriter writes binary classification CSV outputs",
        test_output_writer_binary_classification_outputs
    );

    run_test(
        "OutputWriter writes multiclass classification CSV outputs",
        test_output_writer_multiclass_classification_outputs
    );

    run_test(
        "OutputWriter writes unsupervised CSV outputs",
        test_output_writer_unsupervised_outputs
    );

    run_test(
        "OutputWriter writes training and sweep CSV outputs",
        test_output_writer_training_outputs
    );
}

void run_models_comparison_tests() {
    std::cout << "\n[Phase 11.2] Practical workflow runner sanity tests\n\n";

    run_test(
        "Regression model comparison workflow exports real results",
        test_regression_model_comparison_workflow
    );

    run_test(
        "Binary classification model comparison workflow exports real results",
        test_binary_classification_model_comparison_workflow
    );

    run_test(
        "Multiclass classification model comparison workflow exports real results",
        test_multiclass_classification_model_comparison_workflow
    );

    run_test(
        "Unsupervised model comparison workflow exports real results",
        test_unsupervised_model_comparison_workflow
    );
}

void run_hyperparameter_studies_tests() {
    std::cout << "\n[Phase 11.3] Hyperparameter studies runner sanity tests\n\n";

    run_test(
        "Hyperparameter sweep workflows export consistent results",
        test_hyperparameter_sweep_workflows
    );
}

}  // namespace

namespace ml::experiments {

void run_phase11_practical_workflows_sanity() {
    run_practical_workflow_tests();
    run_models_comparison_tests();
    run_hyperparameter_studies_tests();
}

}  // namespace ml::experiments