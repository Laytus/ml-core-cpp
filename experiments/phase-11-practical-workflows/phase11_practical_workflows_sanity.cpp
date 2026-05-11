#include "phase11_practical_workflows_sanity.hpp"

#include "ml/common/csv_dataset_loader.hpp"
#include "ml/workflows/output_writer.hpp"

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

}  // namespace

namespace ml::experiments {

void run_phase11_practical_workflows_sanity() {
    run_practical_workflow_tests();
}

}  // namespace ml::experiments