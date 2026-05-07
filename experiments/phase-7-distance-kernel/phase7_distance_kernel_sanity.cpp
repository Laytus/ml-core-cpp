#include "phase7_distance_kernel_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/types.hpp"
#include "ml/common/classification_metrics.hpp"
#include "ml/distance/distance_metrics.hpp"
#include "ml/distance/knn_classifier.hpp"
#include "ml/distance/kernels.hpp"
#include "ml/linear_models/linear_svm.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace test = ml::experiments::test;

namespace {

ml::Vector make_vector(std::initializer_list<double> values) {
    ml::Vector result(static_cast<Eigen::Index>(values.size()));

    Eigen::Index index = 0;
    for (double value : values) {
        result(index) = value;
        ++index;
    }

    return result;
}

// -----------------------------------------------------------------------------
// Phase 7.1 distance metric tests
// -----------------------------------------------------------------------------

void test_squared_euclidean_distance_computes_expected_value() {
    const ml::Vector a = make_vector({0.0, 0.0});
    const ml::Vector b = make_vector({3.0, 4.0});

    test::assert_almost_equal(
        ml::squared_euclidean_distance(a, b),
        25.0,
        "test_squared_euclidean_distance_computes_expected_value"
    );
}

void test_euclidean_distance_computes_expected_value() {
    const ml::Vector a = make_vector({0.0, 0.0});
    const ml::Vector b = make_vector({3.0, 4.0});

    test::assert_almost_equal(
        ml::euclidean_distance(a, b),
        5.0,
        "test_euclidean_distance_computes_expected_value"
    );
}

void test_manhattan_distance_computes_expected_value() {
    const ml::Vector a = make_vector({0.0, 0.0});
    const ml::Vector b = make_vector({3.0, 4.0});

    test::assert_almost_equal(
        ml::manhattan_distance(a, b),
        7.0,
        "test_manhattan_distance_computes_expected_value"
    );
}

void test_distance_metrics_return_zero_for_identical_vectors() {
    const ml::Vector a = make_vector({1.0, -2.0, 3.5});
    const ml::Vector b = make_vector({1.0, -2.0, 3.5});

    test::assert_almost_equal(
        ml::squared_euclidean_distance(a, b),
        0.0,
        "squared distance identical vectors"
    );

    test::assert_almost_equal(
        ml::euclidean_distance(a, b),
        0.0,
        "euclidean distance identical vectors"
    );

    test::assert_almost_equal(
        ml::manhattan_distance(a, b),
        0.0,
        "manhattan distance identical vectors"
    );
}

void test_distance_metrics_are_symmetric() {
    const ml::Vector a = make_vector({1.0, 2.0, 3.0});
    const ml::Vector b = make_vector({4.0, -2.0, 1.0});

    test::assert_almost_equal(
        ml::squared_euclidean_distance(a, b),
        ml::squared_euclidean_distance(b, a),
        "squared euclidean symmetry"
    );

    test::assert_almost_equal(
        ml::euclidean_distance(a, b),
        ml::euclidean_distance(b, a),
        "euclidean symmetry"
    );

    test::assert_almost_equal(
        ml::manhattan_distance(a, b),
        ml::manhattan_distance(b, a),
        "manhattan symmetry"
    );
}

void test_squared_euclidean_preserves_euclidean_ordering() {
    const ml::Vector query = make_vector({0.0, 0.0});
    const ml::Vector near = make_vector({1.0, 1.0});
    const ml::Vector far = make_vector({3.0, 4.0});

    const double euclidean_near =
        ml::euclidean_distance(query, near);

    const double euclidean_far =
        ml::euclidean_distance(query, far);

    const double squared_near =
        ml::squared_euclidean_distance(query, near);

    const double squared_far =
        ml::squared_euclidean_distance(query, far);

    if (!(euclidean_near < euclidean_far)) {
        throw std::runtime_error("expected near point to be closer by Euclidean distance");
    }

    if (!(squared_near < squared_far)) {
        throw std::runtime_error("expected squared Euclidean to preserve ordering");
    }
}

void test_distance_metrics_reject_empty_vectors() {
    const ml::Vector a;
    const ml::Vector b = make_vector({1.0, 2.0});

    static_cast<void>(
        ml::euclidean_distance(
            a,
            b
        )
    );
}

void test_distance_metrics_reject_mismatched_vectors() {
    const ml::Vector a = make_vector({1.0, 2.0});
    const ml::Vector b = make_vector({1.0, 2.0, 3.0});

    static_cast<void>(
        ml::euclidean_distance(
            a,
            b
        )
    );
}

void test_distance_metrics_reject_non_finite_values() {
    const ml::Vector a = make_vector({
        1.0,
        std::numeric_limits<double>::quiet_NaN()
    });

    const ml::Vector b = make_vector({
        1.0,
        2.0
    });

    static_cast<void>(
        ml::manhattan_distance(
            a,
            b
        )
    );
}

// -----------------------------------------------------------------------------
// Phase 7.2 multivariate KNNClassifier tests
// -----------------------------------------------------------------------------

ml::Matrix make_knn_test_X() {
    ml::Matrix X(6, 2);
    X << 0.0, 0.0,
         0.5, 0.0,
         1.0, 0.0,
         4.0, 4.0,
         4.5, 4.0,
         5.0, 4.0;

    return X;
}

ml::Vector make_knn_test_y() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    });
}

void test_knn_options_accept_defaults() {
    const ml::KNNClassifierOptions options;

    ml::validate_knn_classifier_options(
        options,
        "test_knn_options_accept_defaults"
    );
}

void test_knn_options_reject_zero_k() {
    ml::KNNClassifierOptions options;
    options.k = 0;

    ml::validate_knn_classifier_options(
        options,
        "test_knn_options_reject_zero_k"
    );
}

void test_distance_metric_name_returns_expected_values() {
    if (ml::distance_metric_name(ml::DistanceMetric::Euclidean) != "euclidean") {
        throw std::runtime_error("expected euclidean metric name");
    }

    if (
        ml::distance_metric_name(ml::DistanceMetric::SquaredEuclidean) !=
        "squared_euclidean"
    ) {
        throw std::runtime_error("expected squared_euclidean metric name");
    }

    if (ml::distance_metric_name(ml::DistanceMetric::Manhattan) != "manhattan") {
        throw std::runtime_error("expected manhattan metric name");
    }
}

void test_knn_reports_not_fitted_initially() {
    const ml::KNNClassifier model;

    if (model.is_fitted()) {
        throw std::runtime_error("expected KNNClassifier to start unfitted");
    }
}

void test_knn_fit_marks_model_as_fitted() {
    const ml::Matrix X = make_knn_test_X();
    const ml::Vector y = make_knn_test_y();
    
    ml::KNNClassifier model;
    model.fit(X, y);
    
    if (!model.is_fitted()) {
        throw std::runtime_error("expected KNNClassifier to be fitted");
    }
    
    if (model.num_train_samples() != static_cast<std::size_t>(X.rows())) {
        throw std::runtime_error("expected stored training sample count");
    }
    
    if (model.num_features() != X.cols()) {
        throw std::runtime_error("expected stored feature count");
    }
}

void test_knn_predict_returns_expected_shape() {
    const ml::Matrix X = make_knn_test_X();
    const ml::Vector y = make_knn_test_y();

    ml::KNNClassifierOptions options;
    options.k = 3;

    ml::KNNClassifier model(options);
    model.fit(X, y);

    const ml::Vector predictions = model.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected one prediction per sample");
    }
}

void test_knn_predicts_simple_clusters_with_euclidean() {
    const ml::Matrix X = make_knn_test_X();
    const ml::Vector y = make_knn_test_y();

    ml::KNNClassifierOptions options;
    options.k = 3;
    options.distance_metric = ml::DistanceMetric::Euclidean;

    ml::KNNClassifier model(options);
    model.fit(X, y);

    ml::Matrix X_query(2, 2);
    X_query << 0.25, 0.0,
               4.75, 4.0;

    const ml::Vector predictions = model.predict(X_query);

    test::assert_almost_equal(
        predictions(0),
        0.0,
        "test_knn_predicts_simple_clusters_with_euclidean first prediction"
    );

    test::assert_almost_equal(
        predictions(1),
        1.0,
        "test_knn_predicts_simple_clusters_with_euclidean second prediction"
    );
}

void test_knn_predicts_simple_clusters_with_manhattan() {
    const ml::Matrix X = make_knn_test_X();
    const ml::Vector y = make_knn_test_y();

    ml::KNNClassifierOptions options;
    options.k = 3;
    options.distance_metric = ml::DistanceMetric::Manhattan;

    ml::KNNClassifier model(options);
    model.fit(X, y);

    ml::Matrix X_query(2, 2);
    X_query << 0.25, 0.0,
               4.75, 4.0;

    const ml::Vector predictions = model.predict(X_query);

    test::assert_almost_equal(
        predictions(0),
        0.0,
        "test_knn_predicts_simple_clusters_with_manhattan first prediction"
    );

    test::assert_almost_equal(
        predictions(1),
        1.0,
        "test_knn_predicts_simple_clusters_with_manhattan second prediction"
    );
}

void test_knn_predicts_simple_clusters_with_squared_euclidean() {
    const ml::Matrix X = make_knn_test_X();
    const ml::Vector y = make_knn_test_y();

    ml::KNNClassifierOptions options;
    options.k = 3;
    options.distance_metric = ml::DistanceMetric::SquaredEuclidean;

    ml::KNNClassifier model(options);
    model.fit(X, y);

    ml::Matrix X_query(2, 2);
    X_query << 0.25, 0.0,
               4.75, 4.0;

    const ml::Vector predictions = model.predict(X_query);

    test::assert_almost_equal(
        predictions(0),
        0.0,
        "test_knn_predicts_simple_clusters_with_squared_euclidean first prediction"
    );

    test::assert_almost_equal(
        predictions(1),
        1.0,
        "test_knn_predicts_simple_clusters_with_squared_euclidean second prediction"
    );
}

void test_knn_uses_smallest_label_tie_break() {
    ml::Matrix X(2, 2);
    X << -1.0, 0.0,
          1.0, 0.0;

    const ml::Vector y = make_vector({
        0.0,
        1.0
    });

    ml::KNNClassifierOptions options;
    options.k = 2;
    options.distance_metric = ml::DistanceMetric::Euclidean;

    ml::KNNClassifier model(options);
    model.fit(X, y);

    ml::Matrix X_query(1, 2);
    X_query << 0.0, 0.0;

    const ml::Vector predictions = model.predict(X_query);

    test::assert_almost_equal(
        predictions(0),
        0.0,
        "test_knn_uses_smallest_label_tie_break"
    );
}

void test_knn_rejects_predict_before_fit() {
    const ml::Matrix X = make_knn_test_X();

    const ml::KNNClassifier model;

    static_cast<void>(model.predict(X));
}

void test_knn_rejects_empty_fit_X() {
    const ml::Matrix X;
    const ml::Vector y = make_knn_test_y();

    ml::KNNClassifier model;
    model.fit(X, y);
}

void test_knn_rejects_empty_fit_y() {
    const ml::Matrix X = make_knn_test_X();
    const ml::Vector y;

    ml::KNNClassifier model;
    model.fit(X, y);
}

void test_knn_rejects_mismatched_fit_data() {
    ml::Matrix X(3, 2);
    X.setZero();

    const ml::Vector y = make_vector({
        0.0,
        1.0
    });

    ml::KNNClassifier model;
    model.fit(X, y);
}

void test_knn_rejects_invalid_class_labels() {
    const ml::Matrix X = make_knn_test_X();
    ml::Vector y = make_knn_test_y();
    y(0) = 0.5;

    ml::KNNClassifier model;
    model.fit(X, y);
}

void test_knn_rejects_k_larger_than_training_size() {
    const ml::Matrix X = make_knn_test_X();
    const ml::Vector y = make_knn_test_y();

    ml::KNNClassifierOptions options;
    options.k = static_cast<std::size_t>(X.rows()) + 1;

    ml::KNNClassifier model(options);
    model.fit(X, y);
}

void test_knn_rejects_predict_feature_mismatch() {
    const ml::Matrix X = make_knn_test_X();
    const ml::Vector y = make_knn_test_y();

    ml::KNNClassifier model;
    model.fit(X, y);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(model.predict(X_bad));
}

void test_knn_rejects_non_finite_predict_values() {
    const ml::Matrix X = make_knn_test_X();
    const ml::Vector y = make_knn_test_y();

    ml::KNNClassifier model;
    model.fit(X, y);

    ml::Matrix X_bad = X;
    X_bad(0, 0) = std::numeric_limits<double>::quiet_NaN();

    static_cast<void>(model.predict(X_bad));
}

// -----------------------------------------------------------------------------
// Phase 7.3 KNN experiment export helpers
// -----------------------------------------------------------------------------

const std::string k_phase7_output_dir = "outputs/phase-7-distance-kernel";

void ensure_phase7_output_dir_exists() {
    std::filesystem::create_directories(k_phase7_output_dir);
}

struct KNNExperimentResult {
    std::string experiment_name;
    std::string variant_name;
    std::size_t k{0};
    std::string distance_metric;
    double accuracy{0.0};
    std::size_t num_predictions{0};
};

struct KNNNeighborhoodRow {
    std::string query_name;
    double query_x0{0.0};
    double query_x1{0.0};
    std::size_t k{0};
    std::string distance_metric;
    double prediction{0.0};
    std::string interpretation;
};

ml::Matrix make_knn_experiment_X() {
    ml::Matrix X(10, 2);

    X << 0.0, 0.0,
         0.5, 0.0,
         1.0, 0.0,
         1.5, 0.1,
         2.0, 0.0,

         3.0, 3.0,
         3.5, 3.0,
         4.0, 3.0,
         4.5, 3.1,
         5.0, 3.0;

    return X;
}

ml::Vector make_knn_experiment_y() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,

        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    });
}

ml::Matrix make_knn_experiment_X_eval() {
    ml::Matrix X(6, 2);

    X << 0.25, 0.0,
         1.25, 0.0,
         1.75, 0.1,
         3.25, 3.0,
         4.25, 3.0,
         4.75, 3.1;

    return X;
}

ml::Vector make_knn_experiment_y_eval() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    });
}

ml::Matrix make_knn_neighborhood_queries() {
    ml::Matrix X(3, 2);

    X << 0.25, 0.0,
         2.5, 1.5,
         4.75, 3.0;

    return X;
}

KNNExperimentResult run_knn_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::KNNClassifierOptions& options
) {
    ml::KNNClassifier model(options);
    model.fit(X_train, y_train);

    const ml::Vector predictions = model.predict(X_eval);

    return KNNExperimentResult{
        experiment_name,
        variant_name,
        options.k,
        ml::distance_metric_name(options.distance_metric),
        ml::accuracy_score(predictions, y_eval),
        static_cast<std::size_t>(predictions.size())
    };
}

std::string neighborhood_interpretation(
    Eigen::Index query_index
) {
    if (query_index == 0) {
        return "near class 0 cluster";
    }

    if (query_index == 1) {
        return "between both clusters";
    }

    return "near class 1 cluster";
}

std::vector<KNNNeighborhoodRow> run_knn_neighborhood_behavior_demo(
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_query
) {
    std::vector<KNNNeighborhoodRow> rows;

    const std::vector<std::size_t> k_values = {1, 3, 5};
    const std::vector<ml::DistanceMetric> metrics = {
        ml::DistanceMetric::Euclidean,
        ml::DistanceMetric::Manhattan
    };

    for (std::size_t k : k_values) {
        for (ml::DistanceMetric metric : metrics) {
            ml::KNNClassifierOptions options;
            options.k = k;
            options.distance_metric = metric;

            ml::KNNClassifier model(options);
            model.fit(X_train, y_train);

            const ml::Vector predictions = model.predict(X_query);

            for (Eigen::Index i = 0; i < X_query.rows(); ++i) {
                rows.push_back(
                    KNNNeighborhoodRow{
                        "query_" + std::to_string(i),
                        X_query(i, 0),
                        X_query(i, 1),
                        k,
                        ml::distance_metric_name(metric),
                        predictions(i),
                        neighborhood_interpretation(i)
                    }
                );
            }
        }
    }

    return rows;
}

void export_knn_experiment_results_csv(
    const std::vector<KNNExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_knn_experiment_results_csv: failed to open output file"
        );
    }

    file << "experiment_name,variant_name,k,distance_metric,accuracy,num_predictions\n";

    for (const KNNExperimentResult& result : results) {
        file << result.experiment_name << ","
             << result.variant_name << ","
             << result.k << ","
             << result.distance_metric << ","
             << result.accuracy << ","
             << result.num_predictions << "\n";
    }
}

void export_knn_experiment_results_txt(
    const std::vector<KNNExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_knn_experiment_results_txt: failed to open output file"
        );
    }

    file << "KNN Metric and k Comparison\n\n";

    file << "This experiment compares:\n"
         << "- different k values\n"
         << "- Euclidean vs Manhattan distance\n"
         << "- local neighborhood behavior through separate query exports\n\n";

    for (const KNNExperimentResult& result : results) {
        file << "Experiment: " << result.experiment_name << "\n"
             << "Variant: " << result.variant_name << "\n"
             << "k: " << result.k << "\n"
             << "Distance metric: " << result.distance_metric << "\n"
             << "Accuracy: " << result.accuracy << "\n"
             << "Predictions: " << result.num_predictions << "\n\n";
    }
}

void export_knn_neighborhood_behavior_csv(
    const std::vector<KNNNeighborhoodRow>& rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_knn_neighborhood_behavior_csv: failed to open output file"
        );
    }

    file << "query_name,query_x0,query_x1,k,distance_metric,prediction,interpretation\n";

    for (const KNNNeighborhoodRow& row : rows) {
        file << row.query_name << ","
             << row.query_x0 << ","
             << row.query_x1 << ","
             << row.k << ","
             << row.distance_metric << ","
             << row.prediction << ","
             << row.interpretation << "\n";
    }
}

// -----------------------------------------------------------------------------
// Phase 7.3 KNN experiment export tests
// -----------------------------------------------------------------------------

void test_experiment_exports_knn_metric_comparison() {
    ensure_phase7_output_dir_exists();

    const ml::Matrix X_train = make_knn_experiment_X();
    const ml::Vector y_train = make_knn_experiment_y();

    const ml::Matrix X_eval = make_knn_experiment_X_eval();
    const ml::Vector y_eval = make_knn_experiment_y_eval();

    std::vector<KNNExperimentResult> results;

    for (std::size_t k : {1, 3, 5}) {
        {
            ml::KNNClassifierOptions options;
            options.k = k;
            options.distance_metric = ml::DistanceMetric::Euclidean;

            results.push_back(
                run_knn_experiment(
                    "different_k_values_and_metrics",
                    "k_" + std::to_string(k) + "_euclidean",
                    X_train,
                    y_train,
                    X_eval,
                    y_eval,
                    options
                )
            );
        }

        {
            ml::KNNClassifierOptions options;
            options.k = k;
            options.distance_metric = ml::DistanceMetric::Manhattan;

            results.push_back(
                run_knn_experiment(
                    "different_k_values_and_metrics",
                    "k_" + std::to_string(k) + "_manhattan",
                    X_train,
                    y_train,
                    X_eval,
                    y_eval,
                    options
                )
            );
        }
    }

    const ml::Matrix X_query = make_knn_neighborhood_queries();

    const std::vector<KNNNeighborhoodRow> neighborhood_rows =
        run_knn_neighborhood_behavior_demo(
            X_train,
            y_train,
            X_query
        );

    const std::string comparison_csv_path =
        k_phase7_output_dir + "/knn_metric_comparison.csv";

    const std::string comparison_txt_path =
        k_phase7_output_dir + "/knn_metric_comparison.txt";

    const std::string neighborhood_csv_path =
        k_phase7_output_dir + "/knn_local_neighborhood_behavior.csv";

    export_knn_experiment_results_csv(
        results,
        comparison_csv_path
    );

    export_knn_experiment_results_txt(
        results,
        comparison_txt_path
    );

    export_knn_neighborhood_behavior_csv(
        neighborhood_rows,
        neighborhood_csv_path
    );

    if (!std::filesystem::exists(comparison_csv_path)) {
        throw std::runtime_error(
            "expected knn_metric_comparison.csv to exist"
        );
    }

    if (!std::filesystem::exists(comparison_txt_path)) {
        throw std::runtime_error(
            "expected knn_metric_comparison.txt to exist"
        );
    }

    if (!std::filesystem::exists(neighborhood_csv_path)) {
        throw std::runtime_error(
            "expected knn_local_neighborhood_behavior.csv to exist"
        );
    }

    if (results.empty()) {
        throw std::runtime_error(
            "expected non-empty KNN comparison results"
        );
    }

    if (neighborhood_rows.empty()) {
        throw std::runtime_error(
            "expected non-empty KNN neighborhood behavior rows"
        );
    }
}

// -----------------------------------------------------------------------------
// Phase 7.4 reusable kernel function tests
// -----------------------------------------------------------------------------

void test_linear_kernel_computes_dot_product() {
    const ml::Vector a = make_vector({1.0, 2.0, 3.0});
    const ml::Vector b = make_vector({4.0, 5.0, 6.0});

    test::assert_almost_equal(
        ml::linear_kernel(a, b),
        32.0,
        "test_linear_kernel_computes_dot_product"
    );
}

void test_polynomial_kernel_computes_expected_value() {
    const ml::Vector a = make_vector({1.0, 2.0});
    const ml::Vector b = make_vector({3.0, 4.0});

    const double dot = 11.0;
    const double coef0 = 1.0;
    const double degree = 2.0;

    const double expected = std::pow(
        dot + coef0,
        degree
    );

    test::assert_almost_equal(
        ml::polynomial_kernel(
            a,
            b,
            degree,
            coef0
        ),
        expected,
        "test_polynomial_kernel_computes_expected_value"
    );
}

void test_rbf_kernel_returns_one_for_identical_vectors() {
    const ml::Vector a = make_vector({1.0, 2.0, 3.0});
    const ml::Vector b = make_vector({1.0, 2.0, 3.0});

    test::assert_almost_equal(
        ml::rbf_kernel(
            a,
            b,
            0.5
        ),
        1.0,
        "test_rbf_kernel_returns_one_for_identical_vectors"
    );
}

void test_rbf_kernel_decreases_with_distance() {
    const ml::Vector query = make_vector({0.0, 0.0});
    const ml::Vector near = make_vector({1.0, 0.0});
    const ml::Vector far = make_vector({3.0, 0.0});

    const double near_similarity =
        ml::rbf_kernel(
            query,
            near,
            0.5
        );

    const double far_similarity =
        ml::rbf_kernel(
            query,
            far,
            0.5
        );

    if (!(near_similarity > far_similarity)) {
        throw std::runtime_error(
            "expected RBF similarity to decrease as distance increases"
        );
    }
}

void test_kernel_functions_are_symmetric() {
    const ml::Vector a = make_vector({1.0, -2.0, 3.0});
    const ml::Vector b = make_vector({4.0, 0.5, -1.0});

    test::assert_almost_equal(
        ml::linear_kernel(a, b),
        ml::linear_kernel(b, a),
        "linear kernel symmetry"
    );

    test::assert_almost_equal(
        ml::polynomial_kernel(a, b, 2.0, 1.0),
        ml::polynomial_kernel(b, a, 2.0, 1.0),
        "polynomial kernel symmetry"
    );

    test::assert_almost_equal(
        ml::rbf_kernel(a, b, 0.25),
        ml::rbf_kernel(b, a, 0.25),
        "rbf kernel symmetry"
    );
}

void test_polynomial_kernel_degree_one_matches_shifted_linear_kernel() {
    const ml::Vector a = make_vector({1.0, 2.0});
    const ml::Vector b = make_vector({3.0, 4.0});

    const double coef0 = 2.0;

    test::assert_almost_equal(
        ml::polynomial_kernel(
            a,
            b,
            1.0,
            coef0
        ),
        ml::linear_kernel(a, b) + coef0,
        "test_polynomial_kernel_degree_one_matches_shifted_linear_kernel"
    );
}

void test_kernel_functions_reject_empty_vectors() {
    const ml::Vector a;
    const ml::Vector b = make_vector({1.0, 2.0});

    static_cast<void>(
        ml::linear_kernel(
            a,
            b
        )
    );
}

void test_kernel_functions_reject_mismatched_vectors() {
    const ml::Vector a = make_vector({1.0, 2.0});
    const ml::Vector b = make_vector({1.0, 2.0, 3.0});

    static_cast<void>(
        ml::rbf_kernel(
            a,
            b,
            0.5
        )
    );
}

void test_kernel_functions_reject_non_finite_values() {
    const ml::Vector a = make_vector({
        1.0,
        std::numeric_limits<double>::quiet_NaN()
    });

    const ml::Vector b = make_vector({
        1.0,
        2.0
    });

    static_cast<void>(
        ml::linear_kernel(
            a,
            b
        )
    );
}

void test_polynomial_kernel_rejects_invalid_degree() {
    const ml::Vector a = make_vector({1.0, 2.0});
    const ml::Vector b = make_vector({3.0, 4.0});

    static_cast<void>(
        ml::polynomial_kernel(
            a,
            b,
            0.0,
            1.0
        )
    );
}

void test_polynomial_kernel_rejects_non_finite_coef0() {
    const ml::Vector a = make_vector({1.0, 2.0});
    const ml::Vector b = make_vector({3.0, 4.0});

    static_cast<void>(
        ml::polynomial_kernel(
            a,
            b,
            2.0,
            std::numeric_limits<double>::infinity()
        )
    );
}

void test_rbf_kernel_rejects_invalid_gamma() {
    const ml::Vector a = make_vector({1.0, 2.0});
    const ml::Vector b = make_vector({3.0, 4.0});

    static_cast<void>(
        ml::rbf_kernel(
            a,
            b,
            0.0
        )
    );
}

// -----------------------------------------------------------------------------
// Phase 7.5 kernel similarity demo export helpers
// -----------------------------------------------------------------------------

struct KernelSimilarityDemoRow {
    std::string point_name;
    double x0{0.0};
    double x1{0.0};
    double squared_distance_to_query{0.0};
    double euclidean_distance_to_query{0.0};
    double linear_similarity{0.0};
    double polynomial_similarity{0.0};
    double rbf_similarity{0.0};
    std::string interpretation;
};

ml::Vector make_kernel_demo_query() {
    return make_vector({0.0, 0.0});
}

std::vector<std::pair<std::string, ml::Vector>> make_kernel_demo_points() {
    return {
        {"identical_point", make_vector({0.0, 0.0})},
        {"near_point", make_vector({1.0, 0.0})},
        {"diagonal_point", make_vector({1.0, 1.0})},
        {"far_point", make_vector({3.0, 0.0})},
        {"very_far_point", make_vector({5.0, 0.0})}
    };
}

std::string kernel_demo_interpretation(
    const std::string& point_name
) {
    if (point_name == "identical_point") {
        return "same as query; RBF similarity should be 1";
    }

    if (point_name == "near_point") {
        return "close to query; high RBF similarity";
    }

    if (point_name == "diagonal_point") {
        return "moderate distance from query";
    }

    if (point_name == "far_point") {
        return "far from query; low RBF similarity";
    }

    return "very far from query; RBF similarity should be very low";
}

std::vector<KernelSimilarityDemoRow> make_kernel_similarity_demo_rows() {
    const ml::Vector query = make_kernel_demo_query();

    const std::vector<std::pair<std::string, ml::Vector>> points =
        make_kernel_demo_points();

    std::vector<KernelSimilarityDemoRow> rows;
    rows.reserve(points.size());

    constexpr double polynomial_degree = 2.0;
    constexpr double polynomial_coef0 = 1.0;
    constexpr double rbf_gamma = 0.5;

    for (const auto& [name, point] : points) {
        rows.push_back(
            KernelSimilarityDemoRow{
                name,
                point(0),
                point(1),
                ml::squared_euclidean_distance(query, point),
                ml::euclidean_distance(query, point),
                ml::linear_kernel(query, point),
                ml::polynomial_kernel(
                    query,
                    point,
                    polynomial_degree,
                    polynomial_coef0
                ),
                ml::rbf_kernel(
                    query,
                    point,
                    rbf_gamma
                ),
                kernel_demo_interpretation(name)
            }
        );
    }

    return rows;
}

void export_kernel_similarity_demo_csv(
    const std::vector<KernelSimilarityDemoRow>& rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_kernel_similarity_demo_csv: failed to open output file"
        );
    }

    file << "point_name,x0,x1,squared_distance_to_query,"
         << "euclidean_distance_to_query,linear_similarity,"
         << "polynomial_similarity,rbf_similarity,interpretation\n";

    for (const KernelSimilarityDemoRow& row : rows) {
        file << row.point_name << ","
             << row.x0 << ","
             << row.x1 << ","
             << row.squared_distance_to_query << ","
             << row.euclidean_distance_to_query << ","
             << row.linear_similarity << ","
             << row.polynomial_similarity << ","
             << row.rbf_similarity << ","
             << row.interpretation << "\n";
    }
}

// -----------------------------------------------------------------------------
// Phase 7.5 kernel similarity demo export tests
// -----------------------------------------------------------------------------

void test_experiment_exports_kernel_similarity_demo() {
    ensure_phase7_output_dir_exists();

    const std::vector<KernelSimilarityDemoRow> rows =
        make_kernel_similarity_demo_rows();

    if (rows.empty()) {
        throw std::runtime_error(
            "expected non-empty kernel similarity demo rows"
        );
    }

    for (const KernelSimilarityDemoRow& row : rows) {
        if (!std::isfinite(row.squared_distance_to_query)) {
            throw std::runtime_error(
                "expected finite squared distance"
            );
        }

        if (!std::isfinite(row.euclidean_distance_to_query)) {
            throw std::runtime_error(
                "expected finite Euclidean distance"
            );
        }

        if (!std::isfinite(row.linear_similarity)) {
            throw std::runtime_error(
                "expected finite linear similarity"
            );
        }

        if (!std::isfinite(row.polynomial_similarity)) {
            throw std::runtime_error(
                "expected finite polynomial similarity"
            );
        }

        if (!std::isfinite(row.rbf_similarity)) {
            throw std::runtime_error(
                "expected finite RBF similarity"
            );
        }

        if (row.rbf_similarity < 0.0 || row.rbf_similarity > 1.0) {
            throw std::runtime_error(
                "expected RBF similarity to be in [0, 1]"
            );
        }
    }

    const ml::Vector query = make_kernel_demo_query();
    const ml::Vector identical = make_vector({0.0, 0.0});
    const ml::Vector near = make_vector({1.0, 0.0});
    const ml::Vector far = make_vector({3.0, 0.0});

    const double identical_rbf =
        ml::rbf_kernel(
            query,
            identical,
            0.5
        );

    const double near_rbf =
        ml::rbf_kernel(
            query,
            near,
            0.5
        );

    const double far_rbf =
        ml::rbf_kernel(
            query,
            far,
            0.5
        );

    if (!(identical_rbf > near_rbf && near_rbf > far_rbf)) {
        throw std::runtime_error(
            "expected RBF similarity to decrease as points move away from query"
        );
    }

    const std::string output_path =
        k_phase7_output_dir + "/kernel_similarity_demo.csv";

    export_kernel_similarity_demo_csv(
        rows,
        output_path
    );

    if (!std::filesystem::exists(output_path)) {
        throw std::runtime_error(
            "expected kernel_similarity_demo.csv to exist"
        );
    }
}

// -----------------------------------------------------------------------------
// Phase 7.6 SVM margin intuition demo export helpers
// -----------------------------------------------------------------------------

void export_svm_margin_intuition_txt(
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_svm_margin_intuition_txt: failed to open output file"
        );
    }

    file << "SVM Margin and Kernel Intuition Demo\n\n";

    file << "Scope decision:\n";
    file << "- Phase 7 keeps SVM and kernel SVM work at theory + small demo level.\n";
    file << "- No full SVM solver is implemented in the main Phase 7 scope.\n";
    file << "- A serious LinearSVM may be reconsidered after the Phase 7 core is complete.\n";
    file << "- Full kernel SVM / SMO / dual optimization is deferred.\n\n";

    file << "1. Margin intuition\n\n";

    file << "A linear classifier uses a score:\n\n";
    file << "    f(x) = w^T x + b\n\n";

    file << "The decision boundary is:\n\n";
    file << "    w^T x + b = 0\n\n";

    file << "The margin is the separation between the decision boundary and the closest training samples.\n";
    file << "SVMs prefer a larger margin because it usually gives a more stable separating boundary.\n\n";

    file << "2. Support vectors\n\n";

    file << "Support vectors are the closest samples to the margin.\n";
    file << "They are the points that directly constrain the separating boundary.\n";
    file << "Moving non-support-vector points often does not change the boundary, but moving a support vector can.\n\n";

    file << "3. Soft-margin intuition\n\n";

    file << "Hard-margin SVM assumes perfectly separable data.\n";
    file << "Soft-margin SVM allows margin violations or misclassified samples.\n\n";

    file << "This introduces a trade-off:\n\n";
    file << "    wider margin vs fewer violations\n\n";

    file << "A regularization parameter controls how expensive violations are.\n\n";

    file << "4. Hinge loss intuition\n\n";

    file << "For labels y in {-1, +1}, hinge loss is:\n\n";
    file << "    max(0, 1 - y * f(x))\n\n";

    file << "If y * f(x) >= 1, the point is correctly classified with enough margin.\n";
    file << "If y * f(x) < 1, the point violates the margin and receives positive loss.\n\n";

    file << "5. Kernel trick intuition\n\n";

    file << "A kernel computes similarity in a transformed feature space:\n\n";
    file << "    K(a, b) = phi(a)^T phi(b)\n\n";

    file << "The transformation phi(x) does not need to be constructed explicitly.\n";
    file << "This makes it possible to reason about nonlinear separation through similarity functions.\n\n";

    file << "Examples implemented in Phase 7:\n\n";
    file << "    linear_kernel(a, b) = a^T b\n";
    file << "    polynomial_kernel(a, b) = (a^T b + coef0)^degree\n";
    file << "    rbf_kernel(a, b) = exp(-gamma * ||a - b||^2)\n\n";

    file << "6. Why kernel SVM is deferred\n\n";

    file << "A proper kernel SVM requires dual optimization machinery:\n\n";
    file << "    kernel matrix\n";
    file << "    alpha coefficients\n";
    file << "    support vectors\n";
    file << "    box constraints\n";
    file << "    KKT conditions\n";
    file << "    SMO-style updates or another constrained optimization method\n\n";

    file << "That is too large for the main Phase 7 scope.\n";
    file << "Phase 7 therefore implements reusable kernel functions and keeps SVM as theory + demo.\n";
}

// -----------------------------------------------------------------------------
// Phase 7.6 SVM margin intuition demo export tests
// -----------------------------------------------------------------------------

void test_experiment_exports_svm_margin_intuition_demo() {
    ensure_phase7_output_dir_exists();

    const std::string output_path =
        k_phase7_output_dir + "/svm_margin_intuition.txt";

    export_svm_margin_intuition_txt(output_path);

    if (!std::filesystem::exists(output_path)) {
        throw std::runtime_error(
            "expected svm_margin_intuition.txt to exist"
        );
    }
}

// -----------------------------------------------------------------------------
// Phase 7.7 LinearSVMOptions tests
// -----------------------------------------------------------------------------

void test_linear_svm_options_accept_defaults() {
    const ml::LinearSVMOptions options;

    ml::validate_linear_svm_options(
        options,
        "test_linear_svm_options_accept_defaults"
    );
}

void test_linear_svm_options_accept_valid_values() {
    ml::LinearSVMOptions options;
    options.learning_rate = 0.05;
    options.max_epochs = 200;
    options.l2_lambda = 0.001;

    ml::validate_linear_svm_options(
        options,
        "test_linear_svm_options_accept_valid_values"
    );
}

void test_linear_svm_options_reject_zero_learning_rate() {
    ml::LinearSVMOptions options;
    options.learning_rate = 0.0;

    ml::validate_linear_svm_options(
        options,
        "test_linear_svm_options_reject_zero_learning_rate"
    );
}

void test_linear_svm_options_reject_negative_learning_rate() {
    ml::LinearSVMOptions options;
    options.learning_rate = -0.01;

    ml::validate_linear_svm_options(
        options,
        "test_linear_svm_options_reject_negative_learning_rate"
    );
}

void test_linear_svm_options_reject_non_finite_learning_rate() {
    ml::LinearSVMOptions options;
    options.learning_rate = std::numeric_limits<double>::infinity();

    ml::validate_linear_svm_options(
        options,
        "test_linear_svm_options_reject_non_finite_learning_rate"
    );
}

void test_linear_svm_options_reject_zero_max_epochs() {
    ml::LinearSVMOptions options;
    options.max_epochs = 0;

    ml::validate_linear_svm_options(
        options,
        "test_linear_svm_options_reject_zero_max_epochs"
    );
}

void test_linear_svm_options_accept_zero_l2_lambda() {
    ml::LinearSVMOptions options;
    options.l2_lambda = 0.0;

    ml::validate_linear_svm_options(
        options,
        "test_linear_svm_options_accept_zero_l2_lambda"
    );
}

void test_linear_svm_options_reject_negative_l2_lambda() {
    ml::LinearSVMOptions options;
    options.l2_lambda = -0.01;

    ml::validate_linear_svm_options(
        options,
        "test_linear_svm_options_reject_negative_l2_lambda"
    );
}

void test_linear_svm_options_reject_non_finite_l2_lambda() {
    ml::LinearSVMOptions options;
    options.l2_lambda = std::numeric_limits<double>::quiet_NaN();

    ml::validate_linear_svm_options(
        options,
        "test_linear_svm_options_reject_non_finite_l2_lambda"
    );
}

// -----------------------------------------------------------------------------
// Phase 7.8 LinearSVM tests
// -----------------------------------------------------------------------------

ml::Matrix make_linear_svm_test_X() {
    ml::Matrix X(8, 2);

    X << -3.0, -2.0,
         -2.5, -1.5,
         -2.0, -2.0,
         -1.5, -1.0,
          1.5,  1.0,
          2.0,  2.0,
          2.5,  1.5,
          3.0,  2.0;

    return X;
}

ml::Vector make_linear_svm_test_y() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0
    });
}

void test_linear_svm_reports_not_fitted_initially() {
    const ml::LinearSVM model;

    if (model.is_fitted()) {
        throw std::runtime_error(
            "expected LinearSVM to start unfitted"
        );
    }
}

void test_linear_svm_fit_marks_model_as_fitted() {
    const ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y = make_linear_svm_test_y();

    ml::LinearSVMOptions options;
    options.learning_rate = 0.01;
    options.max_epochs = 100;
    options.l2_lambda = 0.001;

    ml::LinearSVM model(options);
    model.fit(X, y);

    if (!model.is_fitted()) {
        throw std::runtime_error(
            "expected LinearSVM to be fitted"
        );
    }

    if (model.weights().size() != X.cols()) {
        throw std::runtime_error(
            "expected LinearSVM weight size to match feature count"
        );
    }
}

void test_linear_svm_decision_function_returns_expected_shape() {
    const ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y = make_linear_svm_test_y();

    ml::LinearSVMOptions options;
    options.learning_rate = 0.01;
    options.max_epochs = 100;
    options.l2_lambda = 0.001;

    ml::LinearSVM model(options);
    model.fit(X, y);

    const ml::Vector scores =
        model.decision_function(X);

    if (scores.size() != y.size()) {
        throw std::runtime_error(
            "expected one decision score per sample"
        );
    }
}

void test_linear_svm_predict_returns_expected_shape() {
    const ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y = make_linear_svm_test_y();

    ml::LinearSVMOptions options;
    options.learning_rate = 0.01;
    options.max_epochs = 100;
    options.l2_lambda = 0.001;

    ml::LinearSVM model(options);
    model.fit(X, y);

    const ml::Vector predictions =
        model.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error(
            "expected one prediction per sample"
        );
    }
}

void test_linear_svm_fits_separable_binary_data() {
    const ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y = make_linear_svm_test_y();

    ml::LinearSVMOptions options;
    options.learning_rate = 0.01;
    options.max_epochs = 200;
    options.l2_lambda = 0.001;

    ml::LinearSVM model(options);
    model.fit(X, y);

    const ml::Vector predictions =
        model.predict(X);

    test::assert_vector_almost_equal(
        predictions,
        y,
        "test_linear_svm_fits_separable_binary_data"
    );
}

void test_linear_svm_decision_scores_have_expected_signs() {
    const ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y = make_linear_svm_test_y();

    ml::LinearSVMOptions options;
    options.learning_rate = 0.01;
    options.max_epochs = 200;
    options.l2_lambda = 0.001;

    ml::LinearSVM model(options);
    model.fit(X, y);

    const ml::Vector scores =
        model.decision_function(X);

    for (Eigen::Index i = 0; i < scores.size(); ++i) {
        if (y(i) == 0.0 && !(scores(i) < 0.0)) {
            throw std::runtime_error(
                "expected class 0 samples to have negative decision scores"
            );
        }

        if (y(i) == 1.0 && !(scores(i) >= 0.0)) {
            throw std::runtime_error(
                "expected class 1 samples to have non-negative decision scores"
            );
        }
    }
}

void test_linear_svm_stores_training_loss_history() {
    const ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y = make_linear_svm_test_y();

    ml::LinearSVMOptions options;
    options.learning_rate = 0.01;
    options.max_epochs = 50;
    options.l2_lambda = 0.001;

    ml::LinearSVM model(options);
    model.fit(X, y);

    if (model.training_loss_history().size() != options.max_epochs) {
        throw std::runtime_error(
            "expected one loss value per epoch"
        );
    }

    for (double loss : model.training_loss_history()) {
        if (!std::isfinite(loss) || loss < 0.0) {
            throw std::runtime_error(
                "expected finite non-negative training losses"
            );
        }
    }
}

void test_linear_svm_training_loss_decreases() {
    const ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y = make_linear_svm_test_y();

    ml::LinearSVMOptions options;
    options.learning_rate = 0.01;
    options.max_epochs = 100;
    options.l2_lambda = 0.001;

    ml::LinearSVM model(options);
    model.fit(X, y);

    const std::vector<double>& history =
        model.training_loss_history();

    if (history.empty()) {
        throw std::runtime_error(
            "expected non-empty training loss history"
        );
    }

    if (!(history.back() < history.front())) {
        throw std::runtime_error(
            "expected final loss to be lower than initial loss"
        );
    }
}

void test_linear_svm_rejects_predict_before_fit() {
    const ml::Matrix X = make_linear_svm_test_X();

    const ml::LinearSVM model;

    static_cast<void>(model.predict(X));
}

void test_linear_svm_rejects_decision_function_before_fit() {
    const ml::Matrix X = make_linear_svm_test_X();

    const ml::LinearSVM model;

    static_cast<void>(model.decision_function(X));
}

void test_linear_svm_rejects_empty_fit_X() {
    const ml::Matrix X;
    const ml::Vector y = make_linear_svm_test_y();

    ml::LinearSVM model;
    model.fit(X, y);
}

void test_linear_svm_rejects_empty_fit_y() {
    const ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y;

    ml::LinearSVM model;
    model.fit(X, y);
}

void test_linear_svm_rejects_mismatched_fit_data() {
    ml::Matrix X(3, 2);
    X.setZero();

    const ml::Vector y = make_vector({
        0.0,
        1.0
    });

    ml::LinearSVM model;
    model.fit(X, y);
}

void test_linear_svm_rejects_non_binary_targets() {
    const ml::Matrix X = make_linear_svm_test_X();
    ml::Vector y = make_linear_svm_test_y();
    y(0) = 2.0;

    ml::LinearSVM model;
    model.fit(X, y);
}

void test_linear_svm_rejects_non_integer_targets() {
    const ml::Matrix X = make_linear_svm_test_X();
    ml::Vector y = make_linear_svm_test_y();
    y(0) = 0.5;

    ml::LinearSVM model;
    model.fit(X, y);
}

void test_linear_svm_rejects_predict_feature_mismatch() {
    const ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y = make_linear_svm_test_y();

    ml::LinearSVM model;
    model.fit(X, y);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(model.predict(X_bad));
}

void test_linear_svm_rejects_non_finite_fit_values() {
    ml::Matrix X = make_linear_svm_test_X();
    const ml::Vector y = make_linear_svm_test_y();

    X(0, 0) = std::numeric_limits<double>::quiet_NaN();

    ml::LinearSVM model;
    model.fit(X, y);
}

void test_linear_svm_weights_reject_before_fit() {
    const ml::LinearSVM model;

    static_cast<void>(model.weights());
}

void test_linear_svm_bias_reject_before_fit() {
    const ml::LinearSVM model;

    static_cast<void>(model.bias());
}

// -----------------------------------------------------------------------------
// Phase 7.9 LinearSVM comparison workflow helpers
// -----------------------------------------------------------------------------

struct LinearSVMComparisonResult {
    std::string experiment_name;
    std::string variant_name;
    std::string model_type;

    std::string distance_metric;
    std::size_t k{0};

    double learning_rate{0.0};
    std::size_t max_epochs{0};
    double l2_lambda{0.0};

    double accuracy{0.0};
    std::size_t num_predictions{0};
};

struct LinearSVMMarginBehaviorRow {
    std::string sample_name;
    double x0{0.0};
    double x1{0.0};
    double y_true{0.0};
    double svm_target{0.0};
    double decision_score{0.0};
    double signed_margin{0.0};
    double prediction{0.0};
    std::string margin_status;
};

ml::Matrix make_linear_svm_workflow_X() {
    ml::Matrix X(10, 2);

    X << -3.0, -2.0,
         -2.5, -1.8,
         -2.0, -2.2,
         -1.5, -1.0,
         -1.0, -1.4,

          1.0,  1.4,
          1.5,  1.0,
          2.0,  2.2,
          2.5,  1.8,
          3.0,  2.0;

    return X;
}

ml::Vector make_linear_svm_workflow_y() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,

        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    });
}

ml::Matrix make_linear_svm_workflow_X_eval() {
    ml::Matrix X(6, 2);

    X << -2.75, -2.0,
         -1.25, -1.2,
         -0.25, -0.2,

          0.25,  0.2,
          1.25,  1.2,
          2.75,  2.0;

    return X;
}

ml::Vector make_linear_svm_workflow_y_eval() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    });
}

double binary_to_svm_target_for_export(
    double label
) {
    return label == 1.0 ? 1.0 : -1.0;
}

std::string margin_status_from_signed_margin(
    double signed_margin
) {
    if (signed_margin >= 1.0) {
        return "correct_with_margin";
    }

    if (signed_margin > 0.0) {
        return "correct_inside_margin";
    }

    return "misclassified_or_on_wrong_side";
}

LinearSVMComparisonResult run_knn_workflow_comparison_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::KNNClassifierOptions& options
) {
    ml::KNNClassifier model(options);
    model.fit(X_train, y_train);

    const ml::Vector predictions =
        model.predict(X_eval);

    return LinearSVMComparisonResult{
        experiment_name,
        variant_name,
        "KNNClassifier",
        ml::distance_metric_name(options.distance_metric),
        options.k,
        0.0,
        0,
        0.0,
        ml::accuracy_score(predictions, y_eval),
        static_cast<std::size_t>(predictions.size())
    };
}

LinearSVMComparisonResult run_linear_svm_workflow_comparison_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::LinearSVMOptions& options
) {
    ml::LinearSVM model(options);
    model.fit(X_train, y_train);

    const ml::Vector predictions =
        model.predict(X_eval);

    return LinearSVMComparisonResult{
        experiment_name,
        variant_name,
        "LinearSVM",
        "not_applicable",
        0,
        options.learning_rate,
        options.max_epochs,
        options.l2_lambda,
        ml::accuracy_score(predictions, y_eval),
        static_cast<std::size_t>(predictions.size())
    };
}

std::vector<LinearSVMMarginBehaviorRow> make_linear_svm_margin_behavior_rows(
    const ml::Matrix& X,
    const ml::Vector& y,
    const ml::LinearSVM& model
) {
    const ml::Vector scores =
        model.decision_function(X);

    const ml::Vector predictions =
        model.predict(X);

    std::vector<LinearSVMMarginBehaviorRow> rows;
    rows.reserve(static_cast<std::size_t>(X.rows()));

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const double svm_target =
            binary_to_svm_target_for_export(y(i));

        const double signed_margin =
            svm_target * scores(i);

        rows.push_back(
            LinearSVMMarginBehaviorRow{
                "sample_" + std::to_string(i),
                X(i, 0),
                X(i, 1),
                y(i),
                svm_target,
                scores(i),
                signed_margin,
                predictions(i),
                margin_status_from_signed_margin(signed_margin)
            }
        );
    }

    return rows;
}

void export_linear_svm_comparison_csv(
    const std::vector<LinearSVMComparisonResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_linear_svm_comparison_csv: failed to open output file"
        );
    }

    file << "experiment_name,variant_name,model_type,distance_metric,k,"
         << "learning_rate,max_epochs,l2_lambda,accuracy,num_predictions\n";

    for (const LinearSVMComparisonResult& result : results) {
        file << result.experiment_name << ","
             << result.variant_name << ","
             << result.model_type << ","
             << result.distance_metric << ","
             << result.k << ","
             << result.learning_rate << ","
             << result.max_epochs << ","
             << result.l2_lambda << ","
             << result.accuracy << ","
             << result.num_predictions << "\n";
    }
}

void export_linear_svm_comparison_txt(
    const std::vector<LinearSVMComparisonResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_linear_svm_comparison_txt: failed to open output file"
        );
    }

    file << "LinearSVM Phase 7 Comparison Workflow\n\n";

    file << "This experiment compares:\n"
         << "- k-NN vs LinearSVM on a simple separable dataset\n"
         << "- LinearSVM decision-score behavior through a separate margin export\n\n";

    file << "LogisticRegression comparison note:\n"
         << "- LogisticRegression vs LinearSVM is useful conceptually.\n"
         << "- It is kept optional here to avoid coupling the Phase 7 distance/kernel workflow too strongly to Phase 4 internals.\n"
         << "- The main Phase 7 comparison is k-NN vs LinearSVM because both are geometric classifiers.\n\n";

    for (const LinearSVMComparisonResult& result : results) {
        file << "Experiment: " << result.experiment_name << "\n"
             << "Variant: " << result.variant_name << "\n"
             << "Model: " << result.model_type << "\n"
             << "Distance metric: " << result.distance_metric << "\n"
             << "k: " << result.k << "\n"
             << "learning_rate: " << result.learning_rate << "\n"
             << "max_epochs: " << result.max_epochs << "\n"
             << "l2_lambda: " << result.l2_lambda << "\n"
             << "accuracy: " << result.accuracy << "\n"
             << "num_predictions: " << result.num_predictions << "\n\n";
    }
}

void export_linear_svm_margin_behavior_csv(
    const std::vector<LinearSVMMarginBehaviorRow>& rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_linear_svm_margin_behavior_csv: failed to open output file"
        );
    }

    file << "sample_name,x0,x1,y_true,svm_target,decision_score,"
         << "signed_margin,prediction,margin_status\n";

    for (const LinearSVMMarginBehaviorRow& row : rows) {
        file << row.sample_name << ","
             << row.x0 << ","
             << row.x1 << ","
             << row.y_true << ","
             << row.svm_target << ","
             << row.decision_score << ","
             << row.signed_margin << ","
             << row.prediction << ","
             << row.margin_status << "\n";
    }
}

// -----------------------------------------------------------------------------
// Phase 7.9 LinearSVM comparison workflow tests
// -----------------------------------------------------------------------------

void test_experiment_exports_linear_svm_comparison_workflow() {
    ensure_phase7_output_dir_exists();

    const ml::Matrix X_train =
        make_linear_svm_workflow_X();

    const ml::Vector y_train =
        make_linear_svm_workflow_y();

    const ml::Matrix X_eval =
        make_linear_svm_workflow_X_eval();

    const ml::Vector y_eval =
        make_linear_svm_workflow_y_eval();

    std::vector<LinearSVMComparisonResult> results;

    {
        ml::KNNClassifierOptions options;
        options.k = 1;
        options.distance_metric = ml::DistanceMetric::Euclidean;

        results.push_back(
            run_knn_workflow_comparison_experiment(
                "knn_vs_linear_svm",
                "knn_k_1_euclidean",
                X_train,
                y_train,
                X_eval,
                y_eval,
                options
            )
        );
    }

    {
        ml::KNNClassifierOptions options;
        options.k = 3;
        options.distance_metric = ml::DistanceMetric::Euclidean;

        results.push_back(
            run_knn_workflow_comparison_experiment(
                "knn_vs_linear_svm",
                "knn_k_3_euclidean",
                X_train,
                y_train,
                X_eval,
                y_eval,
                options
            )
        );
    }

    {
        ml::LinearSVMOptions options;
        options.learning_rate = 0.01;
        options.max_epochs = 200;
        options.l2_lambda = 0.001;

        results.push_back(
            run_linear_svm_workflow_comparison_experiment(
                "knn_vs_linear_svm",
                "linear_svm_lr_0_01_lambda_0_001",
                X_train,
                y_train,
                X_eval,
                y_eval,
                options
            )
        );
    }

    ml::LinearSVMOptions margin_options;
    margin_options.learning_rate = 0.01;
    margin_options.max_epochs = 200;
    margin_options.l2_lambda = 0.001;

    ml::LinearSVM margin_model(margin_options);
    margin_model.fit(X_train, y_train);

    const std::vector<LinearSVMMarginBehaviorRow> margin_rows =
        make_linear_svm_margin_behavior_rows(
            X_eval,
            y_eval,
            margin_model
        );

    const std::string comparison_csv_path =
        k_phase7_output_dir + "/linear_svm_comparison.csv";

    const std::string comparison_txt_path =
        k_phase7_output_dir + "/linear_svm_comparison.txt";

    const std::string margin_csv_path =
        k_phase7_output_dir + "/linear_svm_margin_behavior.csv";

    export_linear_svm_comparison_csv(
        results,
        comparison_csv_path
    );

    export_linear_svm_comparison_txt(
        results,
        comparison_txt_path
    );

    export_linear_svm_margin_behavior_csv(
        margin_rows,
        margin_csv_path
    );

    if (!std::filesystem::exists(comparison_csv_path)) {
        throw std::runtime_error(
            "expected linear_svm_comparison.csv to exist"
        );
    }

    if (!std::filesystem::exists(comparison_txt_path)) {
        throw std::runtime_error(
            "expected linear_svm_comparison.txt to exist"
        );
    }

    if (!std::filesystem::exists(margin_csv_path)) {
        throw std::runtime_error(
            "expected linear_svm_margin_behavior.csv to exist"
        );
    }

    if (results.empty()) {
        throw std::runtime_error(
            "expected non-empty LinearSVM comparison results"
        );
    }

    if (margin_rows.empty()) {
        throw std::runtime_error(
            "expected non-empty LinearSVM margin behavior rows"
        );
    }

    for (const LinearSVMMarginBehaviorRow& row : margin_rows) {
        if (!std::isfinite(row.decision_score)) {
            throw std::runtime_error(
                "expected finite LinearSVM decision score"
            );
        }

        if (!std::isfinite(row.signed_margin)) {
            throw std::runtime_error(
                "expected finite LinearSVM signed margin"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Phase 7.10 LinearSVM hyperparameter experiment helpers
// -----------------------------------------------------------------------------

struct LinearSVMHyperparameterResult {
    std::string experiment_name;
    std::string variant_name;

    double learning_rate{0.0};
    std::size_t max_epochs{0};
    double l2_lambda{0.0};

    double accuracy{0.0};
    double final_training_loss{0.0};
    double weight_norm{0.0};
    double bias{0.0};
    double mean_signed_margin{0.0};
    std::size_t num_predictions{0};
};

double mean_signed_margin_for_model(
    const ml::Matrix& X,
    const ml::Vector& y,
    const ml::LinearSVM& model
) {
    const ml::Vector scores =
        model.decision_function(X);

    double total = 0.0;

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const double svm_target =
            binary_to_svm_target_for_export(y(i));

        total += svm_target * scores(i);
    }

    return total / static_cast<double>(X.rows());
}

LinearSVMHyperparameterResult run_linear_svm_hyperparameter_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_eval,
    const ml::Vector& y_eval,
    const ml::LinearSVMOptions& options
) {
    ml::LinearSVM model(options);
    model.fit(X_train, y_train);

    const ml::Vector predictions =
        model.predict(X_eval);

    const std::vector<double>& loss_history =
        model.training_loss_history();

    if (loss_history.empty()) {
        throw std::runtime_error(
            "run_linear_svm_hyperparameter_experiment: expected non-empty loss history"
        );
    }

    return LinearSVMHyperparameterResult{
        experiment_name,
        variant_name,
        options.learning_rate,
        options.max_epochs,
        options.l2_lambda,
        ml::accuracy_score(predictions, y_eval),
        loss_history.back(),
        model.weights().norm(),
        model.bias(),
        mean_signed_margin_for_model(
            X_eval,
            y_eval,
            model
        ),
        static_cast<std::size_t>(predictions.size())
    };
}

void export_linear_svm_hyperparameter_results_csv(
    const std::vector<LinearSVMHyperparameterResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_linear_svm_hyperparameter_results_csv: failed to open output file"
        );
    }

    file << "experiment_name,variant_name,learning_rate,max_epochs,l2_lambda,"
         << "accuracy,final_training_loss,weight_norm,bias,mean_signed_margin,"
         << "num_predictions\n";

    for (const LinearSVMHyperparameterResult& result : results) {
        file << result.experiment_name << ","
             << result.variant_name << ","
             << result.learning_rate << ","
             << result.max_epochs << ","
             << result.l2_lambda << ","
             << result.accuracy << ","
             << result.final_training_loss << ","
             << result.weight_norm << ","
             << result.bias << ","
             << result.mean_signed_margin << ","
             << result.num_predictions << "\n";
    }
}

void export_linear_svm_hyperparameter_results_txt(
    const std::vector<LinearSVMHyperparameterResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_linear_svm_hyperparameter_results_txt: failed to open output file"
        );
    }

    file << "LinearSVM Hyperparameter and Margin Comparison\n\n";

    file << "This experiment compares:\n"
         << "- different L2 regularization strengths\n"
         << "- different learning rates\n"
         << "- margin behavior through mean signed margin\n"
         << "- k-NN vs LinearSVM decision behavior through the Phase 7 comparison workflow\n\n";

    file << "Notes:\n"
         << "- Higher mean signed margin means samples are, on average, farther on the correct side of the boundary.\n"
         << "- Larger L2 regularization usually reduces weight norm.\n"
         << "- Learning rate affects optimization behavior and final loss.\n\n";

    for (const LinearSVMHyperparameterResult& result : results) {
        file << "Experiment: " << result.experiment_name << "\n"
             << "Variant: " << result.variant_name << "\n"
             << "learning_rate: " << result.learning_rate << "\n"
             << "max_epochs: " << result.max_epochs << "\n"
             << "l2_lambda: " << result.l2_lambda << "\n"
             << "accuracy: " << result.accuracy << "\n"
             << "final_training_loss: " << result.final_training_loss << "\n"
             << "weight_norm: " << result.weight_norm << "\n"
             << "bias: " << result.bias << "\n"
             << "mean_signed_margin: " << result.mean_signed_margin << "\n"
             << "num_predictions: " << result.num_predictions << "\n\n";
    }
}

// -----------------------------------------------------------------------------
// Phase 7.10 LinearSVM hyperparameter experiment tests
// -----------------------------------------------------------------------------

void test_experiment_exports_linear_svm_hyperparameter_comparison() {
    ensure_phase7_output_dir_exists();

    const ml::Matrix X_train =
        make_linear_svm_workflow_X();

    const ml::Vector y_train =
        make_linear_svm_workflow_y();

    const ml::Matrix X_eval =
        make_linear_svm_workflow_X_eval();

    const ml::Vector y_eval =
        make_linear_svm_workflow_y_eval();

    std::vector<LinearSVMHyperparameterResult> results;

    for (double l2_lambda : {0.0, 0.001, 0.1}) {
        ml::LinearSVMOptions options;
        options.learning_rate = 0.01;
        options.max_epochs = 200;
        options.l2_lambda = l2_lambda;

        results.push_back(
            run_linear_svm_hyperparameter_experiment(
                "regularization_strength",
                "l2_lambda_" + std::to_string(l2_lambda),
                X_train,
                y_train,
                X_eval,
                y_eval,
                options
            )
        );
    }

    for (double learning_rate : {0.001, 0.01, 0.05}) {
        ml::LinearSVMOptions options;
        options.learning_rate = learning_rate;
        options.max_epochs = 200;
        options.l2_lambda = 0.001;

        results.push_back(
            run_linear_svm_hyperparameter_experiment(
                "learning_rate",
                "learning_rate_" + std::to_string(learning_rate),
                X_train,
                y_train,
                X_eval,
                y_eval,
                options
            )
        );
    }

    {
        ml::LinearSVMOptions options;
        options.learning_rate = 0.01;
        options.max_epochs = 200;
        options.l2_lambda = 0.001;

        results.push_back(
            run_linear_svm_hyperparameter_experiment(
                "margin_behavior",
                "baseline_margin_export_reference",
                X_train,
                y_train,
                X_eval,
                y_eval,
                options
            )
        );
    }

    const std::string comparison_csv_path =
        k_phase7_output_dir + "/linear_svm_hyperparameter_comparison.csv";

    const std::string comparison_txt_path =
        k_phase7_output_dir + "/linear_svm_hyperparameter_comparison.txt";

    export_linear_svm_hyperparameter_results_csv(
        results,
        comparison_csv_path
    );

    export_linear_svm_hyperparameter_results_txt(
        results,
        comparison_txt_path
    );

    if (!std::filesystem::exists(comparison_csv_path)) {
        throw std::runtime_error(
            "expected linear_svm_hyperparameter_comparison.csv to exist"
        );
    }

    if (!std::filesystem::exists(comparison_txt_path)) {
        throw std::runtime_error(
            "expected linear_svm_hyperparameter_comparison.txt to exist"
        );
    }

    if (results.empty()) {
        throw std::runtime_error(
            "expected non-empty LinearSVM hyperparameter results"
        );
    }

    for (const LinearSVMHyperparameterResult& result : results) {
        if (!std::isfinite(result.accuracy)) {
            throw std::runtime_error(
                "expected finite LinearSVM accuracy"
            );
        }

        if (!std::isfinite(result.final_training_loss)) {
            throw std::runtime_error(
                "expected finite LinearSVM final training loss"
            );
        }

        if (!std::isfinite(result.weight_norm)) {
            throw std::runtime_error(
                "expected finite LinearSVM weight norm"
            );
        }

        if (!std::isfinite(result.mean_signed_margin)) {
            throw std::runtime_error(
                "expected finite LinearSVM mean signed margin"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Test runners
// -----------------------------------------------------------------------------

void run_distance_metric_tests() {
    std::cout << "\n[Phase 7.1] Distance metric tests\n\n";

    test::expect_no_throw(
        "squared_euclidean_distance computes expected value",
        test_squared_euclidean_distance_computes_expected_value
    );

    test::expect_no_throw(
        "euclidean_distance computes expected value",
        test_euclidean_distance_computes_expected_value
    );

    test::expect_no_throw(
        "manhattan_distance computes expected value",
        test_manhattan_distance_computes_expected_value
    );

    test::expect_no_throw(
        "distance metrics return zero for identical vectors",
        test_distance_metrics_return_zero_for_identical_vectors
    );

    test::expect_no_throw(
        "distance metrics are symmetric",
        test_distance_metrics_are_symmetric
    );

    test::expect_no_throw(
        "squared Euclidean preserves Euclidean ordering",
        test_squared_euclidean_preserves_euclidean_ordering
    );

    test::expect_invalid_argument(
        "distance metrics reject empty vectors",
        test_distance_metrics_reject_empty_vectors
    );

    test::expect_invalid_argument(
        "distance metrics reject mismatched vectors",
        test_distance_metrics_reject_mismatched_vectors
    );

    test::expect_invalid_argument(
        "distance metrics reject non-finite values",
        test_distance_metrics_reject_non_finite_values
    );
}

void run_knn_classifier_tests() {
    std::cout << "\n[Phase 7.2] Multivariate KNNClassifier tests\n\n";

    test::expect_no_throw(
        "KNNClassifierOptions accepts defaults",
        test_knn_options_accept_defaults
    );

    test::expect_invalid_argument(
        "KNNClassifierOptions rejects zero k",
        test_knn_options_reject_zero_k
    );

    test::expect_no_throw(
        "distance_metric_name returns expected values",
        test_distance_metric_name_returns_expected_values
    );

    test::expect_no_throw(
        "KNNClassifier reports not fitted initially",
        test_knn_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "KNNClassifier fit marks model as fitted",
        test_knn_fit_marks_model_as_fitted
    );

    test::expect_no_throw(
        "KNNClassifier predict returns expected shape",
        test_knn_predict_returns_expected_shape
    );

    test::expect_no_throw(
        "KNNClassifier predicts simple clusters with Euclidean",
        test_knn_predicts_simple_clusters_with_euclidean
    );

    test::expect_no_throw(
        "KNNClassifier predicts simple clusters with Manhattan",
        test_knn_predicts_simple_clusters_with_manhattan
    );

    test::expect_no_throw(
        "KNNClassifier predicts simple clusters with squared Euclidean",
        test_knn_predicts_simple_clusters_with_squared_euclidean
    );

    test::expect_no_throw(
        "KNNClassifier uses smallest-label tie break",
        test_knn_uses_smallest_label_tie_break
    );

    test::expect_invalid_argument(
        "KNNClassifier rejects predict before fit",
        test_knn_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "KNNClassifier rejects empty fit X",
        test_knn_rejects_empty_fit_X
    );

    test::expect_invalid_argument(
        "KNNClassifier rejects empty fit y",
        test_knn_rejects_empty_fit_y
    );

    test::expect_invalid_argument(
        "KNNClassifier rejects mismatched fit data",
        test_knn_rejects_mismatched_fit_data
    );

    test::expect_invalid_argument(
        "KNNClassifier rejects invalid class labels",
        test_knn_rejects_invalid_class_labels
    );

    test::expect_invalid_argument(
        "KNNClassifier rejects k larger than training size",
        test_knn_rejects_k_larger_than_training_size
    );

    test::expect_invalid_argument(
        "KNNClassifier rejects predict feature mismatch",
        test_knn_rejects_predict_feature_mismatch
    );

    test::expect_invalid_argument(
        "KNNClassifier rejects non-finite predict values",
        test_knn_rejects_non_finite_predict_values
    );
}

void run_knn_experiment_export_tests() {
    std::cout << "\n[Phase 7.3] KNN experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports KNN metric comparison",
        test_experiment_exports_knn_metric_comparison
    );
}

void run_kernel_function_tests() {
    std::cout << "\n[Phase 7.4] Reusable kernel function tests\n\n";

    test::expect_no_throw(
        "linear_kernel computes dot product",
        test_linear_kernel_computes_dot_product
    );

    test::expect_no_throw(
        "polynomial_kernel computes expected value",
        test_polynomial_kernel_computes_expected_value
    );

    test::expect_no_throw(
        "rbf_kernel returns one for identical vectors",
        test_rbf_kernel_returns_one_for_identical_vectors
    );

    test::expect_no_throw(
        "rbf_kernel decreases with distance",
        test_rbf_kernel_decreases_with_distance
    );

    test::expect_no_throw(
        "kernel functions are symmetric",
        test_kernel_functions_are_symmetric
    );

    test::expect_no_throw(
        "polynomial degree one matches shifted linear kernel",
        test_polynomial_kernel_degree_one_matches_shifted_linear_kernel
    );

    test::expect_invalid_argument(
        "kernel functions reject empty vectors",
        test_kernel_functions_reject_empty_vectors
    );

    test::expect_invalid_argument(
        "kernel functions reject mismatched vectors",
        test_kernel_functions_reject_mismatched_vectors
    );

    test::expect_invalid_argument(
        "kernel functions reject non-finite values",
        test_kernel_functions_reject_non_finite_values
    );

    test::expect_invalid_argument(
        "polynomial_kernel rejects invalid degree",
        test_polynomial_kernel_rejects_invalid_degree
    );

    test::expect_invalid_argument(
        "polynomial_kernel rejects non-finite coef0",
        test_polynomial_kernel_rejects_non_finite_coef0
    );

    test::expect_invalid_argument(
        "rbf_kernel rejects invalid gamma",
        test_rbf_kernel_rejects_invalid_gamma
    );
}

void run_kernel_similarity_demo_tests() {
    std::cout << "\n[Phase 7.5] Kernel similarity demo export tests\n\n";

    test::expect_no_throw(
        "Experiment exports kernel similarity demo",
        test_experiment_exports_kernel_similarity_demo
    );
}

void run_svm_margin_intuition_demo_tests() {
    std::cout << "\n[Phase 7.6] SVM margin intuition demo export tests\n\n";

    test::expect_no_throw(
        "Experiment exports SVM margin intuition demo",
        test_experiment_exports_svm_margin_intuition_demo
    );
}

void run_linear_svm_options_tests() {
    std::cout << "\n[Phase 7.7] LinearSVMOptions tests\n\n";

    test::expect_no_throw(
        "LinearSVMOptions accepts defaults",
        test_linear_svm_options_accept_defaults
    );

    test::expect_no_throw(
        "LinearSVMOptions accepts valid values",
        test_linear_svm_options_accept_valid_values
    );

    test::expect_invalid_argument(
        "LinearSVMOptions rejects zero learning rate",
        test_linear_svm_options_reject_zero_learning_rate
    );

    test::expect_invalid_argument(
        "LinearSVMOptions rejects negative learning rate",
        test_linear_svm_options_reject_negative_learning_rate
    );

    test::expect_invalid_argument(
        "LinearSVMOptions rejects non-finite learning rate",
        test_linear_svm_options_reject_non_finite_learning_rate
    );

    test::expect_invalid_argument(
        "LinearSVMOptions rejects zero max_epochs",
        test_linear_svm_options_reject_zero_max_epochs
    );

    test::expect_no_throw(
        "LinearSVMOptions accepts zero l2_lambda",
        test_linear_svm_options_accept_zero_l2_lambda
    );

    test::expect_invalid_argument(
        "LinearSVMOptions rejects negative l2_lambda",
        test_linear_svm_options_reject_negative_l2_lambda
    );

    test::expect_invalid_argument(
        "LinearSVMOptions rejects non-finite l2_lambda",
        test_linear_svm_options_reject_non_finite_l2_lambda
    );
}

void run_linear_svm_model_tests() {
    std::cout << "\n[Phase 7.8] LinearSVM model tests\n\n";

    test::expect_no_throw(
        "LinearSVM reports not fitted initially",
        test_linear_svm_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "LinearSVM fit marks model as fitted",
        test_linear_svm_fit_marks_model_as_fitted
    );

    test::expect_no_throw(
        "LinearSVM decision_function returns expected shape",
        test_linear_svm_decision_function_returns_expected_shape
    );

    test::expect_no_throw(
        "LinearSVM predict returns expected shape",
        test_linear_svm_predict_returns_expected_shape
    );

    test::expect_no_throw(
        "LinearSVM fits separable binary data",
        test_linear_svm_fits_separable_binary_data
    );

    test::expect_no_throw(
        "LinearSVM decision scores have expected signs",
        test_linear_svm_decision_scores_have_expected_signs
    );

    test::expect_no_throw(
        "LinearSVM stores training loss history",
        test_linear_svm_stores_training_loss_history
    );

    test::expect_no_throw(
        "LinearSVM training loss decreases",
        test_linear_svm_training_loss_decreases
    );

    test::expect_invalid_argument(
        "LinearSVM rejects predict before fit",
        test_linear_svm_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "LinearSVM rejects decision_function before fit",
        test_linear_svm_rejects_decision_function_before_fit
    );

    test::expect_invalid_argument(
        "LinearSVM rejects empty fit X",
        test_linear_svm_rejects_empty_fit_X
    );

    test::expect_invalid_argument(
        "LinearSVM rejects empty fit y",
        test_linear_svm_rejects_empty_fit_y
    );

    test::expect_invalid_argument(
        "LinearSVM rejects mismatched fit data",
        test_linear_svm_rejects_mismatched_fit_data
    );

    test::expect_invalid_argument(
        "LinearSVM rejects non-binary targets",
        test_linear_svm_rejects_non_binary_targets
    );

    test::expect_invalid_argument(
        "LinearSVM rejects non-integer targets",
        test_linear_svm_rejects_non_integer_targets
    );

    test::expect_invalid_argument(
        "LinearSVM rejects predict feature mismatch",
        test_linear_svm_rejects_predict_feature_mismatch
    );

    test::expect_invalid_argument(
        "LinearSVM rejects non-finite fit values",
        test_linear_svm_rejects_non_finite_fit_values
    );

    test::expect_invalid_argument(
        "LinearSVM weights reject before fit",
        test_linear_svm_weights_reject_before_fit
    );

    test::expect_invalid_argument(
        "LinearSVM bias reject before fit",
        test_linear_svm_bias_reject_before_fit
    );
}

void run_linear_svm_comparison_workflow_tests() {
    std::cout << "\n[Phase 7.9] LinearSVM comparison workflow tests\n\n";

    test::expect_no_throw(
        "Experiment exports LinearSVM comparison workflow",
        test_experiment_exports_linear_svm_comparison_workflow
    );
}

void run_linear_svm_hyperparameter_experiment_tests() {
    std::cout << "\n[Phase 7.10] LinearSVM hyperparameter experiment tests\n\n";

    test::expect_no_throw(
        "Experiment exports LinearSVM hyperparameter comparison",
        test_experiment_exports_linear_svm_hyperparameter_comparison
    );
}

}  // namespace

namespace ml::experiments {

void run_phase7_distance_kernel_sanity() {
    run_distance_metric_tests();
    run_knn_classifier_tests();
    run_knn_experiment_export_tests();
    run_kernel_function_tests();
    run_kernel_similarity_demo_tests();
    run_svm_margin_intuition_demo_tests();
    run_linear_svm_options_tests();
    run_linear_svm_model_tests();
    run_linear_svm_comparison_workflow_tests();
    run_linear_svm_hyperparameter_experiment_tests();
}

}  // namespace ml::experiments