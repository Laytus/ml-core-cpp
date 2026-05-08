#include "phase8_unsupervised_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/types.hpp"
#include "ml/unsupervised/kmeans.hpp"
#include "ml/unsupervised/pca.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
// Phase 8.1 KMeans tests
// -----------------------------------------------------------------------------

ml::Matrix make_kmeans_test_X() {
    ml::Matrix X(6, 2);

    X << 0.0, 0.0,
         0.2, 0.1,
         0.1, 0.2,
         5.0, 5.0,
         5.2, 5.1,
         5.1, 5.2;

    return X;
}

void test_kmeans_options_accept_defaults() {
    const ml::KMeansOptions options;

    ml::validate_kmeans_options(
        options,
        "test_kmeans_options_accept_defaults"
    );
}

void test_kmeans_options_reject_zero_clusters() {
    ml::KMeansOptions options;
    options.num_clusters = 0;

    ml::validate_kmeans_options(
        options,
        "test_kmeans_options_reject_zero_clusters"
    );
}

void test_kmeans_options_reject_zero_max_iterations() {
    ml::KMeansOptions options;
    options.max_iterations = 0;

    ml::validate_kmeans_options(
        options,
        "test_kmeans_options_reject_zero_max_iterations"
    );
}

void test_kmeans_options_reject_negative_tolerance() {
    ml::KMeansOptions options;
    options.tolerance = -1.0;

    ml::validate_kmeans_options(
        options,
        "test_kmeans_options_reject_negative_tolerance"
    );
}

void test_kmeans_options_reject_non_finite_tolerance() {
    ml::KMeansOptions options;
    options.tolerance = std::numeric_limits<double>::infinity();

    ml::validate_kmeans_options(
        options,
        "test_kmeans_options_reject_non_finite_tolerance"
    );
}

void test_kmeans_reports_not_fitted_initially() {
    const ml::KMeans model;

    if (model.is_fitted()) {
        throw std::runtime_error(
            "expected KMeans to start unfitted"
        );
    }
}

void test_kmeans_fit_marks_model_as_fitted() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;
    options.max_iterations = 20;

    ml::KMeans model(options);
    model.fit(X);

    if (!model.is_fitted()) {
        throw std::runtime_error(
            "expected KMeans to be fitted"
        );
    }
}

void test_kmeans_centroids_have_expected_shape() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;
    options.max_iterations = 20;

    ml::KMeans model(options);
    model.fit(X);

    const ml::Matrix& centroids =
        model.centroids();

    if (centroids.rows() != 2 || centroids.cols() != X.cols()) {
        throw std::runtime_error(
            "expected centroids shape to be num_clusters x num_features"
        );
    }
}

void test_kmeans_labels_have_expected_shape() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;
    options.max_iterations = 20;

    ml::KMeans model(options);
    model.fit(X);

    const ml::Vector& labels =
        model.labels();

    if (labels.size() != X.rows()) {
        throw std::runtime_error(
            "expected one cluster label per sample"
        );
    }
}

void test_kmeans_fit_predict_matches_labels() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;
    options.max_iterations = 20;

    ml::KMeans model(options);

    const ml::Vector labels =
        model.fit_predict(X);

    test::assert_vector_almost_equal(
        labels,
        model.labels(),
        "test_kmeans_fit_predict_matches_labels"
    );
}

void test_kmeans_predict_returns_expected_shape() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;
    options.max_iterations = 20;

    ml::KMeans model(options);
    model.fit(X);

    const ml::Vector predictions =
        model.predict(X);

    if (predictions.size() != X.rows()) {
        throw std::runtime_error(
            "expected one predicted cluster per sample"
        );
    }
}

void test_kmeans_learns_two_separated_clusters() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;
    options.max_iterations = 50;
    options.tolerance = 1e-9;

    ml::KMeans model(options);
    model.fit(X);

    const ml::Vector predictions =
        model.predict(X);

    const double first_cluster_label =
        predictions(0);

    const double second_cluster_label =
        predictions(3);

    if (first_cluster_label == second_cluster_label) {
        throw std::runtime_error(
            "expected separated groups to receive different cluster labels"
        );
    }

    for (Eigen::Index i = 0; i < 3; ++i) {
        test::assert_almost_equal(
            predictions(i),
            first_cluster_label,
            "expected first natural group to share same cluster"
        );
    }

    for (Eigen::Index i = 3; i < 6; ++i) {
        test::assert_almost_equal(
            predictions(i),
            second_cluster_label,
            "expected second natural group to share same cluster"
        );
    }
}

void test_kmeans_inertia_history_is_recorded() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;
    options.max_iterations = 20;

    ml::KMeans model(options);
    model.fit(X);

    if (model.inertia_history().empty()) {
        throw std::runtime_error(
            "expected non-empty inertia history"
        );
    }

    for (double value : model.inertia_history()) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error(
                "expected finite non-negative inertia values"
            );
        }
    }
}

void test_kmeans_inertia_does_not_increase() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;
    options.max_iterations = 20;
    options.tolerance = 0.0;

    ml::KMeans model(options);
    model.fit(X);

    const std::vector<double>& history =
        model.inertia_history();

    if (history.size() < 2) {
        return;
    }

    for (std::size_t i = 1; i < history.size(); ++i) {
        if (history[i] > history[i - 1] + 1e-9) {
            throw std::runtime_error(
                "expected k-means inertia to be non-increasing"
            );
        }
    }
}

void test_kmeans_rejects_predict_before_fit() {
    const ml::Matrix X = make_kmeans_test_X();

    const ml::KMeans model;

    static_cast<void>(
        model.predict(X)
    );
}

void test_kmeans_rejects_empty_fit_matrix() {
    const ml::Matrix X;

    ml::KMeans model;
    model.fit(X);
}

void test_kmeans_rejects_non_finite_fit_values() {
    ml::Matrix X = make_kmeans_test_X();
    X(0, 0) = std::numeric_limits<double>::quiet_NaN();

    ml::KMeans model;
    model.fit(X);
}

void test_kmeans_rejects_too_many_clusters() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = static_cast<std::size_t>(X.rows()) + 1;

    ml::KMeans model(options);
    model.fit(X);
}

void test_kmeans_rejects_predict_feature_mismatch() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;

    ml::KMeans model(options);
    model.fit(X);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(
        model.predict(X_bad)
    );
}

void test_kmeans_rejects_non_finite_predict_values() {
    const ml::Matrix X = make_kmeans_test_X();

    ml::KMeansOptions options;
    options.num_clusters = 2;

    ml::KMeans model(options);
    model.fit(X);

    ml::Matrix X_bad = X;
    X_bad(0, 0) = std::numeric_limits<double>::quiet_NaN();

    static_cast<void>(
        model.predict(X_bad)
    );
}

void test_kmeans_centroids_reject_before_fit() {
    const ml::KMeans model;

    static_cast<void>(
        model.centroids()
    );
}

void test_kmeans_labels_reject_before_fit() {
    const ml::KMeans model;

    static_cast<void>(
        model.labels()
    );
}

void test_kmeans_inertia_reject_before_fit() {
    const ml::KMeans model;

    static_cast<void>(
        model.inertia()
    );
}

// -----------------------------------------------------------------------------
// Phase 8.2 KMeans experiment export helpers
// -----------------------------------------------------------------------------

const std::string k_phase8_output_dir = "outputs/phase-8-unsupervised";

void ensure_phase8_output_dir_exists() {
    std::filesystem::create_directories(k_phase8_output_dir);
}

struct KMeansExperimentResult {
    std::string experiment_name;
    std::string variant_name;

    std::size_t num_clusters{0};
    std::size_t max_iterations{0};
    double tolerance{0.0};

    double inertia{0.0};
    std::size_t num_iterations{0};
    std::size_t num_samples{0};
    Eigen::Index num_features{0};
};

ml::Matrix make_kmeans_experiment_X() {
    ml::Matrix X(12, 2);

    X << 0.0, 0.0,
         0.2, 0.1,
         0.1, 0.2,
         0.3, 0.2,

         5.0, 5.0,
         5.2, 5.1,
         5.1, 5.2,
         5.3, 5.2,

         9.0, 0.0,
         9.2, 0.1,
         9.1, 0.2,
         9.3, 0.2;

    return X;
}

KMeansExperimentResult run_kmeans_experiment(
    const std::string& experiment_name,
    const std::string& variant_name,
    const ml::Matrix& X,
    const ml::KMeansOptions& options
) {
    ml::KMeans model(options);
    model.fit(X);

    return KMeansExperimentResult{
        experiment_name,
        variant_name,
        options.num_clusters,
        options.max_iterations,
        options.tolerance,
        model.inertia(),
        model.num_iterations(),
        static_cast<std::size_t>(X.rows()),
        X.cols()
    };
}

void export_kmeans_experiment_results_csv(
    const std::vector<KMeansExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_kmeans_experiment_results_csv: failed to open output file"
        );
    }

    file << "experiment_name,variant_name,num_clusters,max_iterations,"
         << "tolerance,inertia,num_iterations,num_samples,num_features\n";

    for (const KMeansExperimentResult& result : results) {
        file << result.experiment_name << ","
             << result.variant_name << ","
             << result.num_clusters << ","
             << result.max_iterations << ","
             << result.tolerance << ","
             << result.inertia << ","
             << result.num_iterations << ","
             << result.num_samples << ","
             << result.num_features << "\n";
    }
}

void export_kmeans_experiment_results_txt(
    const std::vector<KMeansExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_kmeans_experiment_results_txt: failed to open output file"
        );
    }

    file << "KMeans Cluster Comparison\n\n";

    file << "This experiment compares different k values using inertia.\n\n";

    file << "Interpretation:\n";
    file << "- Lower inertia means points are closer to their assigned centroids.\n";
    file << "- Inertia usually decreases as k increases.\n";
    file << "- Lower inertia alone does not mean the chosen k is always better.\n";
    file << "- The elbow method looks for the point where inertia improvement starts slowing down.\n\n";

    for (const KMeansExperimentResult& result : results) {
        file << "Experiment: " << result.experiment_name << "\n"
             << "Variant: " << result.variant_name << "\n"
             << "num_clusters: " << result.num_clusters << "\n"
             << "max_iterations: " << result.max_iterations << "\n"
             << "tolerance: " << result.tolerance << "\n"
             << "inertia: " << result.inertia << "\n"
             << "num_iterations: " << result.num_iterations << "\n"
             << "num_samples: " << result.num_samples << "\n"
             << "num_features: " << result.num_features << "\n\n";
    }
}

// -----------------------------------------------------------------------------
// Phase 8.2 KMeans experiment export tests
// -----------------------------------------------------------------------------

void test_experiment_exports_kmeans_cluster_comparison() {
    ensure_phase8_output_dir_exists();

    const ml::Matrix X =
        make_kmeans_experiment_X();

    std::vector<KMeansExperimentResult> results;

    for (std::size_t k : {1, 2, 3, 4}) {
        ml::KMeansOptions options;
        options.num_clusters = k;
        options.max_iterations = 50;
        options.tolerance = 1e-9;

        results.push_back(
            run_kmeans_experiment(
                "different_k_values_and_inertia",
                "k_" + std::to_string(k),
                X,
                options
            )
        );
    }

    const std::string csv_path =
        k_phase8_output_dir + "/kmeans_cluster_comparison.csv";

    const std::string txt_path =
        k_phase8_output_dir + "/kmeans_cluster_comparison.txt";

    export_kmeans_experiment_results_csv(
        results,
        csv_path
    );

    export_kmeans_experiment_results_txt(
        results,
        txt_path
    );

    if (!std::filesystem::exists(csv_path)) {
        throw std::runtime_error(
            "expected kmeans_cluster_comparison.csv to exist"
        );
    }

    if (!std::filesystem::exists(txt_path)) {
        throw std::runtime_error(
            "expected kmeans_cluster_comparison.txt to exist"
        );
    }

    if (results.size() != 4) {
        throw std::runtime_error(
            "expected one KMeans result per k value"
        );
    }

    for (const KMeansExperimentResult& result : results) {
        if (!std::isfinite(result.inertia) || result.inertia < 0.0) {
            throw std::runtime_error(
                "expected finite non-negative KMeans inertia"
            );
        }

        if (result.num_iterations == 0) {
            throw std::runtime_error(
                "expected KMeans to run at least one iteration"
            );
        }
    }

    for (std::size_t i = 1; i < results.size(); ++i) {
        if (results[i].inertia > results[i - 1].inertia + 1e-9) {
            throw std::runtime_error(
                "expected inertia to not increase as k increases on this dataset"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Phase 8.3 PCA tests
// -----------------------------------------------------------------------------

ml::Matrix make_pca_test_X() {
    ml::Matrix X(5, 2);

    X << 1.0, 2.0,
         2.0, 4.0,
         3.0, 6.0,
         4.0, 8.0,
         5.0, 10.0;

    return X;
}

ml::Matrix make_pca_2d_test_X() {
    ml::Matrix X(6, 2);

    X << -2.0, -1.0,
         -1.0, -0.5,
          0.0,  0.0,
          1.0,  0.5,
          2.0,  1.0,
          3.0,  1.5;

    return X;
}

void test_pca_options_accept_defaults() {
    const ml::PCAOptions options;

    ml::validate_pca_options(
        options,
        "test_pca_options_accept_defaults"
    );
}

void test_pca_options_reject_zero_components() {
    ml::PCAOptions options;
    options.num_components = 0;

    ml::validate_pca_options(
        options,
        "test_pca_options_reject_zero_components"
    );
}

void test_pca_reports_not_fitted_initially() {
    const ml::PCA model;

    if (model.is_fitted()) {
        throw std::runtime_error(
            "expected PCA to start unfitted"
        );
    }
}

void test_pca_fit_marks_model_as_fitted() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    model.fit(X);

    if (!model.is_fitted()) {
        throw std::runtime_error(
            "expected PCA to be fitted"
        );
    }
}

void test_pca_mean_has_expected_shape() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    model.fit(X);

    const ml::Vector& mean =
        model.mean();

    if (mean.size() != X.cols()) {
        throw std::runtime_error(
            "expected PCA mean size to match feature count"
        );
    }

    test::assert_almost_equal(
        mean(0),
        3.0,
        "PCA mean feature 0"
    );

    test::assert_almost_equal(
        mean(1),
        6.0,
        "PCA mean feature 1"
    );
}

void test_pca_components_have_expected_shape() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    model.fit(X);

    const ml::Matrix& components =
        model.components();

    if (components.rows() != X.cols() || components.cols() != 1) {
        throw std::runtime_error(
            "expected PCA components shape to be num_features x num_components"
        );
    }
}

void test_pca_explained_variance_has_expected_shape() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    model.fit(X);

    if (model.explained_variance().size() != 1) {
        throw std::runtime_error(
            "expected explained_variance size to match num_components"
        );
    }

    if (model.explained_variance_ratio().size() != 1) {
        throw std::runtime_error(
            "expected explained_variance_ratio size to match num_components"
        );
    }
}

void test_pca_first_component_explains_most_variance_for_collinear_data() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    model.fit(X);

    const double ratio =
        model.explained_variance_ratio()(0);

    if (ratio < 0.999) {
        throw std::runtime_error(
            "expected first PCA component to explain nearly all variance"
        );
    }
}

void test_pca_transform_returns_expected_shape() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    model.fit(X);

    const ml::Matrix Z =
        model.transform(X);

    if (Z.rows() != X.rows() || Z.cols() != 1) {
        throw std::runtime_error(
            "expected PCA transform shape to be num_samples x num_components"
        );
    }
}

void test_pca_fit_transform_matches_transform() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model_fit_transform(options);
    const ml::Matrix Z_fit_transform =
        model_fit_transform.fit_transform(X);

    ml::PCA model_transform(options);
    model_transform.fit(X);

    const ml::Matrix Z_transform =
        model_transform.transform(X);

    if (
        Z_fit_transform.rows() != Z_transform.rows() ||
        Z_fit_transform.cols() != Z_transform.cols()
    ) {
        throw std::runtime_error(
            "expected fit_transform and transform shapes to match"
        );
    }

    for (Eigen::Index i = 0; i < Z_fit_transform.rows(); ++i) {
        for (Eigen::Index j = 0; j < Z_fit_transform.cols(); ++j) {
            test::assert_almost_equal(
                std::abs(Z_fit_transform(i, j)),
                std::abs(Z_transform(i, j)),
                "expected fit_transform and transform to match up to sign"
            );
        }
    }
}

void test_pca_inverse_transform_returns_original_shape() {
    const ml::Matrix X = make_pca_2d_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    const ml::Matrix Z =
        model.fit_transform(X);

    const ml::Matrix reconstructed =
        model.inverse_transform(Z);

    if (reconstructed.rows() != X.rows() || reconstructed.cols() != X.cols()) {
        throw std::runtime_error(
            "expected inverse_transform to return original feature shape"
        );
    }
}

void test_pca_full_components_reconstructs_training_data() {
    const ml::Matrix X = make_pca_2d_test_X();

    ml::PCAOptions options;
    options.num_components = 2;

    ml::PCA model(options);
    const ml::Matrix Z =
        model.fit_transform(X);

    const ml::Matrix reconstructed =
        model.inverse_transform(Z);

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            test::assert_almost_equal(
                reconstructed(i, j),
                X(i, j),
                "expected full PCA reconstruction to recover original data",
                1e-8
            );
        }
    }
}

void test_pca_rejects_transform_before_fit() {
    const ml::Matrix X = make_pca_test_X();

    const ml::PCA model;

    static_cast<void>(
        model.transform(X)
    );
}

void test_pca_rejects_inverse_transform_before_fit() {
    ml::Matrix Z(2, 1);
    Z << 1.0,
         2.0;

    const ml::PCA model;

    static_cast<void>(
        model.inverse_transform(Z)
    );
}

void test_pca_rejects_empty_fit_matrix() {
    const ml::Matrix X;

    ml::PCA model;
    model.fit(X);
}

void test_pca_rejects_single_sample_fit_matrix() {
    ml::Matrix X(1, 2);
    X << 1.0, 2.0;

    ml::PCA model;
    model.fit(X);
}

void test_pca_rejects_non_finite_fit_values() {
    ml::Matrix X = make_pca_test_X();
    X(0, 0) = std::numeric_limits<double>::quiet_NaN();

    ml::PCA model;
    model.fit(X);
}

void test_pca_rejects_too_many_components() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 3;

    ml::PCA model(options);
    model.fit(X);
}

void test_pca_rejects_transform_feature_mismatch() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    model.fit(X);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(
        model.transform(X_bad)
    );
}

void test_pca_rejects_inverse_transform_component_mismatch() {
    const ml::Matrix X = make_pca_test_X();

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    model.fit(X);

    ml::Matrix Z_bad(2, 2);
    Z_bad.setZero();

    static_cast<void>(
        model.inverse_transform(Z_bad)
    );
}

void test_pca_accessors_reject_before_fit() {
    const ml::PCA model;

    static_cast<void>(model.mean());
}

// -----------------------------------------------------------------------------
// Phase 8.4 PCA dimensionality-reduction experiment helpers
// -----------------------------------------------------------------------------

struct PCAProjectionRow {
    std::string sample_name;
    double original_x0{0.0};
    double original_x1{0.0};
    double original_x2{0.0};
    double pc1{0.0};
    double pc2{0.0};
    double cluster_label{0.0};
};

struct PCAExplainedVarianceRow {
    std::size_t component_index{0};
    double explained_variance{0.0};
    double explained_variance_ratio{0.0};
};

struct PCAReconstructionSummary {
    std::size_t num_components{0};
    double reconstruction_mse{0.0};
    double retained_variance_ratio{0.0};
};

ml::Matrix make_pca_experiment_X() {
    ml::Matrix X(10, 3);

    X << 0.0, 0.0, 0.1,
         0.2, 0.1, 0.0,
         0.1, 0.2, 0.2,
         0.3, 0.2, 0.1,
         0.2, 0.3, 0.2,

         5.0, 5.0, 5.1,
         5.2, 5.1, 5.0,
         5.1, 5.2, 5.2,
         5.3, 5.2, 5.1,
         5.2, 5.3, 5.2;

    return X;
}

double reconstruction_mse(
    const ml::Matrix& X,
    const ml::Matrix& reconstructed
) {
    if (X.rows() != reconstructed.rows() || X.cols() != reconstructed.cols()) {
        throw std::runtime_error(
            "reconstruction_mse: matrices must have the same shape"
        );
    }

    double total = 0.0;

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            const double residual =
                X(i, j) - reconstructed(i, j);

            total += residual * residual;
        }
    }

    return total / static_cast<double>(X.rows() * X.cols());
}

double retained_variance_ratio(
    const ml::Vector& explained_variance_ratio
) {
    double total = 0.0;

    for (Eigen::Index i = 0; i < explained_variance_ratio.size(); ++i) {
        total += explained_variance_ratio(i);
    }

    return total;
}

std::vector<PCAExplainedVarianceRow> make_pca_explained_variance_rows(
    const ml::PCA& model
) {
    const ml::Vector& variance =
        model.explained_variance();

    const ml::Vector& ratio =
        model.explained_variance_ratio();

    std::vector<PCAExplainedVarianceRow> rows;
    rows.reserve(static_cast<std::size_t>(variance.size()));

    for (Eigen::Index i = 0; i < variance.size(); ++i) {
        rows.push_back(
            PCAExplainedVarianceRow{
                static_cast<std::size_t>(i + 1),
                variance(i),
                ratio(i)
            }
        );
    }

    return rows;
}

std::vector<PCAProjectionRow> make_pca_projection_rows(
    const ml::Matrix& X,
    const ml::Matrix& Z,
    const ml::Vector& cluster_labels
) {
    if (X.rows() != Z.rows() || X.rows() != cluster_labels.size()) {
        throw std::runtime_error(
            "make_pca_projection_rows: X, Z, and cluster labels must have matching rows"
        );
    }

    if (X.cols() < 3 || Z.cols() < 2) {
        throw std::runtime_error(
            "make_pca_projection_rows: expected at least 3 original features and 2 PCA components"
        );
    }

    std::vector<PCAProjectionRow> rows;
    rows.reserve(static_cast<std::size_t>(X.rows()));

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        rows.push_back(
            PCAProjectionRow{
                "sample_" + std::to_string(i),
                X(i, 0),
                X(i, 1),
                X(i, 2),
                Z(i, 0),
                Z(i, 1),
                cluster_labels(i)
            }
        );
    }

    return rows;
}

PCAReconstructionSummary make_pca_reconstruction_summary(
    const ml::Matrix& X,
    std::size_t num_components
) {
    ml::PCAOptions options;
    options.num_components = num_components;

    ml::PCA model(options);

    const ml::Matrix Z =
        model.fit_transform(X);

    const ml::Matrix reconstructed =
        model.inverse_transform(Z);

    return PCAReconstructionSummary{
        num_components,
        reconstruction_mse(
            X,
            reconstructed
        ),
        retained_variance_ratio(
            model.explained_variance_ratio()
        )
    };
}

void export_pca_explained_variance_csv(
    const std::vector<PCAExplainedVarianceRow>& rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_pca_explained_variance_csv: failed to open output file"
        );
    }

    file << "component_index,explained_variance,explained_variance_ratio\n";

    for (const PCAExplainedVarianceRow& row : rows) {
        file << row.component_index << ","
             << row.explained_variance << ","
             << row.explained_variance_ratio << "\n";
    }
}

void export_pca_projection_csv(
    const std::vector<PCAProjectionRow>& rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_pca_projection_csv: failed to open output file"
        );
    }

    file << "sample_name,original_x0,original_x1,original_x2,"
         << "pc1,pc2,cluster_label\n";

    for (const PCAProjectionRow& row : rows) {
        file << row.sample_name << ","
             << row.original_x0 << ","
             << row.original_x1 << ","
             << row.original_x2 << ","
             << row.pc1 << ","
             << row.pc2 << ","
             << row.cluster_label << "\n";
    }
}

void export_pca_reconstruction_summary_txt(
    const std::vector<PCAReconstructionSummary>& summaries,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_pca_reconstruction_summary_txt: failed to open output file"
        );
    }

    file << "PCA Reconstruction Summary\n\n";

    file << "This experiment compares reconstruction quality for different numbers of PCA components.\n\n";

    file << "Interpretation:\n";
    file << "- More components should generally retain more variance.\n";
    file << "- More components should generally reduce reconstruction error.\n";
    file << "- Keeping all components should reconstruct the original data up to numerical precision.\n\n";

    for (const PCAReconstructionSummary& summary : summaries) {
        file << "num_components: " << summary.num_components << "\n"
             << "retained_variance_ratio: " << summary.retained_variance_ratio << "\n"
             << "reconstruction_mse: " << summary.reconstruction_mse << "\n\n";
    }
}

void export_unsupervised_representation_summary_txt(
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_unsupervised_representation_summary_txt: failed to open output file"
        );
    }

    file << "Unsupervised Representation Summary\n\n";

    file << "This output connects the two Phase 8 models qualitatively.\n\n";

    file << "KMeans:\n";
    file << "- discovers groups using distances to centroids\n";
    file << "- produces cluster labels and centroids\n";
    file << "- depends strongly on k and feature scale\n\n";

    file << "PCA:\n";
    file << "- discovers high-variance directions\n";
    file << "- produces lower-dimensional projections\n";
    file << "- explains how much variance is retained by each component\n\n";

    file << "Combined interpretation:\n";
    file << "- PCA projection can be used to inspect whether KMeans clusters are geometrically separated.\n";
    file << "- KMeans assigns groups; PCA gives a coordinate system for visualizing structure.\n";
    file << "- Both methods depend on feature geometry and are sensitive to scaling.\n\n";
}

// -----------------------------------------------------------------------------
// Phase 8.4 PCA dimensionality-reduction experiment export tests
// -----------------------------------------------------------------------------

void test_experiment_exports_pca_dimensionality_reduction_outputs() {
    ensure_phase8_output_dir_exists();

    const ml::Matrix X =
        make_pca_experiment_X();

    ml::PCAOptions pca_options;
    pca_options.num_components = 2;

    ml::PCA pca(pca_options);

    const ml::Matrix Z =
        pca.fit_transform(X);

    ml::KMeansOptions kmeans_options;
    kmeans_options.num_clusters = 2;
    kmeans_options.max_iterations = 50;
    kmeans_options.tolerance = 1e-9;

    ml::KMeans kmeans(kmeans_options);
    const ml::Vector cluster_labels =
        kmeans.fit_predict(X);

    const std::vector<PCAExplainedVarianceRow> variance_rows =
        make_pca_explained_variance_rows(pca);

    const std::vector<PCAProjectionRow> projection_rows =
        make_pca_projection_rows(
            X,
            Z,
            cluster_labels
        );

    std::vector<PCAReconstructionSummary> reconstruction_summaries;

    for (std::size_t num_components : {1, 2, 3}) {
        reconstruction_summaries.push_back(
            make_pca_reconstruction_summary(
                X,
                num_components
            )
        );
    }

    const std::string explained_variance_path =
        k_phase8_output_dir + "/pca_explained_variance.csv";

    const std::string projection_path =
        k_phase8_output_dir + "/pca_projection.csv";

    const std::string reconstruction_summary_path =
        k_phase8_output_dir + "/pca_reconstruction_summary.txt";

    const std::string representation_summary_path =
        k_phase8_output_dir + "/unsupervised_representation_summary.txt";

    export_pca_explained_variance_csv(
        variance_rows,
        explained_variance_path
    );

    export_pca_projection_csv(
        projection_rows,
        projection_path
    );

    export_pca_reconstruction_summary_txt(
        reconstruction_summaries,
        reconstruction_summary_path
    );

    export_unsupervised_representation_summary_txt(
        representation_summary_path
    );

    if (!std::filesystem::exists(explained_variance_path)) {
        throw std::runtime_error(
            "expected pca_explained_variance.csv to exist"
        );
    }

    if (!std::filesystem::exists(projection_path)) {
        throw std::runtime_error(
            "expected pca_projection.csv to exist"
        );
    }

    if (!std::filesystem::exists(reconstruction_summary_path)) {
        throw std::runtime_error(
            "expected pca_reconstruction_summary.txt to exist"
        );
    }

    if (!std::filesystem::exists(representation_summary_path)) {
        throw std::runtime_error(
            "expected unsupervised_representation_summary.txt to exist"
        );
    }

    if (variance_rows.empty()) {
        throw std::runtime_error(
            "expected non-empty explained variance rows"
        );
    }

    if (projection_rows.empty()) {
        throw std::runtime_error(
            "expected non-empty PCA projection rows"
        );
    }

    if (reconstruction_summaries.size() != 3) {
        throw std::runtime_error(
            "expected reconstruction summaries for 1, 2, and 3 components"
        );
    }

    for (const PCAExplainedVarianceRow& row : variance_rows) {
        if (!std::isfinite(row.explained_variance)) {
            throw std::runtime_error(
                "expected finite explained variance"
            );
        }

        if (
            !std::isfinite(row.explained_variance_ratio) ||
            row.explained_variance_ratio < 0.0 ||
            row.explained_variance_ratio > 1.0 + 1e-9
        ) {
            throw std::runtime_error(
                "expected explained variance ratio in [0, 1]"
            );
        }
    }

    for (const PCAProjectionRow& row : projection_rows) {
        if (!std::isfinite(row.pc1) || !std::isfinite(row.pc2)) {
            throw std::runtime_error(
                "expected finite PCA projection coordinates"
            );
        }
    }

    for (const PCAReconstructionSummary& summary : reconstruction_summaries) {
        if (!std::isfinite(summary.reconstruction_mse) || summary.reconstruction_mse < 0.0) {
            throw std::runtime_error(
                "expected finite non-negative reconstruction MSE"
            );
        }

        if (
            !std::isfinite(summary.retained_variance_ratio) ||
            summary.retained_variance_ratio < 0.0 ||
            summary.retained_variance_ratio > 1.0 + 1e-9
        ) {
            throw std::runtime_error(
                "expected retained variance ratio in [0, 1]"
            );
        }
    }

    const double mse_1 =
        reconstruction_summaries[0].reconstruction_mse;

    const double mse_3 =
        reconstruction_summaries[2].reconstruction_mse;

    if (mse_3 > mse_1 + 1e-9) {
        throw std::runtime_error(
            "expected reconstruction error with 3 components to be no worse than with 1 component"
        );
    }
}

// -----------------------------------------------------------------------------
// Test runners
// -----------------------------------------------------------------------------

void run_kmeans_tests() {
    std::cout << "\n[Phase 8.1] KMeans tests\n\n";

    test::expect_no_throw(
        "KMeansOptions accepts defaults",
        test_kmeans_options_accept_defaults
    );

    test::expect_invalid_argument(
        "KMeansOptions rejects zero clusters",
        test_kmeans_options_reject_zero_clusters
    );

    test::expect_invalid_argument(
        "KMeansOptions rejects zero max_iterations",
        test_kmeans_options_reject_zero_max_iterations
    );

    test::expect_invalid_argument(
        "KMeansOptions rejects negative tolerance",
        test_kmeans_options_reject_negative_tolerance
    );

    test::expect_invalid_argument(
        "KMeansOptions rejects non-finite tolerance",
        test_kmeans_options_reject_non_finite_tolerance
    );

    test::expect_no_throw(
        "KMeans reports not fitted initially",
        test_kmeans_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "KMeans fit marks model as fitted",
        test_kmeans_fit_marks_model_as_fitted
    );

    test::expect_no_throw(
        "KMeans centroids have expected shape",
        test_kmeans_centroids_have_expected_shape
    );

    test::expect_no_throw(
        "KMeans labels have expected shape",
        test_kmeans_labels_have_expected_shape
    );

    test::expect_no_throw(
        "KMeans fit_predict matches labels",
        test_kmeans_fit_predict_matches_labels
    );

    test::expect_no_throw(
        "KMeans predict returns expected shape",
        test_kmeans_predict_returns_expected_shape
    );

    test::expect_no_throw(
        "KMeans learns two separated clusters",
        test_kmeans_learns_two_separated_clusters
    );

    test::expect_no_throw(
        "KMeans records inertia history",
        test_kmeans_inertia_history_is_recorded
    );

    test::expect_no_throw(
        "KMeans inertia does not increase",
        test_kmeans_inertia_does_not_increase
    );

    test::expect_invalid_argument(
        "KMeans rejects predict before fit",
        test_kmeans_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "KMeans rejects empty fit matrix",
        test_kmeans_rejects_empty_fit_matrix
    );

    test::expect_invalid_argument(
        "KMeans rejects non-finite fit values",
        test_kmeans_rejects_non_finite_fit_values
    );

    test::expect_invalid_argument(
        "KMeans rejects too many clusters",
        test_kmeans_rejects_too_many_clusters
    );

    test::expect_invalid_argument(
        "KMeans rejects predict feature mismatch",
        test_kmeans_rejects_predict_feature_mismatch
    );

    test::expect_invalid_argument(
        "KMeans rejects non-finite predict values",
        test_kmeans_rejects_non_finite_predict_values
    );

    test::expect_invalid_argument(
        "KMeans centroids reject before fit",
        test_kmeans_centroids_reject_before_fit
    );

    test::expect_invalid_argument(
        "KMeans labels reject before fit",
        test_kmeans_labels_reject_before_fit
    );

    test::expect_invalid_argument(
        "KMeans inertia reject before fit",
        test_kmeans_inertia_reject_before_fit
    );
}

void run_kmeans_experiment_export_tests() {
    std::cout << "\n[Phase 8.2] KMeans experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports KMeans cluster comparison",
        test_experiment_exports_kmeans_cluster_comparison
    );
}

void run_pca_tests() {
    std::cout << "\n[Phase 8.3] PCA tests\n\n";

    test::expect_no_throw(
        "PCAOptions accepts defaults",
        test_pca_options_accept_defaults
    );

    test::expect_invalid_argument(
        "PCAOptions rejects zero components",
        test_pca_options_reject_zero_components
    );

    test::expect_no_throw(
        "PCA reports not fitted initially",
        test_pca_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "PCA fit marks model as fitted",
        test_pca_fit_marks_model_as_fitted
    );

    test::expect_no_throw(
        "PCA mean has expected shape",
        test_pca_mean_has_expected_shape
    );

    test::expect_no_throw(
        "PCA components have expected shape",
        test_pca_components_have_expected_shape
    );

    test::expect_no_throw(
        "PCA explained variance has expected shape",
        test_pca_explained_variance_has_expected_shape
    );

    test::expect_no_throw(
        "PCA first component explains most variance for collinear data",
        test_pca_first_component_explains_most_variance_for_collinear_data
    );

    test::expect_no_throw(
        "PCA transform returns expected shape",
        test_pca_transform_returns_expected_shape
    );

    test::expect_no_throw(
        "PCA fit_transform matches transform",
        test_pca_fit_transform_matches_transform
    );

    test::expect_no_throw(
        "PCA inverse_transform returns original shape",
        test_pca_inverse_transform_returns_original_shape
    );

    test::expect_no_throw(
        "PCA full components reconstructs training data",
        test_pca_full_components_reconstructs_training_data
    );

    test::expect_invalid_argument(
        "PCA rejects transform before fit",
        test_pca_rejects_transform_before_fit
    );

    test::expect_invalid_argument(
        "PCA rejects inverse_transform before fit",
        test_pca_rejects_inverse_transform_before_fit
    );

    test::expect_invalid_argument(
        "PCA rejects empty fit matrix",
        test_pca_rejects_empty_fit_matrix
    );

    test::expect_invalid_argument(
        "PCA rejects single-sample fit matrix",
        test_pca_rejects_single_sample_fit_matrix
    );

    test::expect_invalid_argument(
        "PCA rejects non-finite fit values",
        test_pca_rejects_non_finite_fit_values
    );

    test::expect_invalid_argument(
        "PCA rejects too many components",
        test_pca_rejects_too_many_components
    );

    test::expect_invalid_argument(
        "PCA rejects transform feature mismatch",
        test_pca_rejects_transform_feature_mismatch
    );

    test::expect_invalid_argument(
        "PCA rejects inverse_transform component mismatch",
        test_pca_rejects_inverse_transform_component_mismatch
    );

    test::expect_invalid_argument(
        "PCA accessors reject before fit",
        test_pca_accessors_reject_before_fit
    );
}

void run_pca_experiment_export_tests() {
    std::cout << "\n[Phase 8.4] PCA experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports PCA dimensionality-reduction outputs",
        test_experiment_exports_pca_dimensionality_reduction_outputs
    );
}

}  // namespace

namespace ml::experiments {

void run_phase8_unsupervised_sanity() {
    run_kmeans_tests();
    run_kmeans_experiment_export_tests();
    run_pca_tests();
    run_pca_experiment_export_tests();
}

}  // namespace ml::experiments