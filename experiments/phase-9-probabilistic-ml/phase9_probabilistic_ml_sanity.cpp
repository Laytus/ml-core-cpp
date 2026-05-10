#include "phase9_probabilistic_ml_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/types.hpp"
#include "ml/probabilistic/naive_bayes.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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
// Phase 9.1 GaussianNaiveBayes tests
// -----------------------------------------------------------------------------

ml::Matrix make_gaussian_nb_test_X() {
    ml::Matrix X(6, 2);

    X << 0.0, 0.1,
         0.2, 0.0,
         0.1, 0.2,

         5.0, 5.1,
         5.2, 5.0,
         5.1, 5.2;

    return X;
}

ml::Vector make_gaussian_nb_test_y() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    });
}

void test_gaussian_nb_options_accept_defaults() {
    const ml::GaussianNaiveBayesOptions options;

    ml::validate_gaussian_naive_bayes_options(
        options,
        "test_gaussian_nb_options_accept_defaults"
    );
}

void test_gaussian_nb_options_accept_valid_values() {
    ml::GaussianNaiveBayesOptions options;
    options.variance_smoothing = 1e-6;

    ml::validate_gaussian_naive_bayes_options(
        options,
        "test_gaussian_nb_options_accept_valid_values"
    );
}

void test_gaussian_nb_options_reject_zero_variance_smoothing() {
    ml::GaussianNaiveBayesOptions options;
    options.variance_smoothing = 0.0;

    ml::validate_gaussian_naive_bayes_options(
        options,
        "test_gaussian_nb_options_reject_zero_variance_smoothing"
    );
}

void test_gaussian_nb_options_reject_negative_variance_smoothing() {
    ml::GaussianNaiveBayesOptions options;
    options.variance_smoothing = -1e-9;

    ml::validate_gaussian_naive_bayes_options(
        options,
        "test_gaussian_nb_options_reject_negative_variance_smoothing"
    );
}

void test_gaussian_nb_options_reject_non_finite_variance_smoothing() {
    ml::GaussianNaiveBayesOptions options;
    options.variance_smoothing =
        std::numeric_limits<double>::infinity();

    ml::validate_gaussian_naive_bayes_options(
        options,
        "test_gaussian_nb_options_reject_non_finite_variance_smoothing"
    );
}

void test_gaussian_nb_reports_not_fitted_initially() {
    const ml::GaussianNaiveBayes model;

    if (model.is_fitted()) {
        throw std::runtime_error(
            "expected GaussianNaiveBayes to start unfitted"
        );
    }
}

void test_gaussian_nb_fit_marks_model_as_fitted() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    if (!model.is_fitted()) {
        throw std::runtime_error(
            "expected GaussianNaiveBayes to be fitted"
        );
    }
}

void test_gaussian_nb_estimates_classes_and_priors() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    const ml::Vector& classes =
        model.classes();

    const ml::Vector& priors =
        model.class_priors();

    if (classes.size() != 2 || priors.size() != 2) {
        throw std::runtime_error(
            "expected two classes and two priors"
        );
    }

    test::assert_almost_equal(
        classes(0),
        0.0,
        "expected first class label"
    );

    test::assert_almost_equal(
        classes(1),
        1.0,
        "expected second class label"
    );

    test::assert_almost_equal(
        priors(0),
        0.5,
        "expected class 0 prior"
    );

    test::assert_almost_equal(
        priors(1),
        0.5,
        "expected class 1 prior"
    );
}

void test_gaussian_nb_estimates_class_feature_means() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    const ml::Matrix& means =
        model.means();

    if (means.rows() != 2 || means.cols() != 2) {
        throw std::runtime_error(
            "expected means shape to be num_classes x num_features"
        );
    }

    test::assert_almost_equal(
        means(0, 0),
        0.1,
        "expected class 0 feature 0 mean"
    );

    test::assert_almost_equal(
        means(0, 1),
        0.1,
        "expected class 0 feature 1 mean"
    );

    test::assert_almost_equal(
        means(1, 0),
        5.1,
        "expected class 1 feature 0 mean"
    );

    test::assert_almost_equal(
        means(1, 1),
        5.1,
        "expected class 1 feature 1 mean"
    );
}

void test_gaussian_nb_estimates_positive_variances() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    const ml::Matrix& variances =
        model.variances();

    if (variances.rows() != 2 || variances.cols() != 2) {
        throw std::runtime_error(
            "expected variances shape to be num_classes x num_features"
        );
    }

    for (Eigen::Index i = 0; i < variances.rows(); ++i) {
        for (Eigen::Index j = 0; j < variances.cols(); ++j) {
            if (!std::isfinite(variances(i, j)) || variances(i, j) <= 0.0) {
                throw std::runtime_error(
                    "expected positive finite variances"
                );
            }
        }
    }
}

void test_gaussian_nb_predict_returns_expected_shape() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    const ml::Vector predictions =
        model.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error(
            "expected one prediction per sample"
        );
    }
}

void test_gaussian_nb_predicts_training_clusters() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    const ml::Vector predictions =
        model.predict(X);

    test::assert_vector_almost_equal(
        predictions,
        y,
        "test_gaussian_nb_predicts_training_clusters"
    );
}

void test_gaussian_nb_predict_proba_returns_expected_shape() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    const ml::Matrix probabilities =
        model.predict_proba(X);

    if (
        probabilities.rows() != X.rows() ||
        probabilities.cols() != 2
    ) {
        throw std::runtime_error(
            "expected probability matrix shape to be num_samples x num_classes"
        );
    }
}

void test_gaussian_nb_predict_log_proba_returns_expected_shape() {
    const ml::Matrix X = make_gaussian_nb_test_X();

    ml::GaussianNaiveBayes model;
    model.fit(
        X,
        make_gaussian_nb_test_y()
    );

    const ml::Matrix log_probabilities =
        model.predict_log_proba(X);

    if (
        log_probabilities.rows() != X.rows() ||
        log_probabilities.cols() != 2
    ) {
        throw std::runtime_error(
            "expected log probability matrix shape to be num_samples x num_classes"
        );
    }
}

void test_gaussian_nb_probability_rows_sum_to_one() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    const ml::Matrix probabilities =
        model.predict_proba(X);

    for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
        double row_sum = 0.0;

        for (Eigen::Index j = 0; j < probabilities.cols(); ++j) {
            row_sum += probabilities(i, j);
        }

        test::assert_almost_equal(
            row_sum,
            1.0,
            "expected probability row to sum to one",
            1e-9
        );
    }
}

void test_gaussian_nb_predict_matches_probability_argmax() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    const ml::Vector predictions =
        model.predict(X);

    const ml::Matrix probabilities =
        model.predict_proba(X);

    const ml::Vector& classes =
        model.classes();

    for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
        Eigen::Index best_index = 0;
        double best_value = probabilities(i, 0);

        for (Eigen::Index j = 1; j < probabilities.cols(); ++j) {
            if (probabilities(i, j) > best_value) {
                best_value = probabilities(i, j);
                best_index = j;
            }
        }

        test::assert_almost_equal(
            predictions(i),
            classes(best_index),
            "expected prediction to match probability argmax"
        );
    }
}

void test_gaussian_nb_supports_multiclass_targets() {
    ml::Matrix X(6, 2);

    X << 0.0, 0.0,
         0.1, 0.2,
         5.0, 5.0,
         5.1, 5.2,
         9.0, 0.0,
         9.1, 0.2;

    const ml::Vector y =
        make_vector({
            0.0,
            0.0,
            1.0,
            1.0,
            2.0,
            2.0
        });

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    if (model.classes().size() != 3) {
        throw std::runtime_error(
            "expected three classes"
        );
    }

    const ml::Matrix probabilities =
        model.predict_proba(X);

    if (probabilities.cols() != 3) {
        throw std::runtime_error(
            "expected three probability columns"
        );
    }
}

void test_gaussian_nb_variance_smoothing_handles_constant_features() {
    ml::Matrix X(4, 2);

    X << 1.0, 1.0,
         1.0, 1.0,
         5.0, 5.0,
         5.0, 5.0;

    const ml::Vector y =
        make_vector({
            0.0,
            0.0,
            1.0,
            1.0
        });

    ml::GaussianNaiveBayesOptions options;
    options.variance_smoothing = 1e-6;

    ml::GaussianNaiveBayes model(options);
    model.fit(X, y);

    const ml::Matrix& variances =
        model.variances();

    for (Eigen::Index i = 0; i < variances.rows(); ++i) {
        for (Eigen::Index j = 0; j < variances.cols(); ++j) {
            test::assert_almost_equal(
                variances(i, j),
                options.variance_smoothing,
                "expected variance smoothing for constant feature",
                1e-12
            );
        }
    }
}

void test_gaussian_nb_rejects_predict_before_fit() {
    const ml::Matrix X = make_gaussian_nb_test_X();

    const ml::GaussianNaiveBayes model;

    static_cast<void>(
        model.predict(X)
    );
}

void test_gaussian_nb_rejects_predict_proba_before_fit() {
    const ml::Matrix X = make_gaussian_nb_test_X();

    const ml::GaussianNaiveBayes model;

    static_cast<void>(
        model.predict_proba(X)
    );
}

void test_gaussian_nb_rejects_empty_fit_X() {
    const ml::Matrix X;
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);
}

void test_gaussian_nb_rejects_empty_fit_y() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y;

    ml::GaussianNaiveBayes model;
    model.fit(X, y);
}

void test_gaussian_nb_rejects_mismatched_fit_data() {
    ml::Matrix X(3, 2);
    X.setZero();

    const ml::Vector y =
        make_vector({
            0.0,
            1.0
        });

    ml::GaussianNaiveBayes model;
    model.fit(X, y);
}

void test_gaussian_nb_rejects_non_integer_targets() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    ml::Vector y = make_gaussian_nb_test_y();
    y(0) = 0.5;

    ml::GaussianNaiveBayes model;
    model.fit(X, y);
}

void test_gaussian_nb_rejects_negative_targets() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    ml::Vector y = make_gaussian_nb_test_y();
    y(0) = -1.0;

    ml::GaussianNaiveBayes model;
    model.fit(X, y);
}

void test_gaussian_nb_rejects_non_finite_fit_values() {
    ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    X(0, 0) = std::numeric_limits<double>::quiet_NaN();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);
}

void test_gaussian_nb_rejects_predict_feature_mismatch() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(
        model.predict(X_bad)
    );
}

void test_gaussian_nb_rejects_non_finite_predict_values() {
    const ml::Matrix X = make_gaussian_nb_test_X();
    const ml::Vector y = make_gaussian_nb_test_y();

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    ml::Matrix X_bad = X;
    X_bad(0, 0) = std::numeric_limits<double>::quiet_NaN();

    static_cast<void>(
        model.predict(X_bad)
    );
}

void test_gaussian_nb_accessors_reject_before_fit() {
    const ml::GaussianNaiveBayes model;

    static_cast<void>(
        model.classes()
    );
}

// -----------------------------------------------------------------------------
// Phase 9.2 GaussianNaiveBayes experiment export helpers
// -----------------------------------------------------------------------------

const std::string k_phase9_output_dir = "outputs/phase-9-probabilistic-ml";

void ensure_phase9_output_dir_exists() {
    std::filesystem::create_directories(k_phase9_output_dir);
}

struct GaussianNBProbabilityRow {
    std::string sample_name;
    double x0{0.0};
    double x1{0.0};
    double true_label{0.0};
    double predicted_label{0.0};
    double probability_class_0{0.0};
    double probability_class_1{0.0};
    double log_probability_class_0{0.0};
    double log_probability_class_1{0.0};
};

struct GaussianNBPriorComparisonRow {
    std::string variant_name;
    double class_0_prior{0.0};
    double class_1_prior{0.0};
    double query_x0{0.0};
    double query_x1{0.0};
    double probability_class_0{0.0};
    double probability_class_1{0.0};
    double predicted_label{0.0};
};

struct GaussianNBVarianceSmoothingRow {
    std::string variant_name;
    double variance_smoothing{0.0};
    double min_variance{0.0};
    double probability_class_0{0.0};
    double probability_class_1{0.0};
    double predicted_label{0.0};
};

ml::Matrix make_gaussian_nb_experiment_X() {
    ml::Matrix X(8, 2);

    X << -0.2,  0.0,
          0.0,  0.1,
          0.2, -0.1,
          0.1,  0.2,
          4.8,  5.0,
          5.0,  5.1,
          5.2,  4.9,
          5.1,  5.2;

    return X;
}

ml::Vector make_gaussian_nb_experiment_y() {
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

ml::Matrix make_gaussian_nb_probability_query_X() {
    ml::Matrix X(5, 2);

    X << 0.0, 0.0,
         0.4, 0.3,
         2.5, 2.5,
         4.6, 4.7,
         5.1, 5.0;

    return X;
}

ml::Vector make_gaussian_nb_probability_query_y() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        1.0,
        1.0
    });
}

ml::Matrix make_gaussian_nb_prior_imbalanced_X() {
    ml::Matrix X(10, 2);

    X << -0.2,  0.0,
          0.0,  0.1,
          0.2, -0.1,
          0.1,  0.2,
         -0.1,  0.1,
          0.3,  0.0,
          0.0, -0.2,
          4.8,  5.0,
          5.0,  5.1,
          5.2,  4.9;

    return X;
}

ml::Vector make_gaussian_nb_prior_imbalanced_y() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    });
}

ml::Matrix make_gaussian_nb_single_query(
    double x0,
    double x1
) {
    ml::Matrix X(1, 2);
    X << x0, x1;
    return X;
}

std::vector<GaussianNBProbabilityRow> make_gaussian_nb_probability_rows() {
    const ml::Matrix X_train =
        make_gaussian_nb_experiment_X();

    const ml::Vector y_train =
        make_gaussian_nb_experiment_y();

    const ml::Matrix X_query =
        make_gaussian_nb_probability_query_X();

    const ml::Vector y_query =
        make_gaussian_nb_probability_query_y();

    ml::GaussianNaiveBayes model;
    model.fit(X_train, y_train);

    const ml::Vector predictions =
        model.predict(X_query);

    const ml::Matrix probabilities =
        model.predict_proba(X_query);

    const ml::Matrix log_probabilities =
        model.predict_log_proba(X_query);

    std::vector<GaussianNBProbabilityRow> rows;
    rows.reserve(static_cast<std::size_t>(X_query.rows()));

    for (Eigen::Index i = 0; i < X_query.rows(); ++i) {
        rows.push_back(
            GaussianNBProbabilityRow{
                "query_" + std::to_string(i),
                X_query(i, 0),
                X_query(i, 1),
                y_query(i),
                predictions(i),
                probabilities(i, 0),
                probabilities(i, 1),
                log_probabilities(i, 0),
                log_probabilities(i, 1)
            }
        );
    }

    return rows;
}

GaussianNBPriorComparisonRow run_gaussian_nb_prior_comparison_variant(
    const std::string& variant_name,
    const ml::Matrix& X_train,
    const ml::Vector& y_train,
    const ml::Matrix& X_query
) {
    ml::GaussianNaiveBayes model;
    model.fit(X_train, y_train);

    const ml::Matrix probabilities =
        model.predict_proba(X_query);

    const ml::Vector predictions =
        model.predict(X_query);

    return GaussianNBPriorComparisonRow{
        variant_name,
        model.class_priors()(0),
        model.class_priors()(1),
        X_query(0, 0),
        X_query(0, 1),
        probabilities(0, 0),
        probabilities(0, 1),
        predictions(0)
    };
}

std::vector<GaussianNBPriorComparisonRow> make_gaussian_nb_prior_comparison_rows() {
    const ml::Matrix X_query =
        make_gaussian_nb_single_query(
            2.4,
            2.4
        );

    std::vector<GaussianNBPriorComparisonRow> rows;

    rows.push_back(
        run_gaussian_nb_prior_comparison_variant(
            "balanced_priors",
            make_gaussian_nb_experiment_X(),
            make_gaussian_nb_experiment_y(),
            X_query
        )
    );

    rows.push_back(
        run_gaussian_nb_prior_comparison_variant(
            "class_0_majority_prior",
            make_gaussian_nb_prior_imbalanced_X(),
            make_gaussian_nb_prior_imbalanced_y(),
            X_query
        )
    );

    return rows;
}

std::vector<GaussianNBVarianceSmoothingRow> make_gaussian_nb_variance_smoothing_rows() {
    ml::Matrix X_train(4, 2);

    X_train << 1.0, 1.0,
               1.0, 1.0,
               5.0, 5.0,
               5.0, 5.0;

    const ml::Vector y_train =
        make_vector({
            0.0,
            0.0,
            1.0,
            1.0
        });

    const ml::Matrix X_query =
        make_gaussian_nb_single_query(
            1.2,
            1.2
        );

    std::vector<GaussianNBVarianceSmoothingRow> rows;

    for (double smoothing : {1e-9, 1e-6, 1e-3}) {
        ml::GaussianNaiveBayesOptions options;
        options.variance_smoothing = smoothing;

        ml::GaussianNaiveBayes model(options);
        model.fit(X_train, y_train);

        const ml::Matrix probabilities =
            model.predict_proba(X_query);

        const ml::Vector predictions =
            model.predict(X_query);

        double min_variance = model.variances()(0, 0);

        for (Eigen::Index i = 0; i < model.variances().rows(); ++i) {
            for (Eigen::Index j = 0; j < model.variances().cols(); ++j) {
                min_variance =
                    std::min(
                        min_variance,
                        model.variances()(i, j)
                    );
            }
        }

        rows.push_back(
            GaussianNBVarianceSmoothingRow{
                "variance_smoothing_" + std::to_string(smoothing),
                smoothing,
                min_variance,
                probabilities(0, 0),
                probabilities(0, 1),
                predictions(0)
            }
        );
    }

    return rows;
}

void export_gaussian_nb_probability_table_csv(
    const std::vector<GaussianNBProbabilityRow>& rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_gaussian_nb_probability_table_csv: failed to open output file"
        );
    }

    file << "sample_name,x0,x1,true_label,predicted_label,"
         << "probability_class_0,probability_class_1,"
         << "log_probability_class_0,log_probability_class_1\n";

    for (const GaussianNBProbabilityRow& row : rows) {
        file << row.sample_name << ","
             << row.x0 << ","
             << row.x1 << ","
             << row.true_label << ","
             << row.predicted_label << ","
             << row.probability_class_0 << ","
             << row.probability_class_1 << ","
             << row.log_probability_class_0 << ","
             << row.log_probability_class_1 << "\n";
    }
}

void export_gaussian_nb_prior_comparison_txt(
    const std::vector<GaussianNBPriorComparisonRow>& prior_rows,
    const std::vector<GaussianNBVarianceSmoothingRow>& smoothing_rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_gaussian_nb_prior_comparison_txt: failed to open output file"
        );
    }

    file << "Gaussian Naive Bayes Prior and Variance Smoothing Comparison\n\n";

    file << "Class prior effect:\n";
    file << "- The same query point is evaluated under balanced and imbalanced training priors.\n";
    file << "- When likelihood evidence is ambiguous, class priors can shift posterior probabilities.\n\n";

    for (const GaussianNBPriorComparisonRow& row : prior_rows) {
        file << "Variant: " << row.variant_name << "\n"
             << "class_0_prior: " << row.class_0_prior << "\n"
             << "class_1_prior: " << row.class_1_prior << "\n"
             << "query: [" << row.query_x0 << ", " << row.query_x1 << "]\n"
             << "P(class 0 | x): " << row.probability_class_0 << "\n"
             << "P(class 1 | x): " << row.probability_class_1 << "\n"
             << "predicted_label: " << row.predicted_label << "\n\n";
    }

    file << "Variance smoothing effect:\n";
    file << "- Constant features produce zero empirical variance.\n";
    file << "- Variance smoothing keeps Gaussian likelihoods numerically valid.\n\n";

    for (const GaussianNBVarianceSmoothingRow& row : smoothing_rows) {
        file << "Variant: " << row.variant_name << "\n"
             << "variance_smoothing: " << row.variance_smoothing << "\n"
             << "min_variance_after_smoothing: " << row.min_variance << "\n"
             << "P(class 0 | x): " << row.probability_class_0 << "\n"
             << "P(class 1 | x): " << row.probability_class_1 << "\n"
             << "predicted_label: " << row.predicted_label << "\n\n";
    }
}

void export_probabilistic_model_summary_txt(
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_probabilistic_model_summary_txt: failed to open output file"
        );
    }

    file << "Probabilistic Model Summary\n\n";

    file << "Phase 9 adds a probability-centered interpretation layer to ML Core.\n\n";

    file << "GaussianNaiveBayes implements:\n";
    file << "- class prior estimation P(y)\n";
    file << "- per-class Gaussian likelihoods P(x_j | y)\n";
    file << "- posterior probability outputs P(y | x)\n";
    file << "- log-probability outputs for numerical stability\n";
    file << "- variance smoothing for constant or near-constant features\n\n";

    file << "Interpretation:\n";
    file << "- fit estimates priors, means, and variances from data.\n";
    file << "- predict_log_proba computes normalized posterior log-probabilities.\n";
    file << "- predict_proba exponentiates normalized log-probabilities.\n";
    file << "- predict chooses the class with the highest posterior probability.\n\n";

    file << "This complements deterministic classifiers by exposing uncertainty and probabilistic scores.\n";
}

void test_experiment_exports_gaussian_nb_probability_outputs() {
    ensure_phase9_output_dir_exists();

    const std::vector<GaussianNBProbabilityRow> probability_rows =
        make_gaussian_nb_probability_rows();

    const std::vector<GaussianNBPriorComparisonRow> prior_rows =
        make_gaussian_nb_prior_comparison_rows();

    const std::vector<GaussianNBVarianceSmoothingRow> smoothing_rows =
        make_gaussian_nb_variance_smoothing_rows();

    const std::string probability_table_path =
        k_phase9_output_dir + "/gaussian_naive_bayes_probability_table.csv";

    const std::string prior_comparison_path =
        k_phase9_output_dir + "/gaussian_naive_bayes_prior_comparison.txt";

    const std::string summary_path =
        k_phase9_output_dir + "/probabilistic_model_summary.txt";

    export_gaussian_nb_probability_table_csv(
        probability_rows,
        probability_table_path
    );

    export_gaussian_nb_prior_comparison_txt(
        prior_rows,
        smoothing_rows,
        prior_comparison_path
    );

    export_probabilistic_model_summary_txt(summary_path);

    if (!std::filesystem::exists(probability_table_path)) {
        throw std::runtime_error(
            "expected gaussian_naive_bayes_probability_table.csv to exist"
        );
    }

    if (!std::filesystem::exists(prior_comparison_path)) {
        throw std::runtime_error(
            "expected gaussian_naive_bayes_prior_comparison.txt to exist"
        );
    }

    if (!std::filesystem::exists(summary_path)) {
        throw std::runtime_error(
            "expected probabilistic_model_summary.txt to exist"
        );
    }

    if (probability_rows.empty()) {
        throw std::runtime_error(
            "expected non-empty GaussianNB probability rows"
        );
    }

    if (prior_rows.size() != 2) {
        throw std::runtime_error(
            "expected balanced and imbalanced prior comparison rows"
        );
    }

    if (smoothing_rows.size() != 3) {
        throw std::runtime_error(
            "expected three variance smoothing comparison rows"
        );
    }

    for (const GaussianNBProbabilityRow& row : probability_rows) {
        const double row_sum =
            row.probability_class_0 + row.probability_class_1;

        test::assert_almost_equal(
            row_sum,
            1.0,
            "expected GaussianNB exported probability row to sum to one",
            1e-9
        );
    }

    for (const GaussianNBVarianceSmoothingRow& row : smoothing_rows) {
        if (!std::isfinite(row.min_variance) || row.min_variance <= 0.0) {
            throw std::runtime_error(
                "expected positive finite smoothed variance"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Test runners
// -----------------------------------------------------------------------------

void run_gaussian_naive_bayes_tests() {
    std::cout << "\n[Phase 9.1] GaussianNaiveBayes tests\n\n";

    test::expect_no_throw(
        "GaussianNaiveBayesOptions accepts defaults",
        test_gaussian_nb_options_accept_defaults
    );

    test::expect_no_throw(
        "GaussianNaiveBayesOptions accepts valid values",
        test_gaussian_nb_options_accept_valid_values
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayesOptions rejects zero variance_smoothing",
        test_gaussian_nb_options_reject_zero_variance_smoothing
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayesOptions rejects negative variance_smoothing",
        test_gaussian_nb_options_reject_negative_variance_smoothing
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayesOptions rejects non-finite variance_smoothing",
        test_gaussian_nb_options_reject_non_finite_variance_smoothing
    );

    test::expect_no_throw(
        "GaussianNaiveBayes reports not fitted initially",
        test_gaussian_nb_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "GaussianNaiveBayes fit marks model as fitted",
        test_gaussian_nb_fit_marks_model_as_fitted
    );

    test::expect_no_throw(
        "GaussianNaiveBayes estimates classes and priors",
        test_gaussian_nb_estimates_classes_and_priors
    );

    test::expect_no_throw(
        "GaussianNaiveBayes estimates class feature means",
        test_gaussian_nb_estimates_class_feature_means
    );

    test::expect_no_throw(
        "GaussianNaiveBayes estimates positive variances",
        test_gaussian_nb_estimates_positive_variances
    );

    test::expect_no_throw(
        "GaussianNaiveBayes predict returns expected shape",
        test_gaussian_nb_predict_returns_expected_shape
    );

    test::expect_no_throw(
        "GaussianNaiveBayes predicts training clusters",
        test_gaussian_nb_predicts_training_clusters
    );

    test::expect_no_throw(
        "GaussianNaiveBayes predict_proba returns expected shape",
        test_gaussian_nb_predict_proba_returns_expected_shape
    );

    test::expect_no_throw(
        "GaussianNaiveBayes predict_log_proba returns expected shape",
        test_gaussian_nb_predict_log_proba_returns_expected_shape
    );

    test::expect_no_throw(
        "GaussianNaiveBayes probability rows sum to one",
        test_gaussian_nb_probability_rows_sum_to_one
    );

    test::expect_no_throw(
        "GaussianNaiveBayes predict matches probability argmax",
        test_gaussian_nb_predict_matches_probability_argmax
    );

    test::expect_no_throw(
        "GaussianNaiveBayes supports multiclass targets",
        test_gaussian_nb_supports_multiclass_targets
    );

    test::expect_no_throw(
        "GaussianNaiveBayes variance smoothing handles constant features",
        test_gaussian_nb_variance_smoothing_handles_constant_features
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects predict before fit",
        test_gaussian_nb_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects predict_proba before fit",
        test_gaussian_nb_rejects_predict_proba_before_fit
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects empty fit X",
        test_gaussian_nb_rejects_empty_fit_X
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects empty fit y",
        test_gaussian_nb_rejects_empty_fit_y
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects mismatched fit data",
        test_gaussian_nb_rejects_mismatched_fit_data
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects non-integer targets",
        test_gaussian_nb_rejects_non_integer_targets
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects negative targets",
        test_gaussian_nb_rejects_negative_targets
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects non-finite fit values",
        test_gaussian_nb_rejects_non_finite_fit_values
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects predict feature mismatch",
        test_gaussian_nb_rejects_predict_feature_mismatch
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes rejects non-finite predict values",
        test_gaussian_nb_rejects_non_finite_predict_values
    );

    test::expect_invalid_argument(
        "GaussianNaiveBayes accessors reject before fit",
        test_gaussian_nb_accessors_reject_before_fit
    );
}

void run_gaussian_naive_bayes_experiment_tests() {
    std::cout << "\n[Phase 9.2] GaussianNaiveBayes experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports GaussianNaiveBayes probability outputs",
        test_experiment_exports_gaussian_nb_probability_outputs
    );
}

}  // namespace

namespace ml::experiments {

void run_phase9_probabilistic_ml_sanity() {
    run_gaussian_naive_bayes_tests();
    run_gaussian_naive_bayes_experiment_tests();
}

}  // namespace ml::experiments