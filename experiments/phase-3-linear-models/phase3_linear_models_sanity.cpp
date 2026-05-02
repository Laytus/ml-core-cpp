#include "phase3_linear_models_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/baselines.hpp"
#include "ml/common/evaluation_harness.hpp"
#include "ml/common/experiment_summary.hpp"
#include "ml/common/preprocessing_pipeline.hpp"
#include "ml/common/types.hpp"
#include "ml/linear_models/linear_regression.hpp"
#include "ml/linear_models/regularization.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace test = ml::experiments::test;

namespace {

ml::Matrix make_simple_linear_X() {
    ml::Matrix X(5, 2);
    X << 1.0, 0.0,
         2.0, 1.0,
         3.0, 2.0,
         4.0, 3.0,
         5.0, 4.0;

    return X;
}

ml::Vector make_simple_linear_y() {
    // y = 2*x1 + 3*x2 + 5
    ml::Vector y(5);
    y << 7.0, 12.0, 17.0, 22.0, 27.0;

    return y;
}

ml::LinearRegressionOptions make_stable_options() {
    ml::LinearRegressionOptions options;
    options.learning_rate = 0.01;
    options.max_iterations = 20000;
    options.tolerance = 1e-12;
    options.store_loss_history = true;
    options.regularization = ml::RegularizationConfig::none();

    return options;
}

const std::string k_phase3_output_dir = "outputs/phase-3-linear-models";

void ensure_phase3_output_dir_exists() {
    std::filesystem::create_directories(k_phase3_output_dir);
}

ml::Matrix make_scaled_comparison_X() {
    ml::Matrix X(6, 2);
    X << 1.0, 100.0,
         2.0, 200.0,
         3.0, 300.0,
         4.0, 400.0,
         5.0, 500.0,
         6.0, 600.0;

    return X;
}

ml::Vector make_scaled_comparison_y() {
    ml::Vector y(6);
    y << 12.0, 19.0, 26.0, 33.0, 40.0, 47.0;

    return y;
}

ml::RegressionExperimentSummary make_regression_summary(
    const std::string& experiment_name,
    const std::string& split_name,
    const ml::Vector& y_train,
    const ml::Vector& y_eval,
    const ml::Vector& model_predictions,
    const std::string& model_name
) {
    ml::MeanRegressor baseline;
    baseline.fit(y_train);

    const ml::Vector baseline_predictions =
        baseline.predict(y_eval.size());

    const ml::RegressionEvaluationReport report =
        ml::run_regression_evaluation(
            ml::RegressionEvaluationInput{
                y_eval,
                baseline_predictions,
                model_predictions,
                model_name,
                "MeanRegressor"
            }
        );

    return ml::RegressionExperimentSummary{
        experiment_name,
        "synthetic_linear_regression_dataset",
        split_name,
        report
    };
}

void export_phase3_summary(
    const ml::RegressionExperimentSummary& summary,
    const std::string& file_stem
) {
    ensure_phase3_output_dir_exists();

    ml::export_regression_summary_csv(
        summary,
        k_phase3_output_dir + "/" + file_stem + ".csv"
    );

    ml::export_regression_summary_txt(
        summary,
        k_phase3_output_dir + "/" + file_stem + ".txt"
    );
}

void test_regularization_config_none_is_disabled() {
    const ml::RegularizationConfig config = ml::RegularizationConfig::none();

    if (config.is_enabled()) {
        throw std::runtime_error("expected no regularization to be disabled");
    }

    if (config.is_ridge()) {
        throw std::runtime_error("expected no regularization to not be Ridge");
    }

    if (config.is_lasso()) {
        throw std::runtime_error("expected no regularization to not be Lasso");
    }
}

void test_regularization_config_ridge_is_enabled() {
    const ml::RegularizationConfig config = ml::RegularizationConfig::ridge(0.1);

    if (!config.is_enabled()) {
        throw std::runtime_error("expected Ridge regularization to be enabled");
    }

    if (!config.is_ridge()) {
        throw std::runtime_error("expected config to be Ridge");
    }

    if (config.is_lasso()) {
        throw std::runtime_error("expected Ridge config to not be Lasso");
    }

    test::assert_almost_equal(
        config.lambda,
        0.1,
        "test_regularization_config_ridge_is_enabled lambda"
    );
}

void test_regularization_config_rejects_negative_ridge_lambda() {
    static_cast<void>(ml::RegularizationConfig::ridge(-0.1));
}

void test_linear_regression_rejects_empty_X() {
    ml::Matrix X(0, 0);
    ml::Vector y(0);

    ml::LinearRegression model;
    model.fit(X, y);
}

void test_linear_regression_rejects_mismatched_X_y() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector y(2);
    y << 1.0, 2.0;

    ml::LinearRegression model;
    model.fit(X, y);
}

void test_linear_regression_rejects_predict_before_fit() {
    ml::Matrix X = make_simple_linear_X();

    ml::LinearRegression model;
    static_cast<void>(model.predict(X));
}

void test_linear_regression_fits_simple_multivariate_data() {
    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegression model(make_stable_options());
    model.fit(X, y);

    if (!model.is_fitted()) {
        throw std::runtime_error("expected model to be fitted");
    }

    const double mse = model.score_mse(X, y);

    if (mse > 1e-4) {
        throw std::runtime_error(
            "expected small training MSE, got " + std::to_string(mse)
        );
    }
}

void test_linear_regression_predict_returns_expected_shape() {
    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegression model(make_stable_options());
    model.fit(X, y);

    const ml::Vector predictions = model.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error("expected predictions.size() == y.size()");
    }
}

void test_linear_regression_score_mse_is_small_after_fit() {
    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegression model(make_stable_options());
    model.fit(X, y);

    const double mse = model.score_mse(X, y);

    if (mse > 1e-4) {
        throw std::runtime_error(
            "expected score_mse to be small, got " + std::to_string(mse)
        );
    }
}

void test_linear_regression_stores_loss_history() {
    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegression model(make_stable_options());
    model.fit(X, y);

    const ml::LinearRegressionTrainingHistory& history =
        model.training_history();

    if (history.losses.empty()) {
        throw std::runtime_error("expected non-empty loss history");
    }

    if (history.iterations_run == 0) {
        throw std::runtime_error("expected iterations_run > 0");
    }
}

void test_linear_regression_loss_decreases() {
    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegression model(make_stable_options());
    model.fit(X, y);

    const ml::LinearRegressionTrainingHistory& history =
        model.training_history();

    if (history.losses.size() < 2) {
        throw std::runtime_error("expected at least two recorded losses");
    }

    const double first_loss = history.losses.front();
    const double last_loss = history.losses.back();

    if (last_loss >= first_loss) {
        throw std::runtime_error(
            "expected final loss to be lower than initial loss"
        );
    }
}

void test_linear_regression_rejects_invalid_learning_rate() {
    ml::LinearRegressionOptions options = make_stable_options();
    options.learning_rate = 0.0;

    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegression model(options);
    model.fit(X, y);
}

void test_linear_regression_rejects_lasso_for_now() {
    ml::LinearRegressionOptions options = make_stable_options();
    options.regularization = ml::RegularizationConfig::lasso(0.1);

    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegression model(options);
    model.fit(X, y);
}

void test_linear_regression_supports_ridge_lambda_positive() {
    ml::LinearRegressionOptions options = make_stable_options();
    options.regularization = ml::RegularizationConfig::ridge(0.01);

    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegression model(options);
    model.fit(X, y);

    if (!model.is_fitted()) {
        throw std::runtime_error("expected Ridge model to be fitted");
    }

    const double mse = model.score_mse(X, y);

    if (mse > 1.0) {
        throw std::runtime_error(
            "expected Ridge model to fit simple data reasonably, got MSE " +
            std::to_string(mse)
        );
    }
}

void test_experiment_unregularized_vs_ridge_exports_summaries() {
    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegressionOptions unregularized_options = make_stable_options();
    unregularized_options.regularization = ml::RegularizationConfig::none();

    ml::LinearRegression unregularized_model(unregularized_options);
    unregularized_model.fit(X, y);

    const ml::Vector unregularized_predictions =
        unregularized_model.predict(X);

    const ml::RegressionExperimentSummary unregularized_summary =
        make_regression_summary(
            "phase3_unregularized_linear_regression",
            "training_split",
            y,
            y,
            unregularized_predictions,
            "LinearRegression"
        );

    export_phase3_summary(
        unregularized_summary,
        "unregularized_linear_regression_summary"
    );

    ml::LinearRegressionOptions ridge_options = make_stable_options();
    ridge_options.regularization = ml::RegularizationConfig::ridge(0.01);

    ml::LinearRegression ridge_model(ridge_options);
    ridge_model.fit(X, y);

    const ml::Vector ridge_predictions = ridge_model.predict(X);

    const ml::RegressionExperimentSummary ridge_summary =
        make_regression_summary(
            "phase3_ridge_linear_regression",
            "training_split",
            y,
            y,
            ridge_predictions,
            "RidgeRegression"
        );

    export_phase3_summary(
        ridge_summary,
        "ridge_linear_regression_summary"
    );

    if (!std::filesystem::exists(
            k_phase3_output_dir + "/unregularized_linear_regression_summary.csv"
        )) {
        throw std::runtime_error("expected unregularized CSV summary to exist");
    }

    if (!std::filesystem::exists(
            k_phase3_output_dir + "/ridge_linear_regression_summary.csv"
        )) {
        throw std::runtime_error("expected Ridge CSV summary to exist");
    }
}

void test_experiment_scaled_vs_unscaled_data_exports_summaries() {
    const ml::Matrix X_unscaled = make_scaled_comparison_X();
    const ml::Vector y = make_scaled_comparison_y();

    ml::LinearRegressionOptions options = make_stable_options();
    options.learning_rate = 0.000001;
    options.max_iterations = 3000;

    ml::LinearRegression unscaled_model(options);
    unscaled_model.fit(X_unscaled, y);

    const ml::Vector unscaled_predictions =
        unscaled_model.predict(X_unscaled);

    const ml::RegressionExperimentSummary unscaled_summary =
        make_regression_summary(
            "phase3_unscaled_linear_regression",
            "training_split",
            y,
            y,
            unscaled_predictions,
            "LinearRegressionUnscaled"
        );

    export_phase3_summary(
        unscaled_summary,
        "unscaled_linear_regression_summary"
    );

    ml::StandardScaler scaler;
    const ml::Matrix X_scaled = scaler.fit_transform(X_unscaled);

    ml::LinearRegressionOptions scaled_options = make_stable_options();
    scaled_options.learning_rate = 0.01;
    scaled_options.max_iterations = 10000;

    ml::LinearRegression scaled_model(scaled_options);
    scaled_model.fit(X_scaled, y);

    const ml::Vector scaled_predictions = scaled_model.predict(X_scaled);

    const ml::RegressionExperimentSummary scaled_summary =
        make_regression_summary(
            "phase3_scaled_linear_regression",
            "training_split",
            y,
            y,
            scaled_predictions,
            "LinearRegressionScaled"
        );

    export_phase3_summary(
        scaled_summary,
        "scaled_linear_regression_summary"
    );

    if (!std::filesystem::exists(
            k_phase3_output_dir + "/unscaled_linear_regression_summary.txt"
        )) {
        throw std::runtime_error("expected unscaled TXT summary to exist");
    }

    if (!std::filesystem::exists(
            k_phase3_output_dir + "/scaled_linear_regression_summary.txt"
        )) {
        throw std::runtime_error("expected scaled TXT summary to exist");
    }
}

void test_experiment_learning_rate_convergence_comparison_exports_summaries() {
    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegressionOptions slow_options = make_stable_options();
    slow_options.learning_rate = 0.001;
    slow_options.max_iterations = 5000;

    ml::LinearRegression slow_model(slow_options);
    slow_model.fit(X, y);

    const ml::Vector slow_predictions = slow_model.predict(X);

    const ml::RegressionExperimentSummary slow_summary =
        make_regression_summary(
            "phase3_learning_rate_slow",
            "training_split",
            y,
            y,
            slow_predictions,
            "LinearRegressionLR0.001"
        );

    export_phase3_summary(
        slow_summary,
        "learning_rate_slow_summary"
    );

    ml::LinearRegressionOptions faster_options = make_stable_options();
    faster_options.learning_rate = 0.01;
    faster_options.max_iterations = 5000;

    ml::LinearRegression faster_model(faster_options);
    faster_model.fit(X, y);

    const ml::Vector faster_predictions = faster_model.predict(X);

    const ml::RegressionExperimentSummary faster_summary =
        make_regression_summary(
            "phase3_learning_rate_faster",
            "training_split",
            y,
            y,
            faster_predictions,
            "LinearRegressionLR0.01"
        );

    export_phase3_summary(
        faster_summary,
        "learning_rate_faster_summary"
    );

    const auto& slow_history = slow_model.training_history();
    const auto& faster_history = faster_model.training_history();

    if (slow_history.losses.empty()) {
        throw std::runtime_error("expected slow learning-rate loss history");
    }

    if (faster_history.losses.empty()) {
        throw std::runtime_error("expected faster learning-rate loss history");
    }

    if (!std::filesystem::exists(
            k_phase3_output_dir + "/learning_rate_slow_summary.csv"
        )) {
        throw std::runtime_error("expected slow learning-rate CSV summary to exist");
    }

    if (!std::filesystem::exists(
            k_phase3_output_dir + "/learning_rate_faster_summary.csv"
        )) {
        throw std::runtime_error("expected faster learning-rate CSV summary to exist");
    }
}

void export_residual_analysis(
    const ml::Vector& targets,
    const ml::Vector& predictions,
    const std::string& csv_output_path,
    const std::string& txt_output_path
) {
    if (targets.size() != predictions.size()) {
        throw std::runtime_error(
            "export_residual_analysis: targets and predictions must have the same size"
        );
    }

    if (targets.size() == 0) {
        throw std::runtime_error(
            "export_residual_analysis: targets must not be empty"
        );
    }

    std::ofstream csv_file(csv_output_path);

    if (!csv_file.is_open()) {
        throw std::runtime_error(
            "export_residual_analysis: failed to open CSV output file"
        );
    }

    std::ofstream txt_file(txt_output_path);

    if (!txt_file.is_open()) {
        throw std::runtime_error(
            "export_residual_analysis: failed to open TXT output file"
        );
    }

    double residual_sum = 0.0;
    double absolute_residual_sum = 0.0;
    double min_residual = std::numeric_limits<double>::infinity();
    double max_residual = -std::numeric_limits<double>::infinity();
    double largest_absolute_residual = 0.0;

    csv_file << "sample_index,target,prediction,residual,absolute_residual\n";

    for (Eigen::Index i = 0; i < targets.size(); ++i) {
        const double target = targets(i);
        const double prediction = predictions(i);
        const double residual = prediction - target;
        const double absolute_residual = std::abs(residual);

        residual_sum += residual;
        absolute_residual_sum += absolute_residual;
        min_residual = std::min(min_residual, residual);
        max_residual = std::max(max_residual, residual);
        largest_absolute_residual = std::max(
            largest_absolute_residual,
            absolute_residual
        );

        csv_file << i << ","
                 << target << ","
                 << prediction << ","
                 << residual << ","
                 << absolute_residual << "\n";
    }

    const double sample_count = static_cast<double>(targets.size());
    const double mean_residual = residual_sum / sample_count;
    const double mean_absolute_residual = absolute_residual_sum / sample_count;

    txt_file << "Residual Analysis\n\n"
             << "Mean residual: " << mean_residual << "\n"
             << "Mean absolute residual: " << mean_absolute_residual << "\n"
             << "Minimum residual: " << min_residual << "\n"
             << "Maximum residual: " << max_residual << "\n"
             << "Largest absolute residual: " << largest_absolute_residual << "\n\n";

    txt_file << "Interpretation:\n";

    if (std::abs(mean_residual) < 1e-3) {
        txt_file << "  Residuals are centered close to zero.\n";
    } else if (mean_residual > 0.0) {
        txt_file << "  The model tends to overpredict on average.\n";
    } else {
        txt_file << "  The model tends to underpredict on average.\n";
    }

    if (mean_absolute_residual < 1e-2) {
        txt_file << "  Average residual magnitude is very small for this synthetic dataset.\n";
    } else {
        txt_file << "  Average residual magnitude should be inspected relative to target scale.\n";
    }
}

void test_experiment_residual_analysis_exports_files() {
    ensure_phase3_output_dir_exists();

    const ml::Matrix X = make_simple_linear_X();
    const ml::Vector y = make_simple_linear_y();

    ml::LinearRegression model(make_stable_options());
    model.fit(X, y);

    const ml::Vector predictions = model.predict(X);

    const std::string csv_output_path =
        k_phase3_output_dir + "/residual_analysis.csv";
    const std::string txt_output_path =
        k_phase3_output_dir + "/residual_analysis.txt";

    export_residual_analysis(
        y,
        predictions,
        csv_output_path,
        txt_output_path
    );

    if (!std::filesystem::exists(csv_output_path)) {
        throw std::runtime_error("expected residual_analysis.csv to exist");
    }

    if (!std::filesystem::exists(txt_output_path)) {
        throw std::runtime_error("expected residual_analysis.txt to exist");
    }
}

void run_regularization_config_tests() {
    std::cout << "\n[Phase 3.1] Regularization config tests\n\n";

    test::expect_no_throw(
        "RegularizationConfig none is disabled",
        test_regularization_config_none_is_disabled
    );

    test::expect_no_throw(
        "RegularizationConfig ridge is enabled",
        test_regularization_config_ridge_is_enabled
    );

    test::expect_invalid_argument(
        "RegularizationConfig rejects negative Ridge lambda",
        test_regularization_config_rejects_negative_ridge_lambda
    );
}

void run_linear_regression_tests() {
    std::cout << "\n[Phase 3.2] Linear regression implementation tests\n\n";

    test::expect_invalid_argument(
        "LinearRegression rejects empty X/y",
        test_linear_regression_rejects_empty_X
    );

    test::expect_invalid_argument(
        "LinearRegression rejects mismatched X/y",
        test_linear_regression_rejects_mismatched_X_y
    );

    test::expect_invalid_argument(
        "LinearRegression rejects predict before fit",
        test_linear_regression_rejects_predict_before_fit
    );

    test::expect_no_throw(
        "LinearRegression fits simple multivariate data",
        test_linear_regression_fits_simple_multivariate_data
    );

    test::expect_no_throw(
        "LinearRegression predict returns expected shape",
        test_linear_regression_predict_returns_expected_shape
    );

    test::expect_no_throw(
        "LinearRegression score_mse is small after fit",
        test_linear_regression_score_mse_is_small_after_fit
    );

    test::expect_no_throw(
        "LinearRegression stores loss history",
        test_linear_regression_stores_loss_history
    );

    test::expect_no_throw(
        "LinearRegression loss decreases",
        test_linear_regression_loss_decreases
    );

    test::expect_invalid_argument(
        "LinearRegression rejects invalid learning rate",
        test_linear_regression_rejects_invalid_learning_rate
    );

    test::expect_invalid_argument(
        "LinearRegression rejects Lasso fitting for now",
        test_linear_regression_rejects_lasso_for_now
    );

    test::expect_no_throw(
        "LinearRegression supports Ridge lambda > 0",
        test_linear_regression_supports_ridge_lambda_positive
    );
}


void run_linear_regression_experiment_tests() {
    std::cout << "\n[Phase 3.3] Linear regression experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports unregularized vs Ridge summaries",
        test_experiment_unregularized_vs_ridge_exports_summaries
    );

    test::expect_no_throw(
        "Experiment exports scaled vs unscaled summaries",
        test_experiment_scaled_vs_unscaled_data_exports_summaries
    );

    test::expect_no_throw(
        "Experiment exports learning-rate convergence summaries",
        test_experiment_learning_rate_convergence_comparison_exports_summaries
    );
}

void run_residual_analysis_tests() {
    std::cout << "\n[Phase 3.4] Residual analysis export tests\n\n";

    test::expect_no_throw(
        "Experiment exports residual analysis files",
        test_experiment_residual_analysis_exports_files
    );
}

}  // namespace

namespace ml::experiments {

void run_phase3_linear_models_sanity() {
    run_regularization_config_tests();
    run_linear_regression_tests();
    run_linear_regression_experiment_tests();
    run_residual_analysis_tests();
}

}  // namespace ml::experiments