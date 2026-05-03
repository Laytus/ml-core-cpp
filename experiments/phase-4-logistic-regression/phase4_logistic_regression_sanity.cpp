#include "phase4_logistic_regression_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/classification_evaluation.hpp"
#include "ml/common/classification_metrics.hpp"
#include "ml/common/classification_utils.hpp"
#include "ml/common/evaluation_harness.hpp"
#include "ml/common/types.hpp"
#include "ml/linear_models/logistic_regression.hpp"
#include "ml/linear_models/regularization.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace test = ml::experiments::test;

namespace {

ml::Matrix make_binary_classification_X() {
    ml::Matrix X(6, 2);
    X << 0.0, 0.0,
         0.0, 1.0,
         1.0, 0.0,
         4.0, 4.0,
         4.0, 5.0,
         5.0, 4.0;

    return X;
}

ml::Vector make_binary_classification_y() {
    ml::Vector y(6);
    y << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;

    return y;
}

ml::LogisticRegressionOptions make_stable_logistic_options() {
    ml::LogisticRegressionOptions options;
    options.learning_rate = 0.1;
    options.max_iterations = 10000;
    options.tolerance = 1e-12;
    options.store_loss_history = true;
    options.regularization = ml::RegularizationConfig::none();

    return options;
}

const std::string k_phase4_output_dir = "outputs/phase-4-logistic-regression";

void ensure_phase4_output_dir_exists() {
    std::filesystem::create_directories(k_phase4_output_dir);
}

ml::BinaryClassificationEvaluationReport evaluate_model_at_threshold(
    const ml::LogisticRegression& model,
    const ml::Matrix& X,
    const ml::Vector& y,
    double threshold,
    const std::string& model_name
) {
    const ml::Vector probabilities = model.predict_proba(X);
    const ml::Vector predicted_classes = model.predict_classes(X, threshold);

    return ml::run_binary_classification_evaluation(
        ml::BinaryClassificationEvaluationInput{
            y,
            probabilities,
            predicted_classes,
            model_name,
            threshold
        }
    );
}

void export_classification_report(
    const ml::BinaryClassificationEvaluationReport& report,
    const std::string& experiment_name,
    const std::string& csv_output_path,
    const std::string& txt_output_path
) {
    std::ofstream csv_file(csv_output_path);

    if (!csv_file.is_open()) {
        throw std::runtime_error(
            "export_classification_report: failed to open CSV output file"
        );
    }

    std::ofstream txt_file(txt_output_path);

    if (!txt_file.is_open()) {
        throw std::runtime_error(
            "export_classification_report: failed to open TXT output file"
        );
    }

    csv_file << "experiment_name,model_name,threshold,accuracy,precision,recall,f1,bce,"
             << "true_positive,true_negative,false_positive,false_negative\n";

    csv_file << experiment_name << ","
             << report.model_name << ","
             << report.threshold << ","
             << report.evaluation.accuracy << ","
             << report.evaluation.precision << ","
             << report.evaluation.recall << ","
             << report.evaluation.f1 << ","
             << report.evaluation.bce << ","
             << report.evaluation.confusion.true_positive << ","
             << report.evaluation.confusion.true_negative << ","
             << report.evaluation.confusion.false_positive << ","
             << report.evaluation.confusion.false_negative << "\n";

    txt_file << "Experiment: " << experiment_name << "\n"
             << "Model: " << report.model_name << "\n"
             << "Threshold: " << report.threshold << "\n\n";

    txt_file << "Classification metrics:\n"
             << "  Accuracy: " << report.evaluation.accuracy << "\n"
             << "  Precision: " << report.evaluation.precision << "\n"
             << "  Recall: " << report.evaluation.recall << "\n"
             << "  F1: " << report.evaluation.f1 << "\n"
             << "  BCE: " << report.evaluation.bce << "\n\n";

    txt_file << "Confusion matrix:\n"
             << "  True positive: " << report.evaluation.confusion.true_positive << "\n"
             << "  True negative: " << report.evaluation.confusion.true_negative << "\n"
             << "  False positive: " << report.evaluation.confusion.false_positive << "\n"
             << "  False negative: " << report.evaluation.confusion.false_negative << "\n\n";

    txt_file << "Interpretation:\n";

    if (report.has_perfect_accuracy()) {
        txt_file << "  The classifier perfectly separates this synthetic dataset at the selected threshold.\n";
    } else {
        txt_file << "  The classifier makes at least one threshold-dependent classification error.\n";
    }
}

void export_decision_boundary_interpretation(
    const ml::LogisticRegression& model,
    const std::string& csv_output_path,
    const std::string& txt_output_path
) {
    std::ofstream csv_file(csv_output_path);

    if (!csv_file.is_open()) {
        throw std::runtime_error(
            "export_decision_boundary_interpretation: failed to open CSV output file"
        );
    }

    std::ofstream txt_file(txt_output_path);

    if (!txt_file.is_open()) {
        throw std::runtime_error(
            "export_decision_boundary_interpretation: failed to open TXT output file"
        );
    }

    const ml::Vector& weights = model.weights();
    const double bias = model.bias();

    if (weights.size() != 2) {
        throw std::runtime_error(
            "export_decision_boundary_interpretation: expected a 2-feature model"
        );
    }

    csv_file << "weight_0,weight_1,bias\n"
             << weights(0) << ","
             << weights(1) << ","
             << bias << "\n";

    txt_file << "Decision Boundary Interpretation\n\n"
             << "For threshold 0.5, the decision boundary is:\n\n"
             << "  w0 * x0 + w1 * x1 + b = 0\n\n"
             << "Learned parameters:\n"
             << "  w0: " << weights(0) << "\n"
             << "  w1: " << weights(1) << "\n"
             << "  b: " << bias << "\n\n";

    txt_file << "Interpretation:\n"
             << "  Samples with positive logits are classified as class 1 at threshold 0.5.\n"
             << "  Samples with negative logits are classified as class 0 at threshold 0.5.\n"
             << "  Because this experiment uses two features, the boundary is a line in feature space.\n";
}

// ---- Classification utility tests ----

void test_sigmoid_scalar_computes_expected_values() {
    test::assert_almost_equal(
        ml::sigmoid(0.0),
        0.5,
        "test_sigmoid_scalar_computes_expected_values sigmoid(0)"
    );

    if (ml::sigmoid(10.0) <= 0.99) {
        throw std::runtime_error("expected sigmoid(10) to be close to 1");
    }

    if (ml::sigmoid(-10.0) >= 0.01) {
        throw std::runtime_error("expected sigmoid(-10) to be close to 0");
    }
}

void test_sigmoid_vector_computes_expected_values() {
    ml::Vector logits(3);
    logits << 0.0, 10.0, -10.0;

    const ml::Vector probabilities = ml::sigmoid(logits);

    if (probabilities.size() != logits.size()) {
        throw std::runtime_error("expected probabilities.size() == logits.size()");
    }

    test::assert_almost_equal(
        probabilities(0),
        0.5,
        "test_sigmoid_vector_computes_expected_values first probability"
    );

    if (probabilities(1) <= 0.99) {
        throw std::runtime_error("expected positive logit probability close to 1");
    }

    if (probabilities(2) >= 0.01) {
        throw std::runtime_error("expected negative logit probability close to 0");
    }
}

void test_validate_binary_targets_rejects_invalid_values() {
    ml::Vector targets(3);
    targets << 0.0, 1.0, 2.0;

    ml::validate_binary_targets(targets, "test_validate_binary_targets_rejects_invalid_values");
}

void test_threshold_probabilities_computes_expected_classes() {
    ml::Vector probabilities(4);
    probabilities << 0.1, 0.49, 0.5, 0.9;

    const ml::Vector classes = ml::threshold_probabilities(probabilities, 0.5);

    ml::Vector expected(4);
    expected << 0.0, 0.0, 1.0, 1.0;

    test::assert_vector_almost_equal(
        classes,
        expected,
        "test_threshold_probabilities_computes_expected_classes"
    );
}

void test_threshold_probabilities_rejects_invalid_threshold() {
    ml::Vector probabilities(2);
    probabilities << 0.2, 0.8;

    static_cast<void>(ml::threshold_probabilities(probabilities, 1.5));
}

void test_binary_cross_entropy_computes_expected_value() {
    ml::Vector probabilities(2);
    probabilities << 0.8, 0.25;

    ml::Vector targets(2);
    targets << 1.0, 0.0;

    const double result = ml::binary_cross_entropy(probabilities, targets);

    const double expected = -0.5 * (
        std::log(0.8) + std::log(1.0 - 0.25)
    );

    test::assert_almost_equal(
        result,
        expected,
        "test_binary_cross_entropy_computes_expected_value"
    );
}

void test_binary_cross_entropy_rejects_invalid_targets() {
    ml::Vector probabilities(2);
    probabilities << 0.8, 0.25;

    ml::Vector targets(2);
    targets << 1.0, 2.0;

    static_cast<void>(ml::binary_cross_entropy(probabilities, targets));
}

// ---- Classification metric tests ----

void test_confusion_matrix_computes_expected_counts() {
    ml::Vector predictions(4);
    predictions << 1.0, 0.0, 1.0, 0.0;

    ml::Vector targets(4);
    targets << 1.0, 0.0, 0.0, 1.0;

    const ml::ConfusionMatrix matrix = ml::confusion_matrix(predictions, targets);

    if (matrix.true_positive != 1) {
        throw std::runtime_error("expected true_positive == 1");
    }

    if (matrix.true_negative != 1) {
        throw std::runtime_error("expected true_negative == 1");
    }

    if (matrix.false_positive != 1) {
        throw std::runtime_error("expected false_positive == 1");
    }

    if (matrix.false_negative != 1) {
        throw std::runtime_error("expected false_negative == 1");
    }
}

void test_classification_metrics_compute_expected_values() {
    ml::Vector predictions(4);
    predictions << 1.0, 0.0, 1.0, 0.0;

    ml::Vector targets(4);
    targets << 1.0, 0.0, 0.0, 1.0;

    test::assert_almost_equal(
        ml::accuracy_score(predictions, targets),
        0.5,
        "test_classification_metrics_compute_expected_values accuracy"
    );

    test::assert_almost_equal(
        ml::precision_score(predictions, targets),
        0.5,
        "test_classification_metrics_compute_expected_values precision"
    );

    test::assert_almost_equal(
        ml::recall_score(predictions, targets),
        0.5,
        "test_classification_metrics_compute_expected_values recall"
    );

    test::assert_almost_equal(
        ml::f1_score(predictions, targets),
        0.5,
        "test_classification_metrics_compute_expected_values f1"
    );
}

void test_classification_metrics_reject_invalid_predictions() {
    ml::Vector predictions(3);
    predictions << 1.0, 0.0, 0.5;

    ml::Vector targets(3);
    targets << 1.0, 0.0, 1.0;

    static_cast<void>(ml::accuracy_score(predictions, targets));
}

// ---- Classification evaluation harness tests ----

void test_evaluate_binary_classification_returns_expected_metrics() {
    ml::Vector probabilities(4);
    probabilities << 0.9, 0.2, 0.8, 0.4;

    ml::Vector predicted_classes(4);
    predicted_classes << 1.0, 0.0, 1.0, 0.0;

    ml::Vector targets(4);
    targets << 1.0, 0.0, 0.0, 1.0;

    const ml::ClassificationEvaluation evaluation =
        ml::evaluate_binary_classification(
            probabilities,
            predicted_classes,
            targets
        );

    test::assert_almost_equal(
        evaluation.accuracy,
        0.5,
        "test_evaluate_binary_classification_returns_expected_metrics accuracy"
    );

    test::assert_almost_equal(
        evaluation.precision,
        0.5,
        "test_evaluate_binary_classification_returns_expected_metrics precision"
    );

    test::assert_almost_equal(
        evaluation.recall,
        0.5,
        "test_evaluate_binary_classification_returns_expected_metrics recall"
    );

    test::assert_almost_equal(
        evaluation.f1,
        0.5,
        "test_evaluate_binary_classification_returns_expected_metrics f1"
    );

    if (evaluation.bce <= 0.0) {
        throw std::runtime_error("expected BCE to be positive");
    }

    if (evaluation.confusion.true_positive != 1) {
        throw std::runtime_error("expected true_positive == 1");
    }

    if (evaluation.confusion.true_negative != 1) {
        throw std::runtime_error("expected true_negative == 1");
    }

    if (evaluation.confusion.false_positive != 1) {
        throw std::runtime_error("expected false_positive == 1");
    }

    if (evaluation.confusion.false_negative != 1) {
        throw std::runtime_error("expected false_negative == 1");
    }
}

void test_evaluate_binary_classification_rejects_invalid_probabilities() {
    ml::Vector probabilities(3);
    probabilities << 0.2, 1.2, 0.8;

    ml::Vector predicted_classes(3);
    predicted_classes << 0.0, 1.0, 1.0;

    ml::Vector targets(3);
    targets << 0.0, 1.0, 1.0;

    static_cast<void>(ml::evaluate_binary_classification(
        probabilities,
        predicted_classes,
        targets
    ));
}

void test_evaluate_binary_classification_rejects_mismatched_vectors() {
    ml::Vector probabilities(3);
    probabilities << 0.2, 0.7, 0.8;

    ml::Vector predicted_classes(2);
    predicted_classes << 0.0, 1.0;

    ml::Vector targets(3);
    targets << 0.0, 1.0, 1.0;

    static_cast<void>(ml::evaluate_binary_classification(
        probabilities,
        predicted_classes,
        targets
    ));
}

void test_run_binary_classification_evaluation_returns_report() {
    ml::Vector probabilities(4);
    probabilities << 0.9, 0.2, 0.8, 0.4;

    ml::Vector predicted_classes(4);
    predicted_classes << 1.0, 0.0, 1.0, 0.0;

    ml::Vector targets(4);
    targets << 1.0, 0.0, 0.0, 1.0;

    const ml::BinaryClassificationEvaluationReport report =
        ml::run_binary_classification_evaluation(
            ml::BinaryClassificationEvaluationInput{
                targets,
                probabilities,
                predicted_classes,
                "LogisticRegression",
                0.5
            }
        );

    if (report.model_name != "LogisticRegression") {
        throw std::runtime_error("expected model_name == LogisticRegression");
    }

    test::assert_almost_equal(
        report.threshold,
        0.5,
        "test_run_binary_classification_evaluation_returns_report threshold"
    );

    test::assert_almost_equal(
        report.evaluation.accuracy,
        0.5,
        "test_run_binary_classification_evaluation_returns_report accuracy"
    );

    if (report.has_perfect_accuracy()) {
        throw std::runtime_error("expected report to not have perfect accuracy");
    }
}

void test_run_binary_classification_evaluation_detects_perfect_accuracy() {
    ml::Vector probabilities(4);
    probabilities << 0.9, 0.8, 0.1, 0.2;

    ml::Vector predicted_classes(4);
    predicted_classes << 1.0, 1.0, 0.0, 0.0;

    ml::Vector targets(4);
    targets << 1.0, 1.0, 0.0, 0.0;

    const ml::BinaryClassificationEvaluationReport report =
        ml::run_binary_classification_evaluation(
            ml::BinaryClassificationEvaluationInput{
                targets,
                probabilities,
                predicted_classes,
                "PerfectClassifier",
                0.5
            }
        );

    if (!report.has_perfect_accuracy()) {
        throw std::runtime_error("expected report to have perfect accuracy");
    }
}

void test_run_binary_classification_evaluation_rejects_empty_model_name() {
    ml::Vector probabilities(2);
    probabilities << 0.9, 0.1;

    ml::Vector predicted_classes(2);
    predicted_classes << 1.0, 0.0;

    ml::Vector targets(2);
    targets << 1.0, 0.0;

    static_cast<void>(ml::run_binary_classification_evaluation(
        ml::BinaryClassificationEvaluationInput{
            targets,
            probabilities,
            predicted_classes,
            "",
            0.5
        }
    ));
}

void test_run_binary_classification_evaluation_rejects_invalid_threshold() {
    ml::Vector probabilities(2);
    probabilities << 0.9, 0.1;

    ml::Vector predicted_classes(2);
    predicted_classes << 1.0, 0.0;

    ml::Vector targets(2);
    targets << 1.0, 0.0;

    static_cast<void>(ml::run_binary_classification_evaluation(
        ml::BinaryClassificationEvaluationInput{
            targets,
            probabilities,
            predicted_classes,
            "LogisticRegression",
            1.5
        }
    ));
}

// ---- Logistic regression tests ----

void test_logistic_regression_rejects_empty_X_y() {
    ml::Matrix X(0, 0);
    ml::Vector y(0);

    ml::LogisticRegression model;
    model.fit(X, y);
}

void test_logistic_regression_rejects_mismatched_X_y() {
    ml::Matrix X(3, 2);
    X << 0.0, 0.0,
         1.0, 1.0,
         2.0, 2.0;

    ml::Vector y(2);
    y << 0.0, 1.0;

    ml::LogisticRegression model;
    model.fit(X, y);
}

void test_logistic_regression_rejects_non_binary_targets() {
    const ml::Matrix X = make_binary_classification_X();

    ml::Vector y(6);
    y << 0.0, 0.0, 0.0, 1.0, 1.0, 2.0;

    ml::LogisticRegression model;
    model.fit(X, y);
}

void test_logistic_regression_rejects_predict_before_fit() {
    const ml::Matrix X = make_binary_classification_X();

    ml::LogisticRegression model;
    static_cast<void>(model.predict_proba(X));
}

void test_logistic_regression_fits_simple_binary_data() {
    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    if (!model.is_fitted()) {
        throw std::runtime_error("expected LogisticRegression to be fitted");
    }

    const double loss = model.binary_cross_entropy(X, y);

    if (loss > 0.2) {
        throw std::runtime_error(
            "expected small binary cross-entropy, got " + std::to_string(loss)
        );
    }
}

void test_logistic_regression_logits_returns_expected_shape() {
    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    const ml::Vector logits = model.logits(X);

    if (logits.size() != y.size()) {
        throw std::runtime_error("expected logits.size() == y.size()");
    }
}

void test_logistic_regression_predict_proba_returns_probabilities() {
    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    const ml::Vector probabilities = model.predict_proba(X);

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        if (probabilities(i) < 0.0 || probabilities(i) > 1.0) {
            throw std::runtime_error("expected probabilities to be in [0, 1]");
        }
    }
}

void test_logistic_regression_predict_classes_returns_expected_shape() {
    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    const ml::Vector classes = model.predict_classes(X, 0.5);

    if (classes.size() != y.size()) {
        throw std::runtime_error("expected classes.size() == y.size()");
    }

    for (Eigen::Index i = 0; i < classes.size(); ++i) {
        if (classes(i) != 0.0 && classes(i) != 1.0) {
            throw std::runtime_error("expected predicted classes to be binary");
        }
    }
}

void test_logistic_regression_predict_classes_rejects_invalid_threshold() {
    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    static_cast<void>(model.predict_classes(X, -0.1));
}

void test_logistic_regression_binary_cross_entropy_is_small_after_fit() {
    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    const double loss = model.binary_cross_entropy(X, y);

    if (loss > 0.2) {
        throw std::runtime_error(
            "expected binary cross-entropy to be small, got " + std::to_string(loss)
        );
    }
}

void test_logistic_regression_stores_loss_history() {
    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    const ml::LogisticRegressionTrainingHistory& history =
        model.training_history();

    if (history.losses.empty()) {
        throw std::runtime_error("expected non-empty loss history");
    }

    if (history.iterations_run == 0) {
        throw std::runtime_error("expected iterations_run > 0");
    }
}

void test_logistic_regression_loss_decreases() {
    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    const ml::LogisticRegressionTrainingHistory& history =
        model.training_history();

    if (history.losses.size() < 2) {
        throw std::runtime_error("expected at least two recorded losses");
    }

    if (history.losses.back() >= history.losses.front()) {
        throw std::runtime_error("expected final loss to be lower than initial loss");
    }
}

void test_logistic_regression_rejects_invalid_learning_rate() {
    ml::LogisticRegressionOptions options = make_stable_logistic_options();
    options.learning_rate = 0.0;

    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(options);
    model.fit(X, y);
}

void test_logistic_regression_rejects_lasso_for_now() {
    ml::LogisticRegressionOptions options = make_stable_logistic_options();
    options.regularization = ml::RegularizationConfig::lasso(0.1);

    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(options);
    model.fit(X, y);
}

void test_logistic_regression_supports_ridge_lambda_positive() {
    ml::LogisticRegressionOptions options = make_stable_logistic_options();
    options.regularization = ml::RegularizationConfig::ridge(0.01);

    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(options);
    model.fit(X, y);

    if (!model.is_fitted()) {
        throw std::runtime_error("expected Ridge LogisticRegression to be fitted");
    }

    const double loss = model.binary_cross_entropy(X, y);

    if (loss > 0.5) {
        throw std::runtime_error(
            "expected Ridge LogisticRegression to fit simple data reasonably, got loss " +
            std::to_string(loss)
        );
    }
}

void test_logistic_regression_classification_accuracy_is_high_after_fit() {
    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    const ml::Vector predicted_classes = model.predict_classes(X, 0.5);

    const double accuracy = ml::accuracy_score(predicted_classes, y);

    if (accuracy < 1.0) {
        throw std::runtime_error(
            "expected perfect accuracy on simple separable data, got " +
            std::to_string(accuracy)
        );
    }
}

// ---- Logistic regression experiment export tests ----

void test_experiment_threshold_variation_exports_summaries() {
    ensure_phase4_output_dir_exists();

    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    const ml::BinaryClassificationEvaluationReport report_03 =
        evaluate_model_at_threshold(model, X, y, 0.3, "LogisticRegressionThreshold0.3");
    const ml::BinaryClassificationEvaluationReport report_05 =
        evaluate_model_at_threshold(model, X, y, 0.5, "LogisticRegressionThreshold0.5");
    const ml::BinaryClassificationEvaluationReport report_07 =
        evaluate_model_at_threshold(model, X, y, 0.7, "LogisticRegressionThreshold0.7");

    export_classification_report(
        report_03,
        "phase4_threshold_variation_0.3",
        k_phase4_output_dir + "/threshold_0_3_summary.csv",
        k_phase4_output_dir + "/threshold_0_3_summary.txt"
    );

    export_classification_report(
        report_05,
        "phase4_threshold_variation_0.5",
        k_phase4_output_dir + "/threshold_0_5_summary.csv",
        k_phase4_output_dir + "/threshold_0_5_summary.txt"
    );

    export_classification_report(
        report_07,
        "phase4_threshold_variation_0.7",
        k_phase4_output_dir + "/threshold_0_7_summary.csv",
        k_phase4_output_dir + "/threshold_0_7_summary.txt"
    );

    if (!std::filesystem::exists(k_phase4_output_dir + "/threshold_0_3_summary.csv")) {
        throw std::runtime_error("expected threshold_0_3_summary.csv to exist");
    }

    if (!std::filesystem::exists(k_phase4_output_dir + "/threshold_0_5_summary.csv")) {
        throw std::runtime_error("expected threshold_0_5_summary.csv to exist");
    }

    if (!std::filesystem::exists(k_phase4_output_dir + "/threshold_0_7_summary.csv")) {
        throw std::runtime_error("expected threshold_0_7_summary.csv to exist");
    }
}

void test_experiment_regularization_variation_exports_summaries() {
    ensure_phase4_output_dir_exists();

    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegressionOptions unregularized_options = make_stable_logistic_options();
    unregularized_options.regularization = ml::RegularizationConfig::none();

    ml::LogisticRegression unregularized_model(unregularized_options);
    unregularized_model.fit(X, y);

    const ml::BinaryClassificationEvaluationReport unregularized_report =
        evaluate_model_at_threshold(
            unregularized_model,
            X,
            y,
            0.5,
            "LogisticRegressionUnregularized"
        );

    export_classification_report(
        unregularized_report,
        "phase4_unregularized_logistic_regression",
        k_phase4_output_dir + "/unregularized_logistic_regression_summary.csv",
        k_phase4_output_dir + "/unregularized_logistic_regression_summary.txt"
    );

    ml::LogisticRegressionOptions ridge_options = make_stable_logistic_options();
    ridge_options.regularization = ml::RegularizationConfig::ridge(0.01);

    ml::LogisticRegression ridge_model(ridge_options);
    ridge_model.fit(X, y);

    const ml::BinaryClassificationEvaluationReport ridge_report =
        evaluate_model_at_threshold(
            ridge_model,
            X,
            y,
            0.5,
            "LogisticRegressionRidge"
        );

    export_classification_report(
        ridge_report,
        "phase4_ridge_logistic_regression",
        k_phase4_output_dir + "/ridge_logistic_regression_summary.csv",
        k_phase4_output_dir + "/ridge_logistic_regression_summary.txt"
    );

    if (!std::filesystem::exists(k_phase4_output_dir + "/unregularized_logistic_regression_summary.txt")) {
        throw std::runtime_error("expected unregularized logistic TXT summary to exist");
    }

    if (!std::filesystem::exists(k_phase4_output_dir + "/ridge_logistic_regression_summary.txt")) {
        throw std::runtime_error("expected ridge logistic TXT summary to exist");
    }
}

void test_experiment_class_boundary_interpretation_exports_files() {
    ensure_phase4_output_dir_exists();

    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    export_decision_boundary_interpretation(
        model,
        k_phase4_output_dir + "/decision_boundary_interpretation.csv",
        k_phase4_output_dir + "/decision_boundary_interpretation.txt"
    );

    if (!std::filesystem::exists(k_phase4_output_dir + "/decision_boundary_interpretation.csv")) {
        throw std::runtime_error("expected decision_boundary_interpretation.csv to exist");
    }

    if (!std::filesystem::exists(k_phase4_output_dir + "/decision_boundary_interpretation.txt")) {
        throw std::runtime_error("expected decision_boundary_interpretation.txt to exist");
    }
}

void test_experiment_multiple_metrics_exports_summary() {
    ensure_phase4_output_dir_exists();

    const ml::Matrix X = make_binary_classification_X();
    const ml::Vector y = make_binary_classification_y();

    ml::LogisticRegression model(make_stable_logistic_options());
    model.fit(X, y);

    const ml::BinaryClassificationEvaluationReport report =
        evaluate_model_at_threshold(
            model,
            X,
            y,
            0.5,
            "LogisticRegressionMultipleMetrics"
        );

    export_classification_report(
        report,
        "phase4_multiple_classification_metrics",
        k_phase4_output_dir + "/multiple_metrics_summary.csv",
        k_phase4_output_dir + "/multiple_metrics_summary.txt"
    );

    if (!std::filesystem::exists(k_phase4_output_dir + "/multiple_metrics_summary.csv")) {
        throw std::runtime_error("expected multiple_metrics_summary.csv to exist");
    }

    if (!std::filesystem::exists(k_phase4_output_dir + "/multiple_metrics_summary.txt")) {
        throw std::runtime_error("expected multiple_metrics_summary.txt to exist");
    }
}

void run_classification_utils_tests() {
    std::cout << "\n[Phase 4.1] Classification utility tests\n\n";

    test::expect_no_throw(
        "sigmoid scalar computes expected values",
        test_sigmoid_scalar_computes_expected_values
    );

    test::expect_no_throw(
        "sigmoid vector computes expected values",
        test_sigmoid_vector_computes_expected_values
    );

    test::expect_invalid_argument(
        "validate_binary_targets rejects invalid values",
        test_validate_binary_targets_rejects_invalid_values
    );

    test::expect_no_throw(
        "threshold_probabilities computes expected classes",
        test_threshold_probabilities_computes_expected_classes
    );

    test::expect_invalid_argument(
        "threshold_probabilities rejects invalid threshold",
        test_threshold_probabilities_rejects_invalid_threshold
    );

    test::expect_no_throw(
        "binary_cross_entropy computes expected value",
        test_binary_cross_entropy_computes_expected_value
    );

    test::expect_invalid_argument(
        "binary_cross_entropy rejects invalid targets",
        test_binary_cross_entropy_rejects_invalid_targets
    );
}

void run_classification_metrics_tests() {
    std::cout << "\n[Phase 4.2] Classification metric tests\n\n";

    test::expect_no_throw(
        "confusion_matrix computes expected counts",
        test_confusion_matrix_computes_expected_counts
    );

    test::expect_no_throw(
        "classification metrics compute expected values",
        test_classification_metrics_compute_expected_values
    );

    test::expect_invalid_argument(
        "classification metrics reject invalid predictions",
        test_classification_metrics_reject_invalid_predictions
    );
}

void run_classification_evaluation_tests() {
    std::cout << "\n[Phase 4.3] Classification evaluation tests\n\n";

    test::expect_no_throw(
        "evaluate_binary_classification returns expected metrics",
        test_evaluate_binary_classification_returns_expected_metrics
    );

    test::expect_invalid_argument(
        "evaluate_binary_classification rejects invalid probabilities",
        test_evaluate_binary_classification_rejects_invalid_probabilities
    );

    test::expect_invalid_argument(
        "evaluate_binary_classification rejects mismatched vectors",
        test_evaluate_binary_classification_rejects_mismatched_vectors
    );

    test::expect_no_throw(
        "run_binary_classification_evaluation returns report",
        test_run_binary_classification_evaluation_returns_report
    );

    test::expect_no_throw(
        "run_binary_classification_evaluation detects perfect accuracy",
        test_run_binary_classification_evaluation_detects_perfect_accuracy
    );

    test::expect_invalid_argument(
        "run_binary_classification_evaluation rejects empty model name",
        test_run_binary_classification_evaluation_rejects_empty_model_name
    );

    test::expect_invalid_argument(
        "run_binary_classification_evaluation rejects invalid threshold",
        test_run_binary_classification_evaluation_rejects_invalid_threshold
    );
}

void run_logistic_regression_tests() {
    std::cout << "\n[Phase 4.4] Logistic regression implementation tests\n\n";

    test::expect_invalid_argument(
        "LogisticRegression rejects empty X/y",
        test_logistic_regression_rejects_empty_X_y
    );

    test::expect_invalid_argument(
        "LogisticRegression rejects mismatched X/y",
        test_logistic_regression_rejects_mismatched_X_y
    );

    test::expect_invalid_argument(
        "LogisticRegression rejects non-binary targets",
        test_logistic_regression_rejects_non_binary_targets
    );

    test::expect_invalid_argument(
        "LogisticRegression rejects predict before fit",
        test_logistic_regression_rejects_predict_before_fit
    );

    test::expect_no_throw(
        "LogisticRegression fits simple binary data",
        test_logistic_regression_fits_simple_binary_data
    );

    test::expect_no_throw(
        "LogisticRegression logits returns expected shape",
        test_logistic_regression_logits_returns_expected_shape
    );

    test::expect_no_throw(
        "LogisticRegression predict_proba returns probabilities",
        test_logistic_regression_predict_proba_returns_probabilities
    );

    test::expect_no_throw(
        "LogisticRegression predict_classes returns expected shape",
        test_logistic_regression_predict_classes_returns_expected_shape
    );

    test::expect_invalid_argument(
        "LogisticRegression predict_classes rejects invalid threshold",
        test_logistic_regression_predict_classes_rejects_invalid_threshold
    );

    test::expect_no_throw(
        "LogisticRegression binary_cross_entropy is small after fit",
        test_logistic_regression_binary_cross_entropy_is_small_after_fit
    );

    test::expect_no_throw(
        "LogisticRegression stores loss history",
        test_logistic_regression_stores_loss_history
    );

    test::expect_no_throw(
        "LogisticRegression loss decreases",
        test_logistic_regression_loss_decreases
    );

    test::expect_invalid_argument(
        "LogisticRegression rejects invalid learning rate",
        test_logistic_regression_rejects_invalid_learning_rate
    );

    test::expect_invalid_argument(
        "LogisticRegression rejects Lasso fitting for now",
        test_logistic_regression_rejects_lasso_for_now
    );

    test::expect_no_throw(
        "LogisticRegression supports Ridge lambda > 0",
        test_logistic_regression_supports_ridge_lambda_positive
    );

    test::expect_no_throw(
        "LogisticRegression classification accuracy is high after fit",
        test_logistic_regression_classification_accuracy_is_high_after_fit
    );
}

void run_logistic_regression_experiment_export_tests() {
    std::cout << "\n[Phase 4.5] Logistic regression experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports threshold variation summaries",
        test_experiment_threshold_variation_exports_summaries
    );

    test::expect_no_throw(
        "Experiment exports regularization variation summaries",
        test_experiment_regularization_variation_exports_summaries
    );

    test::expect_no_throw(
        "Experiment exports class-boundary interpretation files",
        test_experiment_class_boundary_interpretation_exports_files
    );

    test::expect_no_throw(
        "Experiment exports multiple-metrics summary",
        test_experiment_multiple_metrics_exports_summary
    );
}

}  // namespace

namespace ml::experiments {

void run_phase4_logistic_regression_sanity() {
    run_classification_utils_tests();
    run_classification_metrics_tests();
    run_classification_evaluation_tests();
    run_logistic_regression_tests();
    run_logistic_regression_experiment_export_tests();
}

}  // namespace ml::experiments