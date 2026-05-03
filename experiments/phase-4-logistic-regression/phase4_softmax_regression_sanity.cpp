#include "phase4_softmax_regression_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/classification_utils.hpp"
#include "ml/common/multiclass_metrics.hpp"
#include "ml/common/types.hpp"
#include "ml/linear_models/regularization.hpp"
#include "ml/linear_models/softmax_regression.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace test = ml::experiments::test;

namespace {

ml::Matrix make_multiclass_classification_X() {
    ml::Matrix X(9, 2);
    X << 0.0, 0.0,
         0.0, 1.0,
         1.0, 0.0,
         4.0, 4.0,
         4.0, 5.0,
         5.0, 4.0,
        -4.0, 4.0,
        -4.0, 5.0,
        -5.0, 4.0;

    return X;
}

ml::Vector make_multiclass_classification_y() {
    ml::Vector y(9);
    y << 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0,
         2.0, 2.0, 2.0;

    return y;
}

ml::SoftmaxRegressionOptions make_stable_softmax_options() {
    ml::SoftmaxRegressionOptions options;
    options.learning_rate = 0.1;
    options.max_iterations = 20000;
    options.tolerance = 1e-12;
    options.store_loss_history = true;
    options.regularization = ml::RegularizationConfig::none();

    return options;
}

const std::string k_phase4_output_dir = "outputs/phase-4-logistic-regression";

void ensure_phase4_output_dir_exists() {
    std::filesystem::create_directories(k_phase4_output_dir);
}

double multiclass_accuracy_from_predictions(
    const ml::Vector& predictions,
    const ml::Vector& targets
) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error(
            "multiclass_accuracy_from_predictions: predictions and targets must have the same size"
        );
    }

    Eigen::Index correct = 0;

    for (Eigen::Index i = 0; i < targets.size(); ++i) {
        if (predictions(i) == targets(i)) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(targets.size());
}

void export_softmax_multiclass_summary(
    const std::string& experiment_name,
    const std::string& model_name,
    const ml::Vector& predictions,
    const ml::Vector& targets,
    Eigen::Index num_classes,
    double loss,
    const std::string& csv_output_path,
    const std::string& txt_output_path
) {
    std::ofstream csv_file(csv_output_path);

    if (!csv_file.is_open()) {
        throw std::runtime_error(
            "export_softmax_multiclass_summary: failed to open CSV output file"
        );
    }

    std::ofstream txt_file(txt_output_path);

    if (!txt_file.is_open()) {
        throw std::runtime_error(
            "export_softmax_multiclass_summary: failed to open TXT output file"
        );
    }

    const double accuracy = ml::multiclass_accuracy_score(
        predictions,
        targets,
        num_classes
    );
    const double precision = ml::macro_precision(predictions, targets, num_classes);
    const double recall = ml::macro_recall(predictions, targets, num_classes);
    const double f1 = ml::macro_f1(predictions, targets, num_classes);
    const ml::Matrix confusion = ml::multiclass_confusion_matrix(
        predictions,
        targets,
        num_classes
    );

    csv_file << "experiment_name,model_name,num_classes,loss,accuracy,macro_precision,macro_recall,macro_f1\n";
    csv_file << experiment_name << ","
             << model_name << ","
             << num_classes << ","
             << loss << ","
             << accuracy << ","
             << precision << ","
             << recall << ","
             << f1 << "\n";

    txt_file << "Experiment: " << experiment_name << "\n"
             << "Model: " << model_name << "\n"
             << "Number of classes: " << num_classes << "\n\n";

    txt_file << "Multiclass metrics:\n"
             << "  Loss: " << loss << "\n"
             << "  Accuracy: " << accuracy << "\n"
             << "  Macro precision: " << precision << "\n"
             << "  Macro recall: " << recall << "\n"
             << "  Macro F1: " << f1 << "\n\n";

    txt_file << "Confusion matrix rows=true classes, columns=predicted classes:\n";

    for (Eigen::Index i = 0; i < confusion.rows(); ++i) {
        txt_file << "  ";

        for (Eigen::Index j = 0; j < confusion.cols(); ++j) {
            txt_file << confusion(i, j);

            if (j + 1 < confusion.cols()) {
                txt_file << " ";
            }
        }

        txt_file << "\n";
    }

    txt_file << "\nInterpretation:\n";

    if (accuracy == 1.0) {
        txt_file << "  The softmax classifier perfectly separates this synthetic multiclass dataset.\n";
    } else {
        txt_file << "  The softmax classifier makes at least one multiclass prediction error.\n";
    }
}

void export_softmax_prediction_table(
    const ml::Matrix& probabilities,
    const ml::Vector& predictions,
    const ml::Vector& targets,
    const std::string& csv_output_path,
    const std::string& txt_output_path
) {
    std::ofstream csv_file(csv_output_path);

    if (!csv_file.is_open()) {
        throw std::runtime_error(
            "export_softmax_prediction_table: failed to open CSV output file"
        );
    }

    std::ofstream txt_file(txt_output_path);

    if (!txt_file.is_open()) {
        throw std::runtime_error(
            "export_softmax_prediction_table: failed to open TXT output file"
        );
    }

    csv_file << "sample_index,target,predicted_class";

    for (Eigen::Index class_index = 0; class_index < probabilities.cols(); ++class_index) {
        csv_file << ",probability_class_" << class_index;
    }

    csv_file << "\n";

    txt_file << "Softmax Prediction Table\n\n";

    for (Eigen::Index i = 0; i < targets.size(); ++i) {
        csv_file << i << "," << targets(i) << "," << predictions(i);

        for (Eigen::Index class_index = 0; class_index < probabilities.cols(); ++class_index) {
            csv_file << "," << probabilities(i, class_index);
        }

        csv_file << "\n";

        txt_file << "Sample " << i
                 << ": target=" << targets(i)
                 << ", predicted=" << predictions(i)
                 << ", probabilities=[";

        for (Eigen::Index class_index = 0; class_index < probabilities.cols(); ++class_index) {
            txt_file << probabilities(i, class_index);

            if (class_index + 1 < probabilities.cols()) {
                txt_file << ", ";
            }
        }

        txt_file << "]\n";
    }
}

// ---- Multiclass utility tests ----

void test_softmax_rows_produces_row_probabilities() {
    ml::Matrix logits(2, 3);
    logits << 1.0, 2.0, 3.0,
              0.0, 0.0, 0.0;

    const ml::Matrix probabilities = ml::softmax_rows(logits);

    if (probabilities.rows() != logits.rows()) {
        throw std::runtime_error("expected softmax probabilities to preserve row count");
    }

    if (probabilities.cols() != logits.cols()) {
        throw std::runtime_error("expected softmax probabilities to preserve column count");
    }

    for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
        test::assert_almost_equal(
            probabilities.row(i).sum(),
            1.0,
            "test_softmax_rows_produces_row_probabilities row sum"
        );

        for (Eigen::Index j = 0; j < probabilities.cols(); ++j) {
            if (probabilities(i, j) < 0.0 || probabilities(i, j) > 1.0) {
                throw std::runtime_error("expected softmax probabilities in [0, 1]");
            }
        }
    }
}

void test_softmax_rows_handles_large_logits_stably() {
    ml::Matrix logits(1, 3);
    logits << 1000.0, 1001.0, 1002.0;

    const ml::Matrix probabilities = ml::softmax_rows(logits);

    test::assert_almost_equal(
        probabilities.row(0).sum(),
        1.0,
        "test_softmax_rows_handles_large_logits_stably row sum"
    );

    for (Eigen::Index j = 0; j < probabilities.cols(); ++j) {
        if (!std::isfinite(probabilities(0, j))) {
            throw std::runtime_error("expected finite softmax probability");
        }
    }
}

void test_validate_class_indices_accepts_valid_values() {
    ml::Vector targets(4);
    targets << 0.0, 1.0, 2.0, 1.0;

    ml::validate_class_indices(
        targets,
        3,
        "test_validate_class_indices_accepts_valid_values"
    );
}

void test_validate_class_indices_rejects_out_of_range_values() {
    ml::Vector targets(3);
    targets << 0.0, 1.0, 3.0;

    ml::validate_class_indices(
        targets,
        3,
        "test_validate_class_indices_rejects_out_of_range_values"
    );
}

void test_validate_class_indices_rejects_non_integer_values() {
    ml::Vector targets(3);
    targets << 0.0, 1.5, 2.0;

    ml::validate_class_indices(
        targets,
        3,
        "test_validate_class_indices_rejects_non_integer_values"
    );
}

void test_one_hot_encode_creates_expected_matrix() {
    ml::Vector targets(4);
    targets << 0.0, 2.0, 1.0, 2.0;

    const ml::Matrix encoded = ml::one_hot_encode(targets, 3);

    ml::Matrix expected(4, 3);
    expected << 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0;

    test::assert_matrix_almost_equal(
        encoded,
        expected,
        "test_one_hot_encode_creates_expected_matrix"
    );
}

void test_argmax_rows_returns_expected_classes() {
    ml::Matrix probabilities(3, 3);
    probabilities << 0.7, 0.2, 0.1,
                     0.1, 0.8, 0.1,
                     0.2, 0.3, 0.5;

    const ml::Vector classes = ml::argmax_rows(probabilities);

    ml::Vector expected(3);
    expected << 0.0, 1.0, 2.0;

    test::assert_vector_almost_equal(
        classes,
        expected,
        "test_argmax_rows_returns_expected_classes"
    );
}

void test_argmax_rows_rejects_rows_that_do_not_sum_to_one() {
    ml::Matrix probabilities(2, 3);
    probabilities << 0.7, 0.2, 0.1,
                     0.2, 0.2, 0.2;

    static_cast<void>(ml::argmax_rows(probabilities));
}

void test_categorical_cross_entropy_computes_expected_value() {
    ml::Matrix probabilities(2, 3);
    probabilities << 0.8, 0.1, 0.1,
                     0.2, 0.3, 0.5;

    ml::Matrix targets(2, 3);
    targets << 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0;

    const double result = ml::categorical_cross_entropy(probabilities, targets);
    const double expected = -0.5 * (std::log(0.8) + std::log(0.5));

    test::assert_almost_equal(
        result,
        expected,
        "test_categorical_cross_entropy_computes_expected_value"
    );
}

void test_categorical_cross_entropy_rejects_invalid_one_hot_targets() {
    ml::Matrix probabilities(2, 3);
    probabilities << 0.8, 0.1, 0.1,
                     0.2, 0.3, 0.5;

    ml::Matrix targets(2, 3);
    targets << 1.0, 0.0, 0.0,
               0.0, 0.5, 0.5;

    static_cast<void>(ml::categorical_cross_entropy(probabilities, targets));
}

void test_categorical_cross_entropy_rejects_mismatched_shapes() {
    ml::Matrix probabilities(2, 3);
    probabilities << 0.8, 0.1, 0.1,
                     0.2, 0.3, 0.5;

    ml::Matrix targets(2, 2);
    targets << 1.0, 0.0,
               0.0, 1.0;

    static_cast<void>(ml::categorical_cross_entropy(probabilities, targets));
}

// ---- Multiclass metric tests ----

void test_multiclass_confusion_matrix_computes_expected_counts() {
    ml::Vector predictions(6);
    predictions << 0.0, 1.0, 2.0, 1.0, 2.0, 0.0;

    ml::Vector targets(6);
    targets << 0.0, 1.0, 1.0, 1.0, 2.0, 2.0;

    const ml::Matrix matrix = ml::multiclass_confusion_matrix(
        predictions,
        targets,
        3
    );

    ml::Matrix expected = ml::Matrix::Zero(3, 3);
    expected << 1.0, 0.0, 0.0,
                0.0, 2.0, 1.0,
                1.0, 0.0, 1.0;

    test::assert_matrix_almost_equal(
        matrix,
        expected,
        "test_multiclass_confusion_matrix_computes_expected_counts"
    );
}

void test_multiclass_accuracy_score_computes_expected_value() {
    ml::Vector predictions(6);
    predictions << 0.0, 1.0, 2.0, 1.0, 2.0, 0.0;

    ml::Vector targets(6);
    targets << 0.0, 1.0, 1.0, 1.0, 2.0, 2.0;

    const double accuracy = ml::multiclass_accuracy_score(
        predictions,
        targets,
        3
    );

    test::assert_almost_equal(
        accuracy,
        4.0 / 6.0,
        "test_multiclass_accuracy_score_computes_expected_value"
    );
}

void test_macro_precision_computes_expected_value() {
    ml::Vector predictions(6);
    predictions << 0.0, 1.0, 2.0, 1.0, 2.0, 0.0;

    ml::Vector targets(6);
    targets << 0.0, 1.0, 1.0, 1.0, 2.0, 2.0;

    const double precision = ml::macro_precision(
        predictions,
        targets,
        3
    );

    const double expected = (0.5 + 1.0 + 0.5) / 3.0;

    test::assert_almost_equal(
        precision,
        expected,
        "test_macro_precision_computes_expected_value"
    );
}

void test_macro_recall_computes_expected_value() {
    ml::Vector predictions(6);
    predictions << 0.0, 1.0, 2.0, 1.0, 2.0, 0.0;

    ml::Vector targets(6);
    targets << 0.0, 1.0, 1.0, 1.0, 2.0, 2.0;

    const double recall = ml::macro_recall(
        predictions,
        targets,
        3
    );

    const double expected = (1.0 + (2.0 / 3.0) + 0.5) / 3.0;

    test::assert_almost_equal(
        recall,
        expected,
        "test_macro_recall_computes_expected_value"
    );
}

void test_macro_f1_computes_expected_value() {
    ml::Vector predictions(6);
    predictions << 0.0, 1.0, 2.0, 1.0, 2.0, 0.0;

    ml::Vector targets(6);
    targets << 0.0, 1.0, 1.0, 1.0, 2.0, 2.0;

    const double f1 = ml::macro_f1(
        predictions,
        targets,
        3
    );

    const double class_0_f1 = 2.0 * 0.5 * 1.0 / (0.5 + 1.0);
    const double class_1_f1 = 2.0 * 1.0 * (2.0 / 3.0) / (1.0 + (2.0 / 3.0));
    const double class_2_f1 = 2.0 * 0.5 * 0.5 / (0.5 + 0.5);
    const double expected = (class_0_f1 + class_1_f1 + class_2_f1) / 3.0;

    test::assert_almost_equal(
        f1,
        expected,
        "test_macro_f1_computes_expected_value"
    );
}

void test_multiclass_metrics_reject_invalid_predictions() {
    ml::Vector predictions(3);
    predictions << 0.0, 1.0, 3.0;

    ml::Vector targets(3);
    targets << 0.0, 1.0, 2.0;

    static_cast<void>(ml::multiclass_accuracy_score(
        predictions,
        targets,
        3
    ));
}

void test_multiclass_metrics_reject_mismatched_vectors() {
    ml::Vector predictions(2);
    predictions << 0.0, 1.0;

    ml::Vector targets(3);
    targets << 0.0, 1.0, 2.0;

    static_cast<void>(ml::multiclass_accuracy_score(
        predictions,
        targets,
        3
    ));
}

// ---- Softmax regression model tests ----

void test_softmax_regression_rejects_empty_X_y() {
    ml::Matrix X(0, 0);
    ml::Vector y(0);

    ml::SoftmaxRegression model;
    model.fit(X, y, 3);
}

void test_softmax_regression_rejects_mismatched_X_y() {
    ml::Matrix X(3, 2);
    X << 0.0, 0.0,
         1.0, 1.0,
         2.0, 2.0;

    ml::Vector y(2);
    y << 0.0, 1.0;

    ml::SoftmaxRegression model;
    model.fit(X, y, 3);
}

void test_softmax_regression_rejects_invalid_class_targets() {
    const ml::Matrix X = make_multiclass_classification_X();

    ml::Vector y(9);
    y << 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0,
         2.0, 2.0, 3.0;

    ml::SoftmaxRegression model;
    model.fit(X, y, 3);
}

void test_softmax_regression_rejects_invalid_num_classes() {
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model;
    model.fit(X, y, 1);
}

void test_softmax_regression_rejects_predict_before_fit() {
    const ml::Matrix X = make_multiclass_classification_X();

    ml::SoftmaxRegression model;
    static_cast<void>(model.predict_proba(X));
}

void test_softmax_regression_fits_simple_multiclass_data() {
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    if (!model.is_fitted()) {
        throw std::runtime_error("expected SoftmaxRegression to be fitted");
    }

    const double loss = model.categorical_cross_entropy(X, y);

    if (loss > 0.2) {
        throw std::runtime_error(
            "expected small categorical cross-entropy, got " + std::to_string(loss)
        );
    }
}

void test_softmax_regression_logits_returns_expected_shape() {
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    const ml::Matrix logits = model.logits(X);

    if (logits.rows() != X.rows()) {
        throw std::runtime_error("expected logits.rows() == X.rows()");
    }

    if (logits.cols() != 3) {
        throw std::runtime_error("expected logits.cols() == num_classes");
    }
}

void test_softmax_regression_predict_proba_returns_probabilities() {
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    const ml::Matrix probabilities = model.predict_proba(X);

    if (probabilities.rows() != X.rows()) {
        throw std::runtime_error("expected probabilities.rows() == X.rows()");
    }

    if (probabilities.cols() != 3) {
        throw std::runtime_error("expected probabilities.cols() == num_classes");
    }

    for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
        test::assert_almost_equal(
            probabilities.row(i).sum(),
            1.0,
            "test_softmax_regression_predict_proba_returns_probabilities row sum"
        );

        for (Eigen::Index j = 0; j < probabilities.cols(); ++j) {
            if (probabilities(i, j) < 0.0 || probabilities(i, j) > 1.0) {
                throw std::runtime_error("expected softmax probabilities in [0, 1]");
            }
        }
    }
}

void test_softmax_regression_predict_classes_returns_expected_shape() {
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    const ml::Vector classes = model.predict_classes(X);

    if (classes.size() != y.size()) {
        throw std::runtime_error("expected classes.size() == y.size()");
    }

    ml::validate_class_indices(
        classes,
        3,
        "test_softmax_regression_predict_classes_returns_expected_shape"
    );
}

void test_softmax_regression_categorical_cross_entropy_is_small_after_fit() {
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    const double loss = model.categorical_cross_entropy(X, y);

    if (loss > 0.2) {
        throw std::runtime_error(
            "expected categorical cross-entropy to be small, got " + std::to_string(loss)
        );
    }
}

void test_softmax_regression_stores_loss_history() {
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    const ml::SoftmaxRegressionTrainingHistory& history =
        model.training_history();

    if (history.losses.empty()) {
        throw std::runtime_error("expected non-empty softmax loss history");
    }

    if (history.iterations_run == 0) {
        throw std::runtime_error("expected softmax iterations_run > 0");
    }
}

void test_softmax_regression_loss_decreases() {
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    const ml::SoftmaxRegressionTrainingHistory& history =
        model.training_history();

    if (history.losses.size() < 2) {
        throw std::runtime_error("expected at least two softmax recorded losses");
    }

    if (history.losses.back() >= history.losses.front()) {
        throw std::runtime_error("expected final softmax loss to be lower than initial loss");
    }
}

void test_softmax_regression_rejects_invalid_learning_rate() {
    ml::SoftmaxRegressionOptions options = make_stable_softmax_options();
    options.learning_rate = 0.0;

    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(options);
    model.fit(X, y, 3);
}

void test_softmax_regression_rejects_lasso_for_now() {
    ml::SoftmaxRegressionOptions options = make_stable_softmax_options();
    options.regularization = ml::RegularizationConfig::lasso(0.1);

    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(options);
    model.fit(X, y, 3);
}

void test_softmax_regression_supports_ridge_lambda_positive() {
    ml::SoftmaxRegressionOptions options = make_stable_softmax_options();
    options.regularization = ml::RegularizationConfig::ridge(0.01);

    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(options);
    model.fit(X, y, 3);

    if (!model.is_fitted()) {
        throw std::runtime_error("expected Ridge SoftmaxRegression to be fitted");
    }

    const double loss = model.categorical_cross_entropy(X, y);

    if (loss > 0.5) {
        throw std::runtime_error(
            "expected Ridge SoftmaxRegression to fit simple data reasonably, got loss " +
            std::to_string(loss)
        );
    }
}

void test_softmax_regression_classification_accuracy_is_high_after_fit() {
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    const ml::Vector classes = model.predict_classes(X);

    Eigen::Index correct = 0;

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (classes(i) == y(i)) {
            ++correct;
        }
    }

    const double accuracy =
        static_cast<double>(correct) / static_cast<double>(y.size());

    if (accuracy < 1.0) {
        throw std::runtime_error(
            "expected perfect multiclass accuracy on simple separable data, got " +
            std::to_string(accuracy)
        );
    }
}

// ---- Softmax regression experiment export tests ----

void test_experiment_softmax_multiclass_summary_exports_files() {
    ensure_phase4_output_dir_exists();

    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    const ml::Vector predictions = model.predict_classes(X);
    const double loss = model.categorical_cross_entropy(X, y);

    export_softmax_multiclass_summary(
        "phase4_softmax_multiclass_summary",
        "SoftmaxRegression",
        predictions,
        y,
        3,
        loss,
        k_phase4_output_dir + "/softmax_multiclass_summary.csv",
        k_phase4_output_dir + "/softmax_multiclass_summary.txt"
    );

    if (!std::filesystem::exists(k_phase4_output_dir + "/softmax_multiclass_summary.csv")) {
        throw std::runtime_error("expected softmax_multiclass_summary.csv to exist");
    }

    if (!std::filesystem::exists(k_phase4_output_dir + "/softmax_multiclass_summary.txt")) {
        throw std::runtime_error("expected softmax_multiclass_summary.txt to exist");
    }
}

void test_experiment_softmax_regularization_comparison_exports_files() {
    ensure_phase4_output_dir_exists();

    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegressionOptions unregularized_options = make_stable_softmax_options();
    unregularized_options.regularization = ml::RegularizationConfig::none();

    ml::SoftmaxRegression unregularized_model(unregularized_options);
    unregularized_model.fit(X, y, 3);

    const ml::Vector unregularized_predictions = unregularized_model.predict_classes(X);
    const double unregularized_loss = unregularized_model.categorical_cross_entropy(X, y);
    const double unregularized_accuracy = ml::multiclass_accuracy_score(
        unregularized_predictions,
        y,
        3
    );

    ml::SoftmaxRegressionOptions ridge_options = make_stable_softmax_options();
    ridge_options.regularization = ml::RegularizationConfig::ridge(0.01);

    ml::SoftmaxRegression ridge_model(ridge_options);
    ridge_model.fit(X, y, 3);

    const ml::Vector ridge_predictions = ridge_model.predict_classes(X);
    const double ridge_loss = ridge_model.categorical_cross_entropy(X, y);
    const double ridge_accuracy = ml::multiclass_accuracy_score(
        ridge_predictions,
        y,
        3
    );

    const std::string csv_output_path =
        k_phase4_output_dir + "/softmax_regularization_comparison.csv";
    const std::string txt_output_path =
        k_phase4_output_dir + "/softmax_regularization_comparison.txt";

    std::ofstream csv_file(csv_output_path);

    if (!csv_file.is_open()) {
        throw std::runtime_error(
            "test_experiment_softmax_regularization_comparison_exports_files: failed to open CSV output file"
        );
    }

    std::ofstream txt_file(txt_output_path);

    if (!txt_file.is_open()) {
        throw std::runtime_error(
            "test_experiment_softmax_regularization_comparison_exports_files: failed to open TXT output file"
        );
    }

    csv_file << "model_name,regularization,loss,accuracy,weight_squared_norm\n";
    csv_file << "SoftmaxRegression,None,"
             << unregularized_loss << ","
             << unregularized_accuracy << ","
             << unregularized_model.weights().squaredNorm() << "\n";
    csv_file << "SoftmaxRegression,Ridge,"
             << ridge_loss << ","
             << ridge_accuracy << ","
             << ridge_model.weights().squaredNorm() << "\n";

    txt_file << "Softmax Regularization Comparison\n\n"
             << "Unregularized:\n"
             << "  Loss: " << unregularized_loss << "\n"
             << "  Accuracy: " << unregularized_accuracy << "\n"
             << "  Weight squared norm: " << unregularized_model.weights().squaredNorm() << "\n\n"
             << "Ridge:\n"
             << "  Loss: " << ridge_loss << "\n"
             << "  Accuracy: " << ridge_accuracy << "\n"
             << "  Weight squared norm: " << ridge_model.weights().squaredNorm() << "\n\n"
             << "Interpretation:\n"
             << "  Ridge adds an L2 penalty to the softmax weight matrix.\n"
             << "  On this simple synthetic dataset, both models should classify the training samples correctly.\n";

    if (!std::filesystem::exists(csv_output_path)) {
        throw std::runtime_error("expected softmax_regularization_comparison.csv to exist");
    }

    if (!std::filesystem::exists(txt_output_path)) {
        throw std::runtime_error("expected softmax_regularization_comparison.txt to exist");
    }
}

void test_experiment_softmax_prediction_table_exports_files() {
    ensure_phase4_output_dir_exists();

    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Vector y = make_multiclass_classification_y();

    ml::SoftmaxRegression model(make_stable_softmax_options());
    model.fit(X, y, 3);

    const ml::Matrix probabilities = model.predict_proba(X);
    const ml::Vector predictions = model.predict_classes(X);

    export_softmax_prediction_table(
        probabilities,
        predictions,
        y,
        k_phase4_output_dir + "/softmax_prediction_table.csv",
        k_phase4_output_dir + "/softmax_prediction_table.txt"
    );

    if (!std::filesystem::exists(k_phase4_output_dir + "/softmax_prediction_table.csv")) {
        throw std::runtime_error("expected softmax_prediction_table.csv to exist");
    }

    if (!std::filesystem::exists(k_phase4_output_dir + "/softmax_prediction_table.txt")) {
        throw std::runtime_error("expected softmax_prediction_table.txt to exist");
    }
}

// ---- Test runners ----

void run_multiclass_utility_tests() {
    std::cout << "\n[Phase 4.6] Multiclass utility tests\n\n";

    test::expect_no_throw(
        "softmax_rows produces row probabilities",
        test_softmax_rows_produces_row_probabilities
    );

    test::expect_no_throw(
        "softmax_rows handles large logits stably",
        test_softmax_rows_handles_large_logits_stably
    );

    test::expect_no_throw(
        "validate_class_indices accepts valid values",
        test_validate_class_indices_accepts_valid_values
    );

    test::expect_invalid_argument(
        "validate_class_indices rejects out-of-range values",
        test_validate_class_indices_rejects_out_of_range_values
    );

    test::expect_invalid_argument(
        "validate_class_indices rejects non-integer values",
        test_validate_class_indices_rejects_non_integer_values
    );

    test::expect_no_throw(
        "one_hot_encode creates expected matrix",
        test_one_hot_encode_creates_expected_matrix
    );

    test::expect_no_throw(
        "argmax_rows returns expected classes",
        test_argmax_rows_returns_expected_classes
    );

    test::expect_invalid_argument(
        "argmax_rows rejects rows that do not sum to one",
        test_argmax_rows_rejects_rows_that_do_not_sum_to_one
    );

    test::expect_no_throw(
        "categorical_cross_entropy computes expected value",
        test_categorical_cross_entropy_computes_expected_value
    );

    test::expect_invalid_argument(
        "categorical_cross_entropy rejects invalid one-hot targets",
        test_categorical_cross_entropy_rejects_invalid_one_hot_targets
    );

    test::expect_invalid_argument(
        "categorical_cross_entropy rejects mismatched shapes",
        test_categorical_cross_entropy_rejects_mismatched_shapes
    );
}

void run_multiclass_metric_tests() {
    std::cout << "\n[Phase 4.7] Multiclass metric tests\n\n";

    test::expect_no_throw(
        "multiclass_confusion_matrix computes expected counts",
        test_multiclass_confusion_matrix_computes_expected_counts
    );

    test::expect_no_throw(
        "multiclass_accuracy_score computes expected value",
        test_multiclass_accuracy_score_computes_expected_value
    );

    test::expect_no_throw(
        "macro_precision computes expected value",
        test_macro_precision_computes_expected_value
    );

    test::expect_no_throw(
        "macro_recall computes expected value",
        test_macro_recall_computes_expected_value
    );

    test::expect_no_throw(
        "macro_f1 computes expected value",
        test_macro_f1_computes_expected_value
    );

    test::expect_invalid_argument(
        "multiclass metrics reject invalid predictions",
        test_multiclass_metrics_reject_invalid_predictions
    );

    test::expect_invalid_argument(
        "multiclass metrics reject mismatched vectors",
        test_multiclass_metrics_reject_mismatched_vectors
    );
}

void run_softmax_regression_tests() {
    std::cout << "\n[Phase 4.8] Softmax regression implementation tests\n\n";

    test::expect_invalid_argument(
        "SoftmaxRegression rejects empty X/y",
        test_softmax_regression_rejects_empty_X_y
    );

    test::expect_invalid_argument(
        "SoftmaxRegression rejects mismatched X/y",
        test_softmax_regression_rejects_mismatched_X_y
    );

    test::expect_invalid_argument(
        "SoftmaxRegression rejects invalid class targets",
        test_softmax_regression_rejects_invalid_class_targets
    );

    test::expect_invalid_argument(
        "SoftmaxRegression rejects invalid num_classes",
        test_softmax_regression_rejects_invalid_num_classes
    );

    test::expect_invalid_argument(
        "SoftmaxRegression rejects predict before fit",
        test_softmax_regression_rejects_predict_before_fit
    );

    test::expect_no_throw(
        "SoftmaxRegression fits simple multiclass data",
        test_softmax_regression_fits_simple_multiclass_data
    );

    test::expect_no_throw(
        "SoftmaxRegression logits returns expected shape",
        test_softmax_regression_logits_returns_expected_shape
    );

    test::expect_no_throw(
        "SoftmaxRegression predict_proba returns probabilities",
        test_softmax_regression_predict_proba_returns_probabilities
    );

    test::expect_no_throw(
        "SoftmaxRegression predict_classes returns expected shape",
        test_softmax_regression_predict_classes_returns_expected_shape
    );

    test::expect_no_throw(
        "SoftmaxRegression categorical_cross_entropy is small after fit",
        test_softmax_regression_categorical_cross_entropy_is_small_after_fit
    );

    test::expect_no_throw(
        "SoftmaxRegression stores loss history",
        test_softmax_regression_stores_loss_history
    );

    test::expect_no_throw(
        "SoftmaxRegression loss decreases",
        test_softmax_regression_loss_decreases
    );

    test::expect_invalid_argument(
        "SoftmaxRegression rejects invalid learning rate",
        test_softmax_regression_rejects_invalid_learning_rate
    );

    test::expect_invalid_argument(
        "SoftmaxRegression rejects Lasso fitting for now",
        test_softmax_regression_rejects_lasso_for_now
    );

    test::expect_no_throw(
        "SoftmaxRegression supports Ridge lambda > 0",
        test_softmax_regression_supports_ridge_lambda_positive
    );

    test::expect_no_throw(
        "SoftmaxRegression classification accuracy is high after fit",
        test_softmax_regression_classification_accuracy_is_high_after_fit
    );
}

void run_softmax_regression_experiment_export_tests() {
    std::cout << "\n[Phase 4.9] Softmax regression experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports softmax multiclass summary",
        test_experiment_softmax_multiclass_summary_exports_files
    );

    test::expect_no_throw(
        "Experiment exports softmax regularization comparison",
        test_experiment_softmax_regularization_comparison_exports_files
    );

    test::expect_no_throw(
        "Experiment exports softmax prediction table",
        test_experiment_softmax_prediction_table_exports_files
    );
}

}  // namespace

namespace ml::experiments {

void run_phase4_softmax_regression_sanity() {
    run_multiclass_utility_tests();
    run_multiclass_metric_tests();
    run_softmax_regression_tests();
    run_softmax_regression_experiment_export_tests();
}

}  // namespace ml::experiments