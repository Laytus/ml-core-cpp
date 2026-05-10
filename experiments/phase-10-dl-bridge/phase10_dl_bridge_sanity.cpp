#include "phase10_dl_bridge_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/types.hpp"
#include "ml/dl_bridge/perceptron.hpp"
#include "ml/dl_bridge/mlp.hpp"

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
// Phase 10.1 Perceptron tests
// -----------------------------------------------------------------------------

ml::Matrix make_perceptron_linearly_separable_X() {
    ml::Matrix X(6, 2);

    X << -2.0, -1.0,
         -1.5, -1.0,
         -1.0, -2.0,
          1.0,  2.0,
          1.5,  1.0,
          2.0,  1.0;

    return X;
}

ml::Vector make_perceptron_linearly_separable_y() {
    return make_vector({
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    });
}

void test_perceptron_options_accept_defaults() {
    const ml::PerceptronOptions options;

    ml::validate_perceptron_options(
        options,
        "test_perceptron_options_accept_defaults"
    );
}

void test_perceptron_options_accept_valid_values() {
    ml::PerceptronOptions options;
    options.learning_rate = 0.05;
    options.max_epochs = 10;

    ml::validate_perceptron_options(
        options,
        "test_perceptron_options_accept_valid_values"
    );
}

void test_perceptron_options_reject_zero_learning_rate() {
    ml::PerceptronOptions options;
    options.learning_rate = 0.0;

    ml::validate_perceptron_options(
        options,
        "test_perceptron_options_reject_zero_learning_rate"
    );
}

void test_perceptron_options_reject_negative_learning_rate() {
    ml::PerceptronOptions options;
    options.learning_rate = -0.1;

    ml::validate_perceptron_options(
        options,
        "test_perceptron_options_reject_negative_learning_rate"
    );
}

void test_perceptron_options_reject_non_finite_learning_rate() {
    ml::PerceptronOptions options;
    options.learning_rate =
        std::numeric_limits<double>::infinity();

    ml::validate_perceptron_options(
        options,
        "test_perceptron_options_reject_non_finite_learning_rate"
    );
}

void test_perceptron_options_reject_zero_max_epochs() {
    ml::PerceptronOptions options;
    options.max_epochs = 0;

    ml::validate_perceptron_options(
        options,
        "test_perceptron_options_reject_zero_max_epochs"
    );
}

void test_perceptron_reports_not_fitted_initially() {
    const ml::Perceptron model;

    if (model.is_fitted()) {
        throw std::runtime_error(
            "expected Perceptron to start unfitted"
        );
    }
}

void test_perceptron_fit_marks_model_as_fitted() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::Perceptron model;
    model.fit(X, y);

    if (!model.is_fitted()) {
        throw std::runtime_error(
            "expected Perceptron to be fitted"
        );
    }
}

void test_perceptron_weights_have_expected_shape() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::Perceptron model;
    model.fit(X, y);

    if (model.weights().size() != X.cols()) {
        throw std::runtime_error(
            "expected Perceptron weight size to match feature count"
        );
    }

    if (model.num_features() != X.cols()) {
        throw std::runtime_error(
            "expected Perceptron num_features to match X.cols()"
        );
    }
}

void test_perceptron_decision_function_returns_expected_shape() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::Perceptron model;
    model.fit(X, y);

    const ml::Vector scores =
        model.decision_function(X);

    if (scores.size() != X.rows()) {
        throw std::runtime_error(
            "expected one perceptron decision score per sample"
        );
    }
}

void test_perceptron_predict_returns_expected_shape() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::Perceptron model;
    model.fit(X, y);

    const ml::Vector predictions =
        model.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error(
            "expected one perceptron prediction per sample"
        );
    }
}

void test_perceptron_fits_linearly_separable_data() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::PerceptronOptions options;
    options.learning_rate = 0.1;
    options.max_epochs = 50;

    ml::Perceptron model(options);
    model.fit(X, y);

    const ml::Vector predictions =
        model.predict(X);

    test::assert_vector_almost_equal(
        predictions,
        y,
        "test_perceptron_fits_linearly_separable_data"
    );
}

void test_perceptron_decision_scores_have_expected_signs() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::PerceptronOptions options;
    options.learning_rate = 0.1;
    options.max_epochs = 50;

    ml::Perceptron model(options);
    model.fit(X, y);

    const ml::Vector scores =
        model.decision_function(X);

    for (Eigen::Index i = 0; i < scores.size(); ++i) {
        if (y(i) == 0.0 && !(scores(i) < 0.0)) {
            throw std::runtime_error(
                "expected class 0 samples to have negative perceptron scores"
            );
        }

        if (y(i) == 1.0 && !(scores(i) >= 0.0)) {
            throw std::runtime_error(
                "expected class 1 samples to have non-negative perceptron scores"
            );
        }
    }
}

void test_perceptron_stores_mistake_history() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::PerceptronOptions options;
    options.learning_rate = 0.1;
    options.max_epochs = 50;

    ml::Perceptron model(options);
    model.fit(X, y);

    if (model.mistake_history().empty()) {
        throw std::runtime_error(
            "expected non-empty perceptron mistake history"
        );
    }

    for (double mistakes : model.mistake_history()) {
        if (!std::isfinite(mistakes) || mistakes < 0.0) {
            throw std::runtime_error(
                "expected finite non-negative mistake counts"
            );
        }
    }
}

void test_perceptron_rejects_predict_before_fit() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Perceptron model;

    static_cast<void>(
        model.predict(X)
    );
}

void test_perceptron_rejects_decision_function_before_fit() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Perceptron model;

    static_cast<void>(
        model.decision_function(X)
    );
}

void test_perceptron_rejects_empty_fit_X() {
    const ml::Matrix X;
    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::Perceptron model;
    model.fit(X, y);
}

void test_perceptron_rejects_empty_fit_y() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y;

    ml::Perceptron model;
    model.fit(X, y);
}

void test_perceptron_rejects_mismatched_fit_data() {
    ml::Matrix X(2, 2);
    X.setZero();

    const ml::Vector y =
        make_vector({
            0.0,
            1.0,
            1.0
        });

    ml::Perceptron model;
    model.fit(X, y);
}

void test_perceptron_rejects_non_binary_targets() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    ml::Vector y =
        make_perceptron_linearly_separable_y();

    y(0) = 2.0;

    ml::Perceptron model;
    model.fit(X, y);
}

void test_perceptron_rejects_non_integer_targets() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    ml::Vector y =
        make_perceptron_linearly_separable_y();

    y(0) = 0.5;

    ml::Perceptron model;
    model.fit(X, y);
}

void test_perceptron_rejects_non_finite_fit_values() {
    ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    X(0, 0) =
        std::numeric_limits<double>::quiet_NaN();

    ml::Perceptron model;
    model.fit(X, y);
}

void test_perceptron_rejects_predict_feature_mismatch() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::Perceptron model;
    model.fit(X, y);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(
        model.predict(X_bad)
    );
}

void test_perceptron_rejects_non_finite_predict_values() {
    const ml::Matrix X =
        make_perceptron_linearly_separable_X();

    const ml::Vector y =
        make_perceptron_linearly_separable_y();

    ml::Perceptron model;
    model.fit(X, y);

    ml::Matrix X_bad = X;
    X_bad(0, 0) =
        std::numeric_limits<double>::quiet_NaN();

    static_cast<void>(
        model.predict(X_bad)
    );
}

void test_perceptron_accessors_reject_before_fit() {
    const ml::Perceptron model;

    static_cast<void>(
        model.weights()
    );
}

// -----------------------------------------------------------------------------
// Phase 10.2 TinyMLPBinaryClassifier tests
// -----------------------------------------------------------------------------

ml::Matrix make_tiny_mlp_xor_X() {
    ml::Matrix X(4, 2);

    X << 0.0, 0.0,
         0.0, 1.0,
         1.0, 0.0,
         1.0, 1.0;

    return X;
}

ml::Vector make_tiny_mlp_xor_y() {
    return make_vector({
        0.0,
        1.0,
        1.0,
        0.0
    });
}

void test_tiny_mlp_options_accept_defaults() {
    const ml::TinyMLPBinaryClassifierOptions options;

    ml::validate_tiny_mlp_binary_classifier_options(
        options,
        "test_tiny_mlp_options_accept_defaults"
    );
}

void test_tiny_mlp_options_reject_zero_hidden_units() {
    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 0;

    ml::validate_tiny_mlp_binary_classifier_options(
        options,
        "test_tiny_mlp_options_reject_zero_hidden_units"
    );
}

void test_tiny_mlp_options_reject_zero_learning_rate() {
    ml::TinyMLPBinaryClassifierOptions options;
    options.learning_rate = 0.0;

    ml::validate_tiny_mlp_binary_classifier_options(
        options,
        "test_tiny_mlp_options_reject_zero_learning_rate"
    );
}

void test_tiny_mlp_options_reject_negative_learning_rate() {
    ml::TinyMLPBinaryClassifierOptions options;
    options.learning_rate = -0.1;

    ml::validate_tiny_mlp_binary_classifier_options(
        options,
        "test_tiny_mlp_options_reject_negative_learning_rate"
    );
}

void test_tiny_mlp_options_reject_non_finite_learning_rate() {
    ml::TinyMLPBinaryClassifierOptions options;
    options.learning_rate =
        std::numeric_limits<double>::infinity();

    ml::validate_tiny_mlp_binary_classifier_options(
        options,
        "test_tiny_mlp_options_reject_non_finite_learning_rate"
    );
}

void test_tiny_mlp_options_reject_zero_max_epochs() {
    ml::TinyMLPBinaryClassifierOptions options;
    options.max_epochs = 0;

    ml::validate_tiny_mlp_binary_classifier_options(
        options,
        "test_tiny_mlp_options_reject_zero_max_epochs"
    );
}

void test_tiny_mlp_options_reject_zero_batch_size() {
    ml::TinyMLPBinaryClassifierOptions options;
    options.batch_size = 0;

    ml::validate_tiny_mlp_binary_classifier_options(
        options,
        "test_tiny_mlp_options_reject_zero_batch_size"
    );
}

void test_tiny_mlp_reports_not_fitted_initially() {
    const ml::TinyMLPBinaryClassifier model;

    if (model.is_fitted()) {
        throw std::runtime_error(
            "expected TinyMLPBinaryClassifier to start unfitted"
        );
    }
}

void test_tiny_mlp_fit_marks_model_as_fitted() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 4;
    options.learning_rate = 0.5;
    options.max_epochs = 2000;
    options.batch_size = 4;
    options.random_seed = 7;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    if (!model.is_fitted()) {
        throw std::runtime_error(
            "expected TinyMLPBinaryClassifier to be fitted"
        );
    }
}

void test_tiny_mlp_parameter_shapes() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 4;
    options.learning_rate = 0.5;
    options.max_epochs = 100;
    options.batch_size = 4;
    options.random_seed = 7;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    if (model.W1().rows() != X.cols()) {
        throw std::runtime_error(
            "expected W1 rows to match feature count"
        );
    }

    if (model.W1().cols() != static_cast<Eigen::Index>(options.hidden_units)) {
        throw std::runtime_error(
            "expected W1 cols to match hidden_units"
        );
    }

    if (model.b1().size() != static_cast<Eigen::Index>(options.hidden_units)) {
        throw std::runtime_error(
            "expected b1 size to match hidden_units"
        );
    }

    if (model.W2().rows() != static_cast<Eigen::Index>(options.hidden_units)) {
        throw std::runtime_error(
            "expected W2 rows to match hidden_units"
        );
    }

    if (model.W2().cols() != 1) {
        throw std::runtime_error(
            "expected W2 to have one output column"
        );
    }
}

void test_tiny_mlp_forward_shapes() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 4;
    options.learning_rate = 0.5;
    options.max_epochs = 100;
    options.batch_size = 4;
    options.random_seed = 7;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    const ml::TinyMLPForwardCache cache =
        model.forward(X);

    if (cache.Z1.rows() != X.rows() || cache.Z1.cols() != 4) {
        throw std::runtime_error(
            "expected Z1 shape to be num_samples x hidden_units"
        );
    }

    if (cache.A1.rows() != X.rows() || cache.A1.cols() != 4) {
        throw std::runtime_error(
            "expected A1 shape to be num_samples x hidden_units"
        );
    }

    if (cache.Z2.rows() != X.rows() || cache.Z2.cols() != 1) {
        throw std::runtime_error(
            "expected Z2 shape to be num_samples x 1"
        );
    }

    if (cache.A2.rows() != X.rows() || cache.A2.cols() != 1) {
        throw std::runtime_error(
            "expected A2 shape to be num_samples x 1"
        );
    }
}

void test_tiny_mlp_predict_proba_returns_expected_shape() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 4;
    options.learning_rate = 0.5;
    options.max_epochs = 100;
    options.batch_size = 4;
    options.random_seed = 7;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    const ml::Vector probabilities =
        model.predict_proba(X);

    if (probabilities.size() != X.rows()) {
        throw std::runtime_error(
            "expected one MLP probability per sample"
        );
    }

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        if (
            !std::isfinite(probabilities(i)) ||
            probabilities(i) < 0.0 ||
            probabilities(i) > 1.0
        ) {
            throw std::runtime_error(
                "expected MLP probabilities in [0, 1]"
            );
        }
    }
}

void test_tiny_mlp_predict_returns_expected_shape() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 4;
    options.learning_rate = 0.5;
    options.max_epochs = 100;
    options.batch_size = 4;
    options.random_seed = 7;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    const ml::Vector predictions =
        model.predict(X);

    if (predictions.size() != y.size()) {
        throw std::runtime_error(
            "expected one MLP prediction per sample"
        );
    }
}

void test_tiny_mlp_fits_xor_demo() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 8;
    options.learning_rate = 0.5;
    options.max_epochs = 5000;
    options.batch_size = 4;
    options.random_seed = 7;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    const ml::Vector predictions =
        model.predict(X);

    test::assert_vector_almost_equal(
        predictions,
        y,
        "test_tiny_mlp_fits_xor_demo"
    );
}

void test_tiny_mlp_stores_loss_history() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 4;
    options.learning_rate = 0.5;
    options.max_epochs = 200;
    options.batch_size = 4;
    options.random_seed = 7;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    if (model.loss_history().empty()) {
        throw std::runtime_error(
            "expected non-empty MLP loss history"
        );
    }

    for (double loss : model.loss_history()) {
        if (!std::isfinite(loss) || loss < 0.0) {
            throw std::runtime_error(
                "expected finite non-negative MLP loss"
            );
        }
    }
}

void test_tiny_mlp_loss_decreases() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 8;
    options.learning_rate = 0.5;
    options.max_epochs = 1000;
    options.batch_size = 4;
    options.random_seed = 7;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    const std::vector<double>& history =
        model.loss_history();

    if (history.size() < 2) {
        throw std::runtime_error(
            "expected at least two loss history entries"
        );
    }

    if (history.back() >= history.front()) {
        throw std::runtime_error(
            "expected MLP training loss to decrease"
        );
    }
}

void test_tiny_mlp_rejects_predict_before_fit() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::TinyMLPBinaryClassifier model;

    static_cast<void>(
        model.predict(X)
    );
}

void test_tiny_mlp_rejects_forward_before_fit() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::TinyMLPBinaryClassifier model;

    static_cast<void>(
        model.forward(X)
    );
}

void test_tiny_mlp_rejects_empty_fit_X() {
    const ml::Matrix X;
    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifier model;
    model.fit(X, y);
}

void test_tiny_mlp_rejects_empty_fit_y() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y;

    ml::TinyMLPBinaryClassifier model;
    model.fit(X, y);
}

void test_tiny_mlp_rejects_mismatched_fit_data() {
    ml::Matrix X(2, 2);
    X.setZero();

    const ml::Vector y =
        make_vector({
            0.0,
            1.0,
            1.0
        });

    ml::TinyMLPBinaryClassifier model;
    model.fit(X, y);
}

void test_tiny_mlp_rejects_non_binary_targets() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    ml::Vector y =
        make_tiny_mlp_xor_y();

    y(0) = 2.0;

    ml::TinyMLPBinaryClassifier model;
    model.fit(X, y);
}

void test_tiny_mlp_rejects_non_integer_targets() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    ml::Vector y =
        make_tiny_mlp_xor_y();

    y(0) = 0.5;

    ml::TinyMLPBinaryClassifier model;
    model.fit(X, y);
}

void test_tiny_mlp_rejects_non_finite_fit_values() {
    ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    X(0, 0) =
        std::numeric_limits<double>::quiet_NaN();

    ml::TinyMLPBinaryClassifier model;
    model.fit(X, y);
}

void test_tiny_mlp_rejects_predict_feature_mismatch() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifier model;
    model.fit(X, y);

    ml::Matrix X_bad(2, 3);
    X_bad.setZero();

    static_cast<void>(
        model.predict(X_bad)
    );
}

void test_tiny_mlp_rejects_non_finite_predict_values() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifier model;
    model.fit(X, y);

    ml::Matrix X_bad = X;
    X_bad(0, 0) =
        std::numeric_limits<double>::quiet_NaN();

    static_cast<void>(
        model.predict(X_bad)
    );
}

void test_tiny_mlp_accessors_reject_before_fit() {
    const ml::TinyMLPBinaryClassifier model;

    static_cast<void>(
        model.W1()
    );
}

// -----------------------------------------------------------------------------
// Phase 10.3 DL bridge demo export helpers
// -----------------------------------------------------------------------------

const std::string k_phase10_output_dir = "outputs/phase-10-dl-bridge";

void ensure_phase10_output_dir_exists() {
    std::filesystem::create_directories(k_phase10_output_dir);
}

struct TinyMLPXorPredictionRow {
    std::string sample_name;
    double x0{0.0};
    double x1{0.0};
    double true_label{0.0};
    double perceptron_prediction{0.0};
    double mlp_probability{0.0};
    double mlp_prediction{0.0};
};

struct TinyMLPLossHistoryRow {
    std::size_t epoch{0};
    double loss{0.0};
};

std::vector<TinyMLPXorPredictionRow> make_tiny_mlp_xor_prediction_rows() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::PerceptronOptions perceptron_options;
    perceptron_options.learning_rate = 0.1;
    perceptron_options.max_epochs = 50;

    ml::Perceptron perceptron(perceptron_options);
    perceptron.fit(X, y);

    const ml::Vector perceptron_predictions =
        perceptron.predict(X);

    ml::TinyMLPBinaryClassifierOptions mlp_options;
    mlp_options.hidden_units = 8;
    mlp_options.learning_rate = 0.5;
    mlp_options.max_epochs = 5000;
    mlp_options.batch_size = 4;
    mlp_options.random_seed = 7;

    ml::TinyMLPBinaryClassifier mlp(mlp_options);
    mlp.fit(X, y);

    const ml::Vector mlp_probabilities =
        mlp.predict_proba(X);

    const ml::Vector mlp_predictions =
        mlp.predict(X);

    std::vector<TinyMLPXorPredictionRow> rows;
    rows.reserve(static_cast<std::size_t>(X.rows()));

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        rows.push_back(
            TinyMLPXorPredictionRow{
                "xor_" + std::to_string(i),
                X(i, 0),
                X(i, 1),
                y(i),
                perceptron_predictions(i),
                mlp_probabilities(i),
                mlp_predictions(i)
            }
        );
    }

    return rows;
}

std::vector<TinyMLPLossHistoryRow> make_tiny_mlp_loss_history_rows() {
    const ml::Matrix X =
        make_tiny_mlp_xor_X();

    const ml::Vector y =
        make_tiny_mlp_xor_y();

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 8;
    options.learning_rate = 0.5;
    options.max_epochs = 5000;
    options.batch_size = 4;
    options.random_seed = 7;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    const std::vector<double>& loss_history =
        model.loss_history();

    std::vector<TinyMLPLossHistoryRow> rows;
    rows.reserve(loss_history.size());

    for (std::size_t epoch = 0; epoch < loss_history.size(); ++epoch) {
        rows.push_back(
            TinyMLPLossHistoryRow{
                epoch + 1,
                loss_history[epoch]
            }
        );
    }

    return rows;
}

void export_tiny_mlp_xor_predictions_csv(
    const std::vector<TinyMLPXorPredictionRow>& rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_tiny_mlp_xor_predictions_csv: failed to open output file"
        );
    }

    file << "sample_name,x0,x1,true_label,"
         << "perceptron_prediction,mlp_probability,mlp_prediction\n";

    for (const TinyMLPXorPredictionRow& row : rows) {
        file << row.sample_name << ","
             << row.x0 << ","
             << row.x1 << ","
             << row.true_label << ","
             << row.perceptron_prediction << ","
             << row.mlp_probability << ","
             << row.mlp_prediction << "\n";
    }
}

void export_tiny_mlp_loss_history_csv(
    const std::vector<TinyMLPLossHistoryRow>& rows,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_tiny_mlp_loss_history_csv: failed to open output file"
        );
    }

    file << "epoch,loss\n";

    for (const TinyMLPLossHistoryRow& row : rows) {
        file << row.epoch << ","
             << row.loss << "\n";
    }
}

void export_dl_bridge_summary_txt(
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_dl_bridge_summary_txt: failed to open output file"
        );
    }

    file << "Deep Learning Bridge Summary\n\n";

    file << "Phase 10 connects classical ML models to neural-network training.\n\n";

    file << "Perceptron:\n";
    file << "- computes a linear score w^T x + b\n";
    file << "- applies a hard threshold\n";
    file << "- can solve linearly separable data\n";
    file << "- cannot solve XOR because XOR is not linearly separable\n\n";

    file << "TinyMLPBinaryClassifier:\n";
    file << "- uses architecture: input -> hidden layer -> ReLU -> output layer -> sigmoid\n";
    file << "- uses binary cross-entropy loss\n";
    file << "- trains with mini-batch gradient descent\n";
    file << "- implements vectorized forward propagation\n";
    file << "- implements manual backpropagation as structured chain rule\n";
    file << "- can solve XOR because the hidden ReLU layer creates a nonlinear representation\n\n";

    file << "Connection to previous phases:\n";
    file << "- Linear models introduced weights, biases, and vectorized predictions.\n";
    file << "- Logistic regression introduced sigmoid probabilities and BCE.\n";
    file << "- Optimization introduced learning rates, gradient descent, and training histories.\n";
    file << "- Probabilistic ML connected BCE to likelihood-based thinking.\n";
    file << "- The MLP combines these ideas into layered differentiable computation.\n\n";

    file << "What stays the same from ML to DL:\n";
    file << "- trainable parameters\n";
    file << "- losses\n";
    file << "- gradients\n";
    file << "- optimization loops\n";
    file << "- validation and evaluation discipline\n\n";

    file << "What changes in DL:\n";
    file << "- models learn intermediate representations\n";
    file << "- multiple layers are composed\n";
    file << "- backpropagation computes gradients through all layers\n";
    file << "- initialization and activation functions become more important\n";
    file << "- models can represent nonlinear decision boundaries more naturally\n\n";

    file << "Core mental model:\n";
    file << "A neural network is a layered differentiable function trained by forward pass, loss computation, backpropagation, and parameter updates.\n";
}

void test_experiment_exports_tiny_mlp_binary_classification_demo() {
    ensure_phase10_output_dir_exists();

    const std::vector<TinyMLPXorPredictionRow> prediction_rows =
        make_tiny_mlp_xor_prediction_rows();

    const std::vector<TinyMLPLossHistoryRow> loss_rows =
        make_tiny_mlp_loss_history_rows();

    const std::string prediction_path =
        k_phase10_output_dir + "/tiny_mlp_xor_predictions.csv";

    const std::string loss_history_path =
        k_phase10_output_dir + "/tiny_mlp_loss_history.csv";

    const std::string summary_path =
        k_phase10_output_dir + "/dl_bridge_summary.txt";

    export_tiny_mlp_xor_predictions_csv(
        prediction_rows,
        prediction_path
    );

    export_tiny_mlp_loss_history_csv(
        loss_rows,
        loss_history_path
    );

    export_dl_bridge_summary_txt(
        summary_path
    );

    if (!std::filesystem::exists(prediction_path)) {
        throw std::runtime_error(
            "expected tiny_mlp_xor_predictions.csv to exist"
        );
    }

    if (!std::filesystem::exists(loss_history_path)) {
        throw std::runtime_error(
            "expected tiny_mlp_loss_history.csv to exist"
        );
    }

    if (!std::filesystem::exists(summary_path)) {
        throw std::runtime_error(
            "expected dl_bridge_summary.txt to exist"
        );
    }

    if (prediction_rows.size() != 4) {
        throw std::runtime_error(
            "expected four XOR prediction rows"
        );
    }

    if (loss_rows.empty()) {
        throw std::runtime_error(
            "expected non-empty MLP loss history rows"
        );
    }

    bool perceptron_makes_xor_mistake = false;
    bool mlp_solves_xor = true;

    for (const TinyMLPXorPredictionRow& row : prediction_rows) {
        if (row.perceptron_prediction != row.true_label) {
            perceptron_makes_xor_mistake = true;
        }

        if (row.mlp_prediction != row.true_label) {
            mlp_solves_xor = false;
        }

        if (
            !std::isfinite(row.mlp_probability) ||
            row.mlp_probability < 0.0 ||
            row.mlp_probability > 1.0
        ) {
            throw std::runtime_error(
                "expected exported MLP probability to be in [0, 1]"
            );
        }
    }

    if (!perceptron_makes_xor_mistake) {
        throw std::runtime_error(
            "expected perceptron to fail at least one XOR sample"
        );
    }

    if (!mlp_solves_xor) {
        throw std::runtime_error(
            "expected TinyMLPBinaryClassifier to solve XOR demo"
        );
    }

    if (loss_rows.back().loss >= loss_rows.front().loss) {
        throw std::runtime_error(
            "expected exported MLP loss history to decrease"
        );
    }
}

// -----------------------------------------------------------------------------
// Test runners
// -----------------------------------------------------------------------------

void run_perceptron_tests() {
    std::cout << "\n[Phase 10.1] Perceptron tests\n\n";

    test::expect_no_throw(
        "PerceptronOptions accepts defaults",
        test_perceptron_options_accept_defaults
    );

    test::expect_no_throw(
        "PerceptronOptions accepts valid values",
        test_perceptron_options_accept_valid_values
    );

    test::expect_invalid_argument(
        "PerceptronOptions rejects zero learning_rate",
        test_perceptron_options_reject_zero_learning_rate
    );

    test::expect_invalid_argument(
        "PerceptronOptions rejects negative learning_rate",
        test_perceptron_options_reject_negative_learning_rate
    );

    test::expect_invalid_argument(
        "PerceptronOptions rejects non-finite learning_rate",
        test_perceptron_options_reject_non_finite_learning_rate
    );

    test::expect_invalid_argument(
        "PerceptronOptions rejects zero max_epochs",
        test_perceptron_options_reject_zero_max_epochs
    );

    test::expect_no_throw(
        "Perceptron reports not fitted initially",
        test_perceptron_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "Perceptron fit marks model as fitted",
        test_perceptron_fit_marks_model_as_fitted
    );

    test::expect_no_throw(
        "Perceptron weights have expected shape",
        test_perceptron_weights_have_expected_shape
    );

    test::expect_no_throw(
        "Perceptron decision_function returns expected shape",
        test_perceptron_decision_function_returns_expected_shape
    );

    test::expect_no_throw(
        "Perceptron predict returns expected shape",
        test_perceptron_predict_returns_expected_shape
    );

    test::expect_no_throw(
        "Perceptron fits linearly separable data",
        test_perceptron_fits_linearly_separable_data
    );

    test::expect_no_throw(
        "Perceptron decision scores have expected signs",
        test_perceptron_decision_scores_have_expected_signs
    );

    test::expect_no_throw(
        "Perceptron stores mistake history",
        test_perceptron_stores_mistake_history
    );

    test::expect_invalid_argument(
        "Perceptron rejects predict before fit",
        test_perceptron_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "Perceptron rejects decision_function before fit",
        test_perceptron_rejects_decision_function_before_fit
    );

    test::expect_invalid_argument(
        "Perceptron rejects empty fit X",
        test_perceptron_rejects_empty_fit_X
    );

    test::expect_invalid_argument(
        "Perceptron rejects empty fit y",
        test_perceptron_rejects_empty_fit_y
    );

    test::expect_invalid_argument(
        "Perceptron rejects mismatched fit data",
        test_perceptron_rejects_mismatched_fit_data
    );

    test::expect_invalid_argument(
        "Perceptron rejects non-binary targets",
        test_perceptron_rejects_non_binary_targets
    );

    test::expect_invalid_argument(
        "Perceptron rejects non-integer targets",
        test_perceptron_rejects_non_integer_targets
    );

    test::expect_invalid_argument(
        "Perceptron rejects non-finite fit values",
        test_perceptron_rejects_non_finite_fit_values
    );

    test::expect_invalid_argument(
        "Perceptron rejects predict feature mismatch",
        test_perceptron_rejects_predict_feature_mismatch
    );

    test::expect_invalid_argument(
        "Perceptron rejects non-finite predict values",
        test_perceptron_rejects_non_finite_predict_values
    );

    test::expect_invalid_argument(
        "Perceptron accessors reject before fit",
        test_perceptron_accessors_reject_before_fit
    );
}

void run_tiny_mlp_tests() {
    std::cout << "\n[Phase 10.2] TinyMLPBinaryClassifier tests\n\n";

    test::expect_no_throw(
        "TinyMLPBinaryClassifierOptions accepts defaults",
        test_tiny_mlp_options_accept_defaults
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifierOptions rejects zero hidden_units",
        test_tiny_mlp_options_reject_zero_hidden_units
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifierOptions rejects zero learning_rate",
        test_tiny_mlp_options_reject_zero_learning_rate
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifierOptions rejects negative learning_rate",
        test_tiny_mlp_options_reject_negative_learning_rate
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifierOptions rejects non-finite learning_rate",
        test_tiny_mlp_options_reject_non_finite_learning_rate
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifierOptions rejects zero max_epochs",
        test_tiny_mlp_options_reject_zero_max_epochs
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifierOptions rejects zero batch_size",
        test_tiny_mlp_options_reject_zero_batch_size
    );

    test::expect_no_throw(
        "TinyMLPBinaryClassifier reports not fitted initially",
        test_tiny_mlp_reports_not_fitted_initially
    );

    test::expect_no_throw(
        "TinyMLPBinaryClassifier fit marks model as fitted",
        test_tiny_mlp_fit_marks_model_as_fitted
    );

    test::expect_no_throw(
        "TinyMLPBinaryClassifier parameter shapes are correct",
        test_tiny_mlp_parameter_shapes
    );

    test::expect_no_throw(
        "TinyMLPBinaryClassifier forward shapes are correct",
        test_tiny_mlp_forward_shapes
    );

    test::expect_no_throw(
        "TinyMLPBinaryClassifier predict_proba returns expected shape",
        test_tiny_mlp_predict_proba_returns_expected_shape
    );

    test::expect_no_throw(
        "TinyMLPBinaryClassifier predict returns expected shape",
        test_tiny_mlp_predict_returns_expected_shape
    );

    test::expect_no_throw(
        "TinyMLPBinaryClassifier fits XOR demo",
        test_tiny_mlp_fits_xor_demo
    );

    test::expect_no_throw(
        "TinyMLPBinaryClassifier stores loss history",
        test_tiny_mlp_stores_loss_history
    );

    test::expect_no_throw(
        "TinyMLPBinaryClassifier loss decreases",
        test_tiny_mlp_loss_decreases
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects predict before fit",
        test_tiny_mlp_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects forward before fit",
        test_tiny_mlp_rejects_forward_before_fit
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects empty fit X",
        test_tiny_mlp_rejects_empty_fit_X
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects empty fit y",
        test_tiny_mlp_rejects_empty_fit_y
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects mismatched fit data",
        test_tiny_mlp_rejects_mismatched_fit_data
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects non-binary targets",
        test_tiny_mlp_rejects_non_binary_targets
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects non-integer targets",
        test_tiny_mlp_rejects_non_integer_targets
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects non-finite fit values",
        test_tiny_mlp_rejects_non_finite_fit_values
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects predict feature mismatch",
        test_tiny_mlp_rejects_predict_feature_mismatch
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier rejects non-finite predict values",
        test_tiny_mlp_rejects_non_finite_predict_values
    );

    test::expect_invalid_argument(
        "TinyMLPBinaryClassifier accessors reject before fit",
        test_tiny_mlp_accessors_reject_before_fit
    );
}

void run_dl_bridge_demo_export_tests() {
    std::cout << "\n[Phase 10.3] DL bridge demo export tests\n\n";

    test::expect_no_throw(
        "Experiment exports tiny MLP binary classification demo",
        test_experiment_exports_tiny_mlp_binary_classification_demo
    );
}

}  // namespace

namespace ml::experiments {

void run_phase10_dl_bridge_sanity() {
    run_perceptron_tests();
    run_tiny_mlp_tests();
    run_dl_bridge_demo_export_tests();
}

}  // namespace ml::experiments