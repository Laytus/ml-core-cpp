#include "phase5_optimization_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/classification_utils.hpp"
#include "ml/common/types.hpp"
#include "ml/linear_models/regularization.hpp"
#include "ml/optimization/gradient_descent.hpp"
#include "ml/optimization/mini_batch_gradient_descent.hpp"
#include "ml/optimization/model_optimization_adapters.hpp"
#include "ml/optimization/sgd.hpp"
#include "ml/optimization/training_history.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test = ml::experiments::test;

namespace {

ml::Matrix vector_to_column_matrix(const ml::Vector& y) {
    ml::Matrix result(y.size(), 1);
    result.col(0) = y;
    return result;
}

ml::Matrix make_linear_X() {
    ml::Matrix X(6, 2);
    X << 0.0, 0.0,
         1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0,
         2.0, 1.0,
         1.0, 2.0;

    return X;
}

ml::Matrix make_linear_y_matrix() {
    const ml::Matrix X = make_linear_X();

    ml::Vector y(X.rows());

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        y(i) = 2.0 * X(i, 0) - 3.0 * X(i, 1) + 1.0;
    }

    return vector_to_column_matrix(y);
}

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

ml::Matrix make_binary_classification_y_matrix() {
    ml::Vector y(6);
    y << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;

    return vector_to_column_matrix(y);
}

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

ml::Matrix make_multiclass_classification_y_one_hot() {
    ml::Vector y(9);
    y << 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0,
         2.0, 2.0, 2.0;

    return ml::one_hot_encode(y, 3);
}

ml::Matrix make_poorly_scaled_linear_X() {
    ml::Matrix X(6, 2);
    X << 0.0, 0.0,
         1.0, 1000.0,
         2.0, 2000.0,
         3.0, 3000.0,
         4.0, 4000.0,
         5.0, 5000.0;

    return X;
}

ml::Matrix make_poorly_scaled_linear_y_matrix() {
    const ml::Matrix X = make_poorly_scaled_linear_X();

    ml::Vector y(X.rows());

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        y(i) = 2.0 * X(i, 0) - 0.003 * X(i, 1) + 1.0;
    }

    return vector_to_column_matrix(y);
}

ml::Matrix make_manually_scaled_linear_X() {
    const ml::Matrix X = make_poorly_scaled_linear_X();

    ml::Matrix scaled = X;
    scaled.col(1) = scaled.col(1) / 1000.0;

    return scaled;
}

ml::OptimizerOptions make_batch_options() {
    ml::OptimizerOptions options;
    options.learning_rate = 0.05;
    options.max_epochs = 5000;
    options.max_iterations = 5000;
    options.batch_size = 32;
    options.random_seed = 42;
    options.shuffle = true;
    options.loss_tolerance = 1e-12;
    options.gradient_tolerance = 0.0;
    options.store_loss_history = true;

    return options;
}

ml::OptimizerOptions make_sgd_options() {
    ml::OptimizerOptions options;
    options.learning_rate = 0.01;
    options.max_epochs = 1000;
    options.max_iterations = 100000;
    options.batch_size = 1;
    options.random_seed = 42;
    options.shuffle = true;
    options.loss_tolerance = 1e-12;
    options.gradient_tolerance = 0.0;
    options.store_loss_history = true;

    return options;
}

ml::OptimizerOptions make_mini_batch_options() {
    ml::OptimizerOptions options;
    options.learning_rate = 0.05;
    options.max_epochs = 3000;
    options.max_iterations = 100000;
    options.batch_size = 2;
    options.random_seed = 42;
    options.shuffle = true;
    options.loss_tolerance = 1e-12;
    options.gradient_tolerance = 0.0;
    options.store_loss_history = true;

    return options;
}

// ---- Experiment export helpers ----

const std::string k_phase5_output_dir = "outputs/phase-5-optimization";

void ensure_phase5_output_dir_exists() {
    std::filesystem::create_directories(k_phase5_output_dir);
}

struct OptimizationExperimentResult {
    std::string experiment_name;
    std::string dataset_name;
    std::string optimizer_name;

    double learning_rate{0.0};
    double momentum{0.0};
    std::size_t batch_size{0};

    bool scaled{false};

    double initial_loss{0.0};
    double final_loss{0.0};
    double best_loss{0.0};
    double loss_improvement{0.0};

    std::size_t iterations_run{0};
    std::size_t epochs_run{0};

    bool converged{false};
    std::string stop_reason;
};

OptimizationExperimentResult make_optimization_experiment_result(
    const std::string& experiment_name,
    const std::string& dataset_name,
    bool scaled,
    const ml::TrainingHistory& history
) {
    if (!history.has_losses()) {
        throw std::runtime_error(
            "make_optimization_experiment_result: history must contain losses"
        );
    }

    return OptimizationExperimentResult{
        experiment_name,
        dataset_name,
        history.optimizer_name,
        history.learning_rate,
        history.momentum,
        history.batch_size,
        scaled,
        history.initial_loss(),
        history.final_loss(),
        history.best_loss(),
        history.loss_improvement(),
        history.iterations_run,
        history.epochs_run,
        history.converged,
        ml::optimization_stop_reason_name(history.stop_reason)
    };
}

void export_optimization_results_csv(
    const std::vector<OptimizationExperimentResult>& results,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_optimization_results_csv: failed to open output file"
        );
    }

    file << "experiment_name,dataset_name,optimizer_name,learning_rate,momentum,batch_size,"
         << "scaled,initial_loss,final_loss,best_loss,loss_improvement,"
         << "iterations_run,epochs_run,converged,stop_reason\n";

    for (const OptimizationExperimentResult& result : results) {
        file << result.experiment_name << ","
             << result.dataset_name << ","
             << result.optimizer_name << ","
             << result.learning_rate << ","
             << result.momentum << ","
             << result.batch_size << ","
             << (result.scaled ? "true" : "false") << ","
             << result.initial_loss << ","
             << result.final_loss << ","
             << result.best_loss << ","
             << result.loss_improvement << ","
             << result.iterations_run << ","
             << result.epochs_run << ","
             << (result.converged ? "true" : "false") << ","
             << result.stop_reason << "\n";
    }
}

void export_optimization_results_txt(
    const std::vector<OptimizationExperimentResult>& results,
    const std::string& title,
    const std::string& output_path
) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_optimization_results_txt: failed to open output file"
        );
    }

    file << title << "\n\n";

    for (const OptimizationExperimentResult& result : results) {
        file << "Experiment: " << result.experiment_name << "\n"
             << "Dataset: " << result.dataset_name << "\n"
             << "Optimizer: " << result.optimizer_name << "\n"
             << "Learning rate: " << result.learning_rate << "\n"
             << "Momentum: " << result.momentum << "\n"
             << "Batch size: " << result.batch_size << "\n"
             << "Scaled: " << (result.scaled ? "true" : "false") << "\n"
             << "Initial loss: " << result.initial_loss << "\n"
             << "Final loss: " << result.final_loss << "\n"
             << "Best loss: " << result.best_loss << "\n"
             << "Loss improvement: " << result.loss_improvement << "\n"
             << "Iterations run: " << result.iterations_run << "\n"
             << "Epochs run: " << result.epochs_run << "\n"
             << "Converged: " << (result.converged ? "true" : "false") << "\n"
             << "Stop reason: " << result.stop_reason << "\n\n";
    }
}

// ---- Training History ----

void test_training_history_computes_loss_summary_values() {
    ml::TrainingHistory history;
    history.losses = {10.0, 7.0, 4.0, 3.0};

    test::assert_almost_equal(
        history.initial_loss(),
        10.0,
        "test_training_history_computes_loss_summary_values initial_loss"
    );

    test::assert_almost_equal(
        history.final_loss(),
        3.0,
        "test_training_history_computes_loss_summary_values final_loss"
    );

    test::assert_almost_equal(
        history.best_loss(),
        3.0,
        "test_training_history_computes_loss_summary_values best_loss"
    );

    test::assert_almost_equal(
        history.loss_improvement(),
        7.0,
        "test_training_history_computes_loss_summary_values loss_improvement"
    );
}

void test_training_history_rejects_summary_values_without_losses() {
    ml::TrainingHistory history;

    static_cast<void>(history.initial_loss());
}

void test_optimization_stop_reason_name_returns_expected_value() {
    const std::string name = ml::optimization_stop_reason_name(
        ml::OptimizationStopReason::LossToleranceReached
    );

    if (name != "LossToleranceReached") {
        throw std::runtime_error("unexpected optimization stop reason name");
    }
}

void test_training_history_summary_contains_key_fields() {
    ml::TrainingHistory history;
    history.optimizer_name = "TestOptimizer";
    history.learning_rate = 0.01;
    history.momentum = 0.9;
    history.batch_size = 4;
    history.iterations_run = 10;
    history.epochs_run = 2;
    history.converged = true;
    history.stop_reason = ml::OptimizationStopReason::LossToleranceReached;
    history.losses = {5.0, 2.0};
    history.gradient_norms = {3.0, 1.0};

    const std::string summary = ml::training_history_summary(history);

    if (summary.find("Optimizer: TestOptimizer") == std::string::npos) {
        throw std::runtime_error("expected summary to contain optimizer name");
    }

    if (summary.find("Learning rate: 0.01") == std::string::npos) {
        throw std::runtime_error("expected summary to contain learning rate");
    }

    if (summary.find("Momentum: 0.9") == std::string::npos) {
        throw std::runtime_error("expected summary to contain momentum");
    }

    if (summary.find("Stop reason: LossToleranceReached") == std::string::npos) {
        throw std::runtime_error("expected summary to contain stop reason");
    }

    if (summary.find("Initial loss: 5") == std::string::npos) {
        throw std::runtime_error("expected summary to contain initial loss");
    }

    if (summary.find("Final loss: 2") == std::string::npos) {
        throw std::runtime_error("expected summary to contain final loss");
    }
}

void test_batch_gradient_descent_records_gradient_norms() {
    ml::LinearRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    ml::OptimizerOptions options = make_batch_options();
    options.store_loss_history = true;

    ml::BatchGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    if (history.gradient_norms.empty()) {
        throw std::runtime_error("expected Batch GD to record gradient norms");
    }
}

void test_stochastic_gradient_descent_records_gradient_norms() {
    ml::LinearRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    ml::OptimizerOptions options = make_sgd_options();
    options.store_loss_history = true;

    ml::StochasticGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    if (history.gradient_norms.empty()) {
        throw std::runtime_error("expected SGD to record gradient norms");
    }
}

void test_mini_batch_gradient_descent_records_gradient_norms() {
    ml::LinearRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    ml::OptimizerOptions options = make_mini_batch_options();
    options.store_loss_history = true;

    ml::MiniBatchGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    if (history.gradient_norms.empty()) {
        throw std::runtime_error("expected Mini-batch GD to record gradient norms");
    }
}

// ---- TrainingHistory / optimizer interface tests ----

void test_batch_gradient_descent_reports_name_and_options() {
    ml::OptimizerOptions options = make_batch_options();
    options.learning_rate = 0.123;

    const ml::BatchGradientDescent optimizer(options);

    if (std::string(optimizer.name()) != "BatchGradientDescent") {
        throw std::runtime_error("expected optimizer name to be BatchGradientDescent");
    }

    test::assert_almost_equal(
        optimizer.options().learning_rate,
        0.123,
        "test_batch_gradient_descent_reports_name_and_options learning_rate"
    );
}

void test_stochastic_gradient_descent_reports_name_and_options() {
    ml::OptimizerOptions options = make_sgd_options();
    options.learning_rate = 0.007;

    const ml::StochasticGradientDescent optimizer(options);

    if (std::string(optimizer.name()) != "StochasticGradientDescent") {
        throw std::runtime_error("expected optimizer name to be StochasticGradientDescent");
    }

    test::assert_almost_equal(
        optimizer.options().learning_rate,
        0.007,
        "test_stochastic_gradient_descent_reports_name_and_options learning_rate"
    );
}

void test_mini_batch_gradient_descent_reports_name_and_options() {
    ml::OptimizerOptions options = make_mini_batch_options();
    options.learning_rate = 0.009;
    options.batch_size = 3;
    options.random_seed = 123;
    options.shuffle = false;

    const ml::MiniBatchGradientDescent optimizer(options);

    if (std::string(optimizer.name()) != "MiniBatchGradientDescent") {
        throw std::runtime_error("expected optimizer name to be MiniBatchGradientDescent");
    }

    test::assert_almost_equal(
        optimizer.options().learning_rate,
        0.009,
        "test_mini_batch_gradient_descent_reports_name_and_options learning_rate"
    );

    if (optimizer.options().batch_size != 3) {
        throw std::runtime_error("expected MiniBatchGradientDescent batch_size option to be stored");
    }

    if (optimizer.options().random_seed != 123) {
        throw std::runtime_error("expected MiniBatchGradientDescent random_seed option to be stored");
    }

    if (optimizer.options().shuffle) {
        throw std::runtime_error("expected MiniBatchGradientDescent shuffle option to be stored");
    }
}

void test_batch_gradient_descent_rejects_invalid_learning_rate() {
    ml::OptimizerOptions options = make_batch_options();
    options.learning_rate = 0.0;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::BatchGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

void test_stochastic_gradient_descent_rejects_invalid_learning_rate() {
    ml::OptimizerOptions options = make_sgd_options();
    options.learning_rate = 0.0;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::StochasticGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

void test_mini_batch_gradient_descent_rejects_invalid_learning_rate() {
    ml::OptimizerOptions options = make_mini_batch_options();
    options.learning_rate = 0.0;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::MiniBatchGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

void test_mini_batch_gradient_descent_rejects_invalid_batch_size() {
    ml::OptimizerOptions options = make_mini_batch_options();
    options.batch_size = 0;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::MiniBatchGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

void test_batch_gradient_descent_rejects_negative_momentum() {
    ml::OptimizerOptions options = make_batch_options();
    options.momentum = -0.1;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::BatchGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

void test_batch_gradient_descent_rejects_momentum_one() {
    ml::OptimizerOptions options = make_batch_options();
    options.momentum = 1.0;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::BatchGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

void test_stochastic_gradient_descent_rejects_negative_momentum() {
    ml::OptimizerOptions options = make_sgd_options();
    options.momentum = -0.1;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::StochasticGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

void test_stochastic_gradient_descent_rejects_momentum_one() {
    ml::OptimizerOptions options = make_sgd_options();
    options.momentum = 1.0;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::StochasticGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

void test_mini_batch_gradient_descent_rejects_negative_momentum() {
    ml::OptimizerOptions options = make_mini_batch_options();
    options.momentum = -0.1;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::MiniBatchGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

void test_mini_batch_gradient_descent_rejects_momentum_one() {
    ml::OptimizerOptions options = make_mini_batch_options();
    options.momentum = 1.0;

    ml::LinearRegressionOptimizationProblem problem(2);
    ml::MiniBatchGradientDescent optimizer(options);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    static_cast<void>(optimizer.optimize(problem, X, y));
}

// ---- Optimization adapter tests ----

void test_linear_regression_optimization_problem_computes_loss_and_gradients() {
    ml::LinearRegressionOptimizationProblem problem(2);

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    const double loss = problem.loss(X, y);
    const ml::ParameterGradient gradient = problem.gradients(X, y);

    if (loss <= 0.0) {
        throw std::runtime_error("expected positive initial linear-regression loss");
    }

    if (gradient.weights_gradient.rows() != 2 || gradient.weights_gradient.cols() != 1) {
        throw std::runtime_error("unexpected linear-regression weight-gradient shape");
    }

    if (gradient.bias_gradient.size() != 1) {
        throw std::runtime_error("unexpected linear-regression bias-gradient shape");
    }
}

void test_logistic_regression_optimization_problem_computes_loss_and_gradients() {
    ml::LogisticRegressionOptimizationProblem problem(2);

    const ml::Matrix X = make_binary_classification_X();
    const ml::Matrix y = make_binary_classification_y_matrix();

    const double loss = problem.loss(X, y);
    const ml::ParameterGradient gradient = problem.gradients(X, y);

    if (loss <= 0.0) {
        throw std::runtime_error("expected positive initial logistic-regression loss");
    }

    if (gradient.weights_gradient.rows() != 2 || gradient.weights_gradient.cols() != 1) {
        throw std::runtime_error("unexpected logistic-regression weight-gradient shape");
    }

    if (gradient.bias_gradient.size() != 1) {
        throw std::runtime_error("unexpected logistic-regression bias-gradient shape");
    }
}

void test_softmax_regression_optimization_problem_computes_loss_and_gradients() {
    ml::SoftmaxRegressionOptimizationProblem problem(2, 3);

    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Matrix y = make_multiclass_classification_y_one_hot();

    const double loss = problem.loss(X, y);
    const ml::ParameterGradient gradient = problem.gradients(X, y);

    if (loss <= 0.0) {
        throw std::runtime_error("expected positive initial softmax-regression loss");
    }

    if (gradient.weights_gradient.rows() != 2 || gradient.weights_gradient.cols() != 3) {
        throw std::runtime_error("unexpected softmax-regression weight-gradient shape");
    }

    if (gradient.bias_gradient.size() != 3) {
        throw std::runtime_error("unexpected softmax-regression bias-gradient shape");
    }
}

void test_optimization_problem_set_parameters_updates_state() {
    ml::LinearRegressionOptimizationProblem problem(2);

    ml::Matrix weights(2, 1);
    weights << 2.0,
              -3.0;

    ml::Vector bias(1);
    bias << 1.0;

    problem.set_parameters(weights, bias);

    test::assert_matrix_almost_equal(
        problem.weights(),
        weights,
        "test_optimization_problem_set_parameters_updates_state weights"
    );

    test::assert_vector_almost_equal(
        problem.bias(),
        bias,
        "test_optimization_problem_set_parameters_updates_state bias"
    );
}

void test_optimization_problem_rejects_invalid_parameter_shapes() {
    ml::LinearRegressionOptimizationProblem problem(2);

    ml::Matrix weights(3, 1);
    weights << 1.0,
               2.0,
               3.0;

    ml::Vector bias(1);
    bias << 0.0;

    problem.set_parameters(weights, bias);
}

void test_logistic_optimization_problem_rejects_non_binary_targets() {
    ml::LogisticRegressionOptimizationProblem problem(2);

    const ml::Matrix X = make_binary_classification_X();

    ml::Matrix y(6, 1);
    y << 0.0,
         0.0,
         0.0,
         1.0,
         1.0,
         2.0;

    static_cast<void>(problem.loss(X, y));
}

void test_softmax_optimization_problem_rejects_invalid_target_shape() {
    ml::SoftmaxRegressionOptimizationProblem problem(2, 3);

    const ml::Matrix X = make_multiclass_classification_X();

    ml::Matrix y(9, 2);
    y.setZero();

    static_cast<void>(problem.loss(X, y));
}

void test_optimization_adapters_reject_lasso_for_now() {
    ml::LinearRegressionOptimizationProblem problem(
        2,
        ml::RegularizationConfig::lasso(0.1)
    );
}

// ---- Batch GD tests ----

void test_batch_gradient_descent_trains_linear_regression_problem() {
    ml::LinearRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    const double initial_loss = problem.loss(X, y);

    ml::BatchGradientDescent optimizer(make_batch_options());
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected Batch GD to reduce linear-regression loss");
    }

    if (final_loss > 1e-4) {
        throw std::runtime_error(
            "expected small final linear-regression loss, got " + std::to_string(final_loss)
        );
    }

    if (history.iterations_run == 0 || history.epochs_run == 0) {
        throw std::runtime_error("expected non-empty Batch GD history");
    }

    if (history.optimizer_name != "BatchGradientDescent") {
        throw std::runtime_error("expected Batch GD history optimizer name");
    }
}

void test_batch_gradient_descent_trains_logistic_regression_problem() {
    ml::LogisticRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_binary_classification_X();
    const ml::Matrix y = make_binary_classification_y_matrix();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_batch_options();
    options.learning_rate = 0.1;
    options.max_epochs = 10000;
    options.max_iterations = 10000;

    ml::BatchGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected Batch GD to reduce logistic-regression loss");
    }

    if (final_loss > 0.2) {
        throw std::runtime_error(
            "expected small final logistic-regression loss, got " + std::to_string(final_loss)
        );
    }

    if (history.iterations_run == 0 || history.epochs_run == 0) {
        throw std::runtime_error("expected non-empty Batch GD history");
    }
}

void test_batch_gradient_descent_trains_softmax_regression_problem() {
    ml::SoftmaxRegressionOptimizationProblem problem(2, 3);
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Matrix y = make_multiclass_classification_y_one_hot();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_batch_options();
    options.learning_rate = 0.1;
    options.max_epochs = 20000;
    options.max_iterations = 20000;

    ml::BatchGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected Batch GD to reduce softmax-regression loss");
    }

    if (final_loss > 0.2) {
        throw std::runtime_error(
            "expected small final softmax-regression loss, got " + std::to_string(final_loss)
        );
    }

    if (history.iterations_run == 0 || history.epochs_run == 0) {
        throw std::runtime_error("expected non-empty Batch GD history");
    }
}

// ---- Batch GD with momentum tests ----

void test_batch_gradient_descent_with_momentum_trains_linear_regression_problem() {
    ml::LinearRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_batch_options();
    options.learning_rate = 0.02;
    options.momentum = 0.9;
    options.max_epochs = 5000;
    options.max_iterations = 5000;

    ml::BatchGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected Batch GD with momentum to reduce linear-regression loss");
    }

    if (final_loss > 1e-4) {
        throw std::runtime_error(
            "expected small final linear-regression loss with Batch GD momentum, got " +
            std::to_string(final_loss)
        );
    }

    test::assert_almost_equal(
        history.momentum,
        0.9,
        "test_batch_gradient_descent_with_momentum_trains_linear_regression_problem momentum"
    );
}

// ---- SGD tests ----

void test_stochastic_gradient_descent_trains_linear_regression_problem() {
    ml::LinearRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_sgd_options();
    options.learning_rate = 0.01;
    options.max_epochs = 3000;
    options.max_iterations = 100000;

    ml::StochasticGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected SGD to reduce linear-regression loss");
    }

    if (final_loss > 1e-3) {
        throw std::runtime_error(
            "expected small final linear-regression loss with SGD, got " +
            std::to_string(final_loss)
        );
    }

    if (history.iterations_run == 0 || history.epochs_run == 0) {
        throw std::runtime_error("expected non-empty SGD history");
    }

    if (history.optimizer_name != "StochasticGradientDescent") {
        throw std::runtime_error("expected SGD history optimizer name");
    }

    if (history.batch_size != 1) {
        throw std::runtime_error("expected SGD batch_size == 1");
    }
}

void test_stochastic_gradient_descent_trains_logistic_regression_problem() {
    ml::LogisticRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_binary_classification_X();
    const ml::Matrix y = make_binary_classification_y_matrix();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_sgd_options();
    options.learning_rate = 0.05;
    options.max_epochs = 3000;
    options.max_iterations = 100000;

    ml::StochasticGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected SGD to reduce logistic-regression loss");
    }

    if (final_loss > 0.3) {
        throw std::runtime_error(
            "expected reasonably small logistic-regression loss with SGD, got " +
            std::to_string(final_loss)
        );
    }

    if (history.iterations_run == 0 || history.epochs_run == 0) {
        throw std::runtime_error("expected non-empty SGD history");
    }
}

void test_stochastic_gradient_descent_trains_softmax_regression_problem() {
    ml::SoftmaxRegressionOptimizationProblem problem(2, 3);
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Matrix y = make_multiclass_classification_y_one_hot();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_sgd_options();
    options.learning_rate = 0.05;
    options.max_epochs = 5000;
    options.max_iterations = 100000;

    ml::StochasticGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected SGD to reduce softmax-regression loss");
    }

    if (final_loss > 0.3) {
        throw std::runtime_error(
            "expected reasonably small softmax-regression loss with SGD, got " +
            std::to_string(final_loss)
        );
    }

    if (history.iterations_run == 0 || history.epochs_run == 0) {
        throw std::runtime_error("expected non-empty SGD history");
    }
}

// ---- SGD with momentum tests ----

void test_stochastic_gradient_descent_with_momentum_trains_logistic_regression_problem() {
    ml::LogisticRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_binary_classification_X();
    const ml::Matrix y = make_binary_classification_y_matrix();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_sgd_options();
    options.learning_rate = 0.02;
    options.momentum = 0.9;
    options.max_epochs = 3000;
    options.max_iterations = 100000;
    options.random_seed = 123;
    options.shuffle = true;

    ml::StochasticGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected SGD with momentum to reduce logistic-regression loss");
    }

    if (final_loss > 0.3) {
        throw std::runtime_error(
            "expected reasonably small logistic-regression loss with SGD momentum, got " +
            std::to_string(final_loss)
        );
    }

    test::assert_almost_equal(
        history.momentum,
        0.9,
        "test_stochastic_gradient_descent_with_momentum_trains_logistic_regression_problem momentum"
    );
}

// ---- Mini-batch GD tests ----

void test_mini_batch_gradient_descent_trains_linear_regression_problem() {
    ml::LinearRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_mini_batch_options();
    options.learning_rate = 0.03;
    options.batch_size = 2;
    options.max_epochs = 4000;
    options.max_iterations = 100000;

    ml::MiniBatchGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected Mini-batch GD to reduce linear-regression loss");
    }

    if (final_loss > 1e-3) {
        throw std::runtime_error(
            "expected small final linear-regression loss with Mini-batch GD, got " +
            std::to_string(final_loss)
        );
    }

    if (history.iterations_run == 0 || history.epochs_run == 0) {
        throw std::runtime_error("expected non-empty Mini-batch GD history");
    }

    if (history.optimizer_name != "MiniBatchGradientDescent") {
        throw std::runtime_error("expected Mini-batch GD history optimizer name");
    }

    if (history.batch_size != 2) {
        throw std::runtime_error("expected Mini-batch GD history batch_size == 2");
    }
}

void test_mini_batch_gradient_descent_trains_logistic_regression_problem() {
    ml::LogisticRegressionOptimizationProblem problem(2);
    const ml::Matrix X = make_binary_classification_X();
    const ml::Matrix y = make_binary_classification_y_matrix();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_mini_batch_options();
    options.learning_rate = 0.05;
    options.batch_size = 2;
    options.max_epochs = 4000;
    options.max_iterations = 100000;

    ml::MiniBatchGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected Mini-batch GD to reduce logistic-regression loss");
    }

    if (final_loss > 0.3) {
        throw std::runtime_error(
            "expected reasonably small logistic-regression loss with Mini-batch GD, got " +
            std::to_string(final_loss)
        );
    }

    if (history.iterations_run == 0 || history.epochs_run == 0) {
        throw std::runtime_error("expected non-empty Mini-batch GD history");
    }
}

void test_mini_batch_gradient_descent_trains_softmax_regression_problem() {
    ml::SoftmaxRegressionOptimizationProblem problem(2, 3);
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Matrix y = make_multiclass_classification_y_one_hot();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_mini_batch_options();
    options.learning_rate = 0.05;
    options.batch_size = 3;
    options.max_epochs = 5000;
    options.max_iterations = 100000;

    ml::MiniBatchGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected Mini-batch GD to reduce softmax-regression loss");
    }

    if (final_loss > 0.3) {
        throw std::runtime_error(
            "expected reasonably small softmax-regression loss with Mini-batch GD, got " +
            std::to_string(final_loss)
        );
    }

    if (history.iterations_run == 0 || history.epochs_run == 0) {
        throw std::runtime_error("expected non-empty Mini-batch GD history");
    }

    if (history.batch_size != 3) {
        throw std::runtime_error("expected Mini-batch GD history batch_size == 3");
    }
}

void test_mini_batch_gradient_descent_is_reproducible_with_same_seed() {
    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    ml::OptimizerOptions options = make_mini_batch_options();
    options.learning_rate = 0.03;
    options.batch_size = 2;
    options.max_epochs = 100;
    options.max_iterations = 100000;
    options.random_seed = 123;
    options.shuffle = true;

    ml::LinearRegressionOptimizationProblem problem_a(2);
    ml::LinearRegressionOptimizationProblem problem_b(2);

    const ml::MiniBatchGradientDescent optimizer(options);

    static_cast<void>(optimizer.optimize(problem_a, X, y));
    static_cast<void>(optimizer.optimize(problem_b, X, y));

    test::assert_matrix_almost_equal(
        problem_a.weights(),
        problem_b.weights(),
        "test_mini_batch_gradient_descent_is_reproducible_with_same_seed weights"
    );

    test::assert_vector_almost_equal(
        problem_a.bias(),
        problem_b.bias(),
        "test_mini_batch_gradient_descent_is_reproducible_with_same_seed bias"
    );
}

// ---- Mini-batch GD with momentum tests ----

void test_mini_batch_gradient_descent_with_momentum_trains_softmax_regression_problem() {
    ml::SoftmaxRegressionOptimizationProblem problem(2, 3);
    const ml::Matrix X = make_multiclass_classification_X();
    const ml::Matrix y = make_multiclass_classification_y_one_hot();

    const double initial_loss = problem.loss(X, y);

    ml::OptimizerOptions options = make_mini_batch_options();
    options.learning_rate = 0.02;
    options.momentum = 0.9;
    options.batch_size = 3;
    options.max_epochs = 5000;
    options.max_iterations = 100000;
    options.random_seed = 123;
    options.shuffle = true;

    ml::MiniBatchGradientDescent optimizer(options);
    const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

    const double final_loss = problem.loss(X, y);

    if (final_loss >= initial_loss) {
        throw std::runtime_error("expected Mini-batch GD with momentum to reduce softmax-regression loss");
    }

    if (final_loss > 0.4) {
        throw std::runtime_error(
            "expected reasonably small softmax-regression loss with Mini-batch GD momentum, got " +
            std::to_string(final_loss)
        );
    }

    test::assert_almost_equal(
        history.momentum,
        0.9,
        "test_mini_batch_gradient_descent_with_momentum_trains_softmax_regression_problem momentum"
    );
}

// ---- Optimization experiment export tests ----

void test_experiment_exports_batch_vs_sgd_vs_mini_batch_comparison() {
    ensure_phase5_output_dir_exists();

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    std::vector<OptimizationExperimentResult> results;

    {
        ml::LinearRegressionOptimizationProblem problem(2);

        ml::OptimizerOptions options = make_batch_options();
        options.learning_rate = 0.05;
        options.max_epochs = 2000;
        options.max_iterations = 2000;

        const ml::BatchGradientDescent optimizer(options);
        const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

        results.push_back(make_optimization_experiment_result(
            "batch_vs_sgd_vs_mini_batch",
            "linear_regression_synthetic",
            false,
            history
        ));
    }

    {
        ml::LinearRegressionOptimizationProblem problem(2);

        ml::OptimizerOptions options = make_sgd_options();
        options.learning_rate = 0.01;
        options.max_epochs = 1000;
        options.max_iterations = 100000;
        options.random_seed = 42;
        options.shuffle = true;

        const ml::StochasticGradientDescent optimizer(options);
        const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

        results.push_back(make_optimization_experiment_result(
            "batch_vs_sgd_vs_mini_batch",
            "linear_regression_synthetic",
            false,
            history
        ));
    }

    {
        ml::LinearRegressionOptimizationProblem problem(2);

        ml::OptimizerOptions options = make_mini_batch_options();
        options.learning_rate = 0.03;
        options.batch_size = 2;
        options.max_epochs = 1000;
        options.max_iterations = 100000;
        options.random_seed = 42;
        options.shuffle = true;

        const ml::MiniBatchGradientDescent optimizer(options);
        const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

        results.push_back(make_optimization_experiment_result(
            "batch_vs_sgd_vs_mini_batch",
            "linear_regression_synthetic",
            false,
            history
        ));
    }

    export_optimization_results_csv(
        results,
        k_phase5_output_dir + "/batch_vs_sgd_vs_mini_batch.csv"
    );

    export_optimization_results_txt(
        results,
        "Batch GD vs SGD vs Mini-batch GD",
        k_phase5_output_dir + "/batch_vs_sgd_vs_mini_batch.txt"
    );

    if (!std::filesystem::exists(k_phase5_output_dir + "/batch_vs_sgd_vs_mini_batch.csv")) {
        throw std::runtime_error("expected batch_vs_sgd_vs_mini_batch.csv to exist");
    }

    if (!std::filesystem::exists(k_phase5_output_dir + "/batch_vs_sgd_vs_mini_batch.txt")) {
        throw std::runtime_error("expected batch_vs_sgd_vs_mini_batch.txt to exist");
    }
}

void test_experiment_exports_momentum_comparison() {
    ensure_phase5_output_dir_exists();

    const ml::Matrix X = make_binary_classification_X();
    const ml::Matrix y = make_binary_classification_y_matrix();

    std::vector<OptimizationExperimentResult> results;

    for (double momentum : {0.0, 0.9}) {
        ml::LogisticRegressionOptimizationProblem problem(2);

        ml::OptimizerOptions options = make_mini_batch_options();
        options.learning_rate = 0.03;
        options.momentum = momentum;
        options.batch_size = 2;
        options.max_epochs = 3000;
        options.max_iterations = 100000;
        options.random_seed = 42;
        options.shuffle = true;

        const ml::MiniBatchGradientDescent optimizer(options);
        const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

        results.push_back(make_optimization_experiment_result(
            "momentum_comparison",
            "binary_classification_synthetic",
            false,
            history
        ));
    }

    export_optimization_results_csv(
        results,
        k_phase5_output_dir + "/momentum_comparison.csv"
    );

    export_optimization_results_txt(
        results,
        "Momentum Comparison",
        k_phase5_output_dir + "/momentum_comparison.txt"
    );

    if (!std::filesystem::exists(k_phase5_output_dir + "/momentum_comparison.csv")) {
        throw std::runtime_error("expected momentum_comparison.csv to exist");
    }

    if (!std::filesystem::exists(k_phase5_output_dir + "/momentum_comparison.txt")) {
        throw std::runtime_error("expected momentum_comparison.txt to exist");
    }
}

void test_experiment_exports_learning_rate_comparison() {
    ensure_phase5_output_dir_exists();

    const ml::Matrix X = make_linear_X();
    const ml::Matrix y = make_linear_y_matrix();

    std::vector<OptimizationExperimentResult> results;

    for (double learning_rate : {0.001, 0.01, 0.05}) {
        ml::LinearRegressionOptimizationProblem problem(2);

        ml::OptimizerOptions options = make_batch_options();
        options.learning_rate = learning_rate;
        options.max_epochs = 2000;
        options.max_iterations = 2000;

        const ml::BatchGradientDescent optimizer(options);
        const ml::TrainingHistory history = optimizer.optimize(problem, X, y);

        results.push_back(make_optimization_experiment_result(
            "learning_rate_comparison",
            "linear_regression_synthetic",
            false,
            history
        ));
    }

    export_optimization_results_csv(
        results,
        k_phase5_output_dir + "/learning_rate_comparison.csv"
    );

    export_optimization_results_txt(
        results,
        "Learning Rate Comparison",
        k_phase5_output_dir + "/learning_rate_comparison.txt"
    );

    if (!std::filesystem::exists(k_phase5_output_dir + "/learning_rate_comparison.csv")) {
        throw std::runtime_error("expected learning_rate_comparison.csv to exist");
    }

    if (!std::filesystem::exists(k_phase5_output_dir + "/learning_rate_comparison.txt")) {
        throw std::runtime_error("expected learning_rate_comparison.txt to exist");
    }
}

void test_experiment_exports_scaled_vs_unscaled_comparison() {
    ensure_phase5_output_dir_exists();

    const ml::Matrix X_unscaled = make_poorly_scaled_linear_X();
    const ml::Matrix X_scaled = make_manually_scaled_linear_X();
    const ml::Matrix y = make_poorly_scaled_linear_y_matrix();

    std::vector<OptimizationExperimentResult> results;

    {
        ml::LinearRegressionOptimizationProblem problem(2);

        ml::OptimizerOptions options = make_batch_options();
        options.learning_rate = 1e-8;
        options.max_epochs = 2000;
        options.max_iterations = 2000;

        const ml::BatchGradientDescent optimizer(options);
        const ml::TrainingHistory history = optimizer.optimize(problem, X_unscaled, y);

        results.push_back(make_optimization_experiment_result(
            "scaled_vs_unscaled",
            "poorly_scaled_linear_regression",
            false,
            history
        ));
    }

    {
        ml::LinearRegressionOptimizationProblem problem(2);

        ml::OptimizerOptions options = make_batch_options();
        options.learning_rate = 0.03;
        options.max_epochs = 2000;
        options.max_iterations = 2000;

        const ml::BatchGradientDescent optimizer(options);
        const ml::TrainingHistory history = optimizer.optimize(problem, X_scaled, y);

        results.push_back(make_optimization_experiment_result(
            "scaled_vs_unscaled",
            "poorly_scaled_linear_regression",
            true,
            history
        ));
    }

    export_optimization_results_csv(
        results,
        k_phase5_output_dir + "/scaled_vs_unscaled_comparison.csv"
    );

    export_optimization_results_txt(
        results,
        "Scaled vs Unscaled Optimization Behavior",
        k_phase5_output_dir + "/scaled_vs_unscaled_comparison.txt"
    );

    if (!std::filesystem::exists(k_phase5_output_dir + "/scaled_vs_unscaled_comparison.csv")) {
        throw std::runtime_error("expected scaled_vs_unscaled_comparison.csv to exist");
    }

    if (!std::filesystem::exists(k_phase5_output_dir + "/scaled_vs_unscaled_comparison.txt")) {
        throw std::runtime_error("expected scaled_vs_unscaled_comparison.txt to exist");
    }
}

// ---- Test runners ----

void run_training_history_tests() {
    std::cout << "\n[Phase 5.0] Training history tests\n\n";

    test::expect_no_throw(
        "TrainingHistory computes loss summary values",
        test_training_history_computes_loss_summary_values
    );

    test::expect_invalid_argument(
        "TrainingHistory rejects summary values without losses",
        test_training_history_rejects_summary_values_without_losses
    );

    test::expect_no_throw(
        "optimization_stop_reason_name returns expected value",
        test_optimization_stop_reason_name_returns_expected_value
    );

    test::expect_no_throw(
        "training_history_summary contains key fields",
        test_training_history_summary_contains_key_fields
    );

    test::expect_no_throw(
        "BatchGradientDescent records gradient norms",
        test_batch_gradient_descent_records_gradient_norms
    );

    test::expect_no_throw(
        "StochasticGradientDescent records gradient norms",
        test_stochastic_gradient_descent_records_gradient_norms
    );

    test::expect_no_throw(
        "MiniBatchGradientDescent records gradient norms",
        test_mini_batch_gradient_descent_records_gradient_norms
    );
}

void run_optimization_interface_tests() {
    std::cout << "\n[Phase 5.1] Optimization interface tests\n\n";

    test::expect_no_throw(
        "BatchGradientDescent reports name and options",
        test_batch_gradient_descent_reports_name_and_options
    );

    test::expect_no_throw(
        "StochasticGradientDescent reports name and options",
        test_stochastic_gradient_descent_reports_name_and_options
    );

    test::expect_no_throw(
        "MiniBatchGradientDescent reports name and options",
        test_mini_batch_gradient_descent_reports_name_and_options
    );

    test::expect_invalid_argument(
        "BatchGradientDescent rejects invalid learning rate",
        test_batch_gradient_descent_rejects_invalid_learning_rate
    );

    test::expect_invalid_argument(
        "StochasticGradientDescent rejects invalid learning rate",
        test_stochastic_gradient_descent_rejects_invalid_learning_rate
    );

    test::expect_invalid_argument(
        "MiniBatchGradientDescent rejects invalid learning rate",
        test_mini_batch_gradient_descent_rejects_invalid_learning_rate
    );

    test::expect_invalid_argument(
        "MiniBatchGradientDescent rejects invalid batch size",
        test_mini_batch_gradient_descent_rejects_invalid_batch_size
    );

    test::expect_invalid_argument(
        "BatchGradientDescent rejects negative momentum",
        test_batch_gradient_descent_rejects_negative_momentum
    );

    test::expect_invalid_argument(
        "BatchGradientDescent rejects momentum >= 1",
        test_batch_gradient_descent_rejects_momentum_one
    );

    test::expect_invalid_argument(
        "StochasticGradientDescent rejects negative momentum",
        test_stochastic_gradient_descent_rejects_negative_momentum
    );

    test::expect_invalid_argument(
        "StochasticGradientDescent rejects momentum >= 1",
        test_stochastic_gradient_descent_rejects_momentum_one
    );

    test::expect_invalid_argument(
        "MiniBatchGradientDescent rejects negative momentum",
        test_mini_batch_gradient_descent_rejects_negative_momentum
    );

    test::expect_invalid_argument(
        "MiniBatchGradientDescent rejects momentum >= 1",
        test_mini_batch_gradient_descent_rejects_momentum_one
    );
}

void run_optimization_adapter_tests() {
    std::cout << "\n[Phase 5.2] Optimization adapter tests\n\n";

    test::expect_no_throw(
        "LinearRegressionOptimizationProblem computes loss and gradients",
        test_linear_regression_optimization_problem_computes_loss_and_gradients
    );

    test::expect_no_throw(
        "LogisticRegressionOptimizationProblem computes loss and gradients",
        test_logistic_regression_optimization_problem_computes_loss_and_gradients
    );

    test::expect_no_throw(
        "SoftmaxRegressionOptimizationProblem computes loss and gradients",
        test_softmax_regression_optimization_problem_computes_loss_and_gradients
    );

    test::expect_no_throw(
        "OptimizationProblem set_parameters updates state",
        test_optimization_problem_set_parameters_updates_state
    );

    test::expect_invalid_argument(
        "OptimizationProblem rejects invalid parameter shapes",
        test_optimization_problem_rejects_invalid_parameter_shapes
    );

    test::expect_invalid_argument(
        "Logistic optimization problem rejects non-binary targets",
        test_logistic_optimization_problem_rejects_non_binary_targets
    );

    test::expect_invalid_argument(
        "Softmax optimization problem rejects invalid target shape",
        test_softmax_optimization_problem_rejects_invalid_target_shape
    );

    test::expect_invalid_argument(
        "Optimization adapters reject Lasso for now",
        test_optimization_adapters_reject_lasso_for_now
    );
}

void run_batch_gradient_descent_tests() {
    std::cout << "\n[Phase 5.3] Batch Gradient Descent tests\n\n";

    test::expect_no_throw(
        "BatchGradientDescent trains LinearRegressionOptimizationProblem",
        test_batch_gradient_descent_trains_linear_regression_problem
    );

    test::expect_no_throw(
        "BatchGradientDescent trains LogisticRegressionOptimizationProblem",
        test_batch_gradient_descent_trains_logistic_regression_problem
    );

    test::expect_no_throw(
        "BatchGradientDescent trains SoftmaxRegressionOptimizationProblem",
        test_batch_gradient_descent_trains_softmax_regression_problem
    );

    test::expect_no_throw(
        "BatchGradientDescent with momentum trains LinearRegressionOptimizationProblem",
        test_batch_gradient_descent_with_momentum_trains_linear_regression_problem
    );
}

void run_stochastic_gradient_descent_tests() {
    std::cout << "\n[Phase 5.4] Stochastic Gradient Descent tests\n\n";

    test::expect_no_throw(
        "StochasticGradientDescent trains LinearRegressionOptimizationProblem",
        test_stochastic_gradient_descent_trains_linear_regression_problem
    );

    test::expect_no_throw(
        "StochasticGradientDescent trains LogisticRegressionOptimizationProblem",
        test_stochastic_gradient_descent_trains_logistic_regression_problem
    );

    test::expect_no_throw(
        "StochasticGradientDescent trains SoftmaxRegressionOptimizationProblem",
        test_stochastic_gradient_descent_trains_softmax_regression_problem
    );

    test::expect_no_throw(
        "StochasticGradientDescent with momentum trains LogisticRegressionOptimizationProblem",
        test_stochastic_gradient_descent_with_momentum_trains_logistic_regression_problem
    );
}

void run_mini_batch_gradient_descent_tests() {
    std::cout << "\n[Phase 5.5] Mini-batch Gradient Descent tests\n\n";

    test::expect_no_throw(
        "MiniBatchGradientDescent trains LinearRegressionOptimizationProblem",
        test_mini_batch_gradient_descent_trains_linear_regression_problem
    );

    test::expect_no_throw(
        "MiniBatchGradientDescent trains LogisticRegressionOptimizationProblem",
        test_mini_batch_gradient_descent_trains_logistic_regression_problem
    );

    test::expect_no_throw(
        "MiniBatchGradientDescent trains SoftmaxRegressionOptimizationProblem",
        test_mini_batch_gradient_descent_trains_softmax_regression_problem
    );

    test::expect_no_throw(
        "MiniBatchGradientDescent is reproducible with same seed",
        test_mini_batch_gradient_descent_is_reproducible_with_same_seed
    );

    test::expect_no_throw(
        "MiniBatchGradientDescent with momentum trains SoftmaxRegressionOptimizationProblem",
        test_mini_batch_gradient_descent_with_momentum_trains_softmax_regression_problem
    );
}

void run_optimization_experiment_export_tests() {
    std::cout << "\n[Phase 5.6] Optimization experiment export tests\n\n";

    test::expect_no_throw(
        "Experiment exports batch vs SGD vs mini-batch comparison",
        test_experiment_exports_batch_vs_sgd_vs_mini_batch_comparison
    );

    test::expect_no_throw(
        "Experiment exports momentum comparison",
        test_experiment_exports_momentum_comparison
    );

    test::expect_no_throw(
        "Experiment exports learning-rate comparison",
        test_experiment_exports_learning_rate_comparison
    );

    test::expect_no_throw(
        "Experiment exports scaled vs unscaled comparison",
        test_experiment_exports_scaled_vs_unscaled_comparison
    );
}

}  // namespace

namespace ml::experiments {

void run_phase5_optimization_sanity() {
    run_training_history_tests();
    run_optimization_interface_tests();
    run_optimization_adapter_tests();
    run_batch_gradient_descent_tests();
    run_stochastic_gradient_descent_tests();
    run_mini_batch_gradient_descent_tests();
    run_optimization_experiment_export_tests();
}

}  // namespace ml::experiments