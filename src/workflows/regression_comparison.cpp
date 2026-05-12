#include "ml/workflows/regression_comparison.hpp"

#include "ml/common/csv_dataset_loader.hpp"
#include "ml/common/math_ops.hpp"
#include "ml/common/regression_metrics.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml::workflows {
namespace {

struct RegressionData {
    Matrix X;
    Vector y;
};

struct RegressionSplit {
    RegressionData train;
    RegressionData test;
};

struct StandardizationStats {
    Vector means;
    Vector stddevs;
};

void validate_config(const RegressionComparisonConfig& config) {
    if (config.dataset_path.empty()) {
        throw std::invalid_argument("RegressionComparisonConfig.dataset_path must not be empty");
    }

    if (config.output_root.empty()) {
        throw std::invalid_argument("RegressionComparisonConfig.output_root must not be empty");
    }

    if (config.workflow_folder.empty()) {
        throw std::invalid_argument("RegressionComparisonConfig.workflow_folder must not be empty");
    }

    if (config.workflow_name.empty()) {
        throw std::invalid_argument("RegressionComparisonConfig.workflow_name must not be empty");
    }

    if (config.dataset_name.empty()) {
        throw std::invalid_argument("RegressionComparisonConfig.dataset_name must not be empty");
    }

    if (config.feature_columns.empty()) {
        throw std::invalid_argument("RegressionComparisonConfig.feature_columns must not be empty");
    }

    if (config.target_column.empty()) {
        throw std::invalid_argument("RegressionComparisonConfig.target_column must not be empty");
    }

    if (config.test_ratio <= 0.0 || config.test_ratio >= 1.0) {
        throw std::invalid_argument("RegressionComparisonConfig.test_ratio must be in (0, 1)");
    }
}

RegressionData take_rows(
    const Matrix& X,
    const Vector& y,
    const std::vector<Eigen::Index>& indices
) {
    RegressionData result;
    result.X.resize(static_cast<Eigen::Index>(indices.size()), X.cols());
    result.y.resize(static_cast<Eigen::Index>(indices.size()));

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(indices.size()); ++i) {
        const Eigen::Index source_index = indices[static_cast<std::size_t>(i)];
        result.X.row(i) = X.row(source_index);
        result.y(i) = y(source_index);
    }

    return result;
}

RegressionData maybe_limit_rows(
    const Matrix& X,
    const Vector& y,
    std::size_t max_rows,
    std::uint32_t seed
) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("maybe_limit_rows: X row count must match y size");
    }

    const std::size_t n_rows = static_cast<std::size_t>(X.rows());

    if (max_rows == 0 || n_rows <= max_rows) {
        return {X, y};
    }

    std::vector<Eigen::Index> indices(n_rows);
    std::iota(indices.begin(), indices.end(), static_cast<Eigen::Index>(0));

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    indices.resize(max_rows);
    std::sort(indices.begin(), indices.end());

    return take_rows(X, y, indices);
}

RegressionSplit make_train_test_split(
    const Matrix& X,
    const Vector& y,
    double test_ratio,
    bool shuffle,
    std::uint32_t seed
) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("make_train_test_split: X row count must match y size");
    }

    if (X.rows() < 2) {
        throw std::invalid_argument("make_train_test_split: dataset must contain at least 2 rows");
    }

    const std::size_t n_rows = static_cast<std::size_t>(X.rows());
    const std::size_t test_rows = static_cast<std::size_t>(
        std::round(static_cast<double>(n_rows) * test_ratio)
    );

    if (test_rows == 0 || test_rows >= n_rows) {
        throw std::invalid_argument("make_train_test_split: invalid test size");
    }

    std::vector<Eigen::Index> indices(n_rows);
    std::iota(indices.begin(), indices.end(), static_cast<Eigen::Index>(0));

    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::vector<Eigen::Index> test_indices(
        indices.begin(),
        indices.begin() + static_cast<std::ptrdiff_t>(test_rows)
    );

    std::vector<Eigen::Index> train_indices(
        indices.begin() + static_cast<std::ptrdiff_t>(test_rows),
        indices.end()
    );

    return {
        take_rows(X, y, train_indices),
        take_rows(X, y, test_indices)
    };
}

StandardizationStats fit_standardizer(const Matrix& X_train) {
    if (X_train.rows() == 0 || X_train.cols() == 0) {
        throw std::invalid_argument("fit_standardizer: X_train must be non-empty");
    }

    StandardizationStats stats;
    stats.means = X_train.colwise().mean();
    stats.stddevs.resize(X_train.cols());

    for (Eigen::Index col = 0; col < X_train.cols(); ++col) {
        const Vector centered = X_train.col(col).array() - stats.means(col);
        const double variance = centered.array().square().mean();
        const double stddev = std::sqrt(variance);

        stats.stddevs(col) = stddev < 1e-12 ? 1.0 : stddev;
    }

    return stats;
}

Matrix apply_standardizer(
    const Matrix& X,
    const StandardizationStats& stats
) {
    if (X.cols() != stats.means.size() || X.cols() != stats.stddevs.size()) {
        throw std::invalid_argument("apply_standardizer: incompatible feature count");
    }

    Matrix result = X;

    for (Eigen::Index col = 0; col < X.cols(); ++col) {
        result.col(col).array() -= stats.means(col);
        result.col(col).array() /= stats.stddevs(col);
    }

    return result;
}

MetricsRow make_metric_row(
    const RegressionComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const std::string& metric,
    double value
) {
    return {
        run_id,
        config.workflow_name,
        config.dataset_name,
        model,
        config.split_name,
        metric,
        value
    };
}

std::vector<MetricsRow> build_regression_metric_rows(
    const RegressionComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const Vector& y_true,
    const Vector& y_pred
) {
    return {
        make_metric_row(config, run_id, model, "mse", mean_squared_error(y_pred, y_true)),
        make_metric_row(config, run_id, model, "rmse", root_mean_squared_error(y_pred, y_true)),
        make_metric_row(config, run_id, model, "mae", mean_absolute_error(y_pred, y_true)),
        make_metric_row(config, run_id, model, "r2", r2_score(y_pred, y_true))
    };
}

std::vector<RegressionPredictionRow> build_prediction_rows(
    const RegressionComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const Vector& y_true,
    const Vector& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("build_prediction_rows: y_true and y_pred size mismatch");
    }

    std::vector<RegressionPredictionRow> rows;
    rows.reserve(static_cast<std::size_t>(y_true.size()));

    for (Eigen::Index i = 0; i < y_true.size(); ++i) {
        rows.push_back({
            run_id,
            static_cast<std::size_t>(i),
            config.workflow_name,
            config.dataset_name,
            model,
            config.split_name,
            y_true(i),
            y_pred(i),
            y_pred(i) - y_true(i)
        });
    }

    return rows;
}

void append_rows(
    std::vector<MetricsRow>& destination,
    const std::vector<MetricsRow>& source
) {
    destination.insert(destination.end(), source.begin(), source.end());
}

void append_rows(
    std::vector<RegressionPredictionRow>& destination,
    const std::vector<RegressionPredictionRow>& source
) {
    destination.insert(destination.end(), source.begin(), source.end());
}

std::vector<LossHistoryRow> build_linear_regression_loss_rows(
    const RegressionComparisonConfig& config,
    const std::string& run_id,
    const LinearRegression& model
) {
    std::vector<LossHistoryRow> rows;
    const auto& losses = model.training_history().losses;

    rows.reserve(losses.size());

    for (std::size_t i = 0; i < losses.size(); ++i) {
        rows.push_back({
            run_id,
            config.workflow_name,
            config.dataset_name,
            "LinearRegression",
            "train",
            i,
            losses[i]
        });
    }

    return rows;
}

std::vector<LossHistoryRow> build_gradient_boosting_loss_rows(
    const RegressionComparisonConfig& config,
    const std::string& run_id,
    const GradientBoostingRegressor& model
) {
    std::vector<LossHistoryRow> rows;
    const auto& losses = model.training_loss_history();

    rows.reserve(losses.size());

    for (std::size_t i = 0; i < losses.size(); ++i) {
        rows.push_back({
            run_id,
            config.workflow_name,
            config.dataset_name,
            "GradientBoostingRegressor",
            "train",
            i,
            losses[i]
        });
    }

    return rows;
}

}  // namespace

RegressionComparisonSummary run_regression_comparison(
    const RegressionComparisonConfig& config
) {
    validate_config(config);

    common::CsvDatasetLoader loader;

    const auto loaded = loader.load_supervised(
        config.dataset_path.string(),
        config.feature_columns,
        config.target_column
    );

    const RegressionData limited = maybe_limit_rows(
        loaded.X,
        loaded.y,
        config.max_rows,
        config.seed
    );

    RegressionSplit split = make_train_test_split(
        limited.X,
        limited.y,
        config.test_ratio,
        config.shuffle,
        config.seed
    );

    if (config.standardize_features) {
        const StandardizationStats stats = fit_standardizer(split.train.X);
        split.train.X = apply_standardizer(split.train.X, stats);
        split.test.X = apply_standardizer(split.test.X, stats);
    }

    OutputWriter writer(config.output_root);

    std::vector<MetricsRow> all_metrics;
    std::vector<RegressionPredictionRow> all_predictions;
    std::vector<LossHistoryRow> all_loss_rows;

    {
        const std::string run_id = "linear_regression_baseline";
        const std::string model_name = "LinearRegression";

        LinearRegression model(config.linear_regression_options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict(split.test.X);

        append_rows(
            all_metrics,
            build_regression_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );

        if (config.export_loss_history) {
            const auto loss_rows = build_linear_regression_loss_rows(config, run_id, model);
            all_loss_rows.insert(all_loss_rows.end(), loss_rows.begin(), loss_rows.end());
        }
    }

    {
        const std::string run_id = "ridge_regression_baseline";
        const std::string model_name = "RidgeRegression";

        LinearRegressionOptions ridge_options = config.linear_regression_options;
        ridge_options.regularization = RegularizationConfig::ridge(0.01);

        LinearRegression model(ridge_options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict(split.test.X);

        append_rows(
            all_metrics,
            build_regression_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );

        if (config.export_loss_history) {
            const auto& losses = model.training_history().losses;

            std::vector<LossHistoryRow> loss_rows;
            loss_rows.reserve(losses.size());

            for (std::size_t i = 0; i < losses.size(); ++i) {
                loss_rows.push_back({
                    run_id,
                    config.workflow_name,
                    config.dataset_name,
                    model_name,
                    "train",
                    i,
                    losses[i]
                });
            }

            all_loss_rows.insert(all_loss_rows.end(), loss_rows.begin(), loss_rows.end());
        }
    }

    {
        const std::string run_id = "decision_tree_regressor_baseline";
        const std::string model_name = "DecisionTreeRegressor";

        DecisionTreeRegressor model(config.decision_tree_options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict(split.test.X);

        append_rows(
            all_metrics,
            build_regression_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );
    }

    {
        const std::string run_id = "gradient_boosting_regressor_baseline";
        const std::string model_name = "GradientBoostingRegressor";

        GradientBoostingRegressor model(config.gradient_boosting_options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict(split.test.X);

        append_rows(
            all_metrics,
            build_regression_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );

        if (config.export_loss_history) {
            const auto loss_rows = build_gradient_boosting_loss_rows(config, run_id, model);
            all_loss_rows.insert(all_loss_rows.end(), loss_rows.begin(), loss_rows.end());
        }
    }

    writer.write_metrics(config.workflow_folder, all_metrics, false);

    if (config.export_predictions) {
        writer.write_regression_predictions(config.workflow_folder, all_predictions, false);
    }

    if (config.export_loss_history && !all_loss_rows.empty()) {
        writer.write_loss_history(config.workflow_folder, all_loss_rows, false);
    }

    std::cout << "\n[Regression comparison workflow]\n";
    std::cout << "Dataset: " << config.dataset_name << "\n";
    std::cout << "Rows used: " << static_cast<std::size_t>(limited.X.rows()) << "\n";
    std::cout << "Train rows: " << static_cast<std::size_t>(split.train.X.rows()) << "\n";
    std::cout << "Test rows: " << static_cast<std::size_t>(split.test.X.rows()) << "\n";
    std::cout << "Features: " << static_cast<std::size_t>(limited.X.cols()) << "\n";
    std::cout << "Output folder: " << (config.output_root / config.workflow_folder) << "\n";

    return {
        static_cast<std::size_t>(limited.X.rows()),
        static_cast<std::size_t>(split.train.X.rows()),
        static_cast<std::size_t>(split.test.X.rows()),
        static_cast<std::size_t>(limited.X.cols()),
        all_metrics
    };
}

}  // namespace ml::workflows