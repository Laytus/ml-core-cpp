#include "ml/workflows/binary_classification_comparison.hpp"

#include "ml/common/classification_metrics.hpp"
#include "ml/common/csv_dataset_loader.hpp"

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

struct BinaryData {
    Matrix X;
    Vector y;
};

struct BinarySplit {
    BinaryData train;
    BinaryData test;
};

struct StandardizationStats {
    Vector means;
    Vector stddevs;
};

void validate_config(const BinaryClassificationComparisonConfig& config) {
    if (config.dataset_path.empty()) {
        throw std::invalid_argument("BinaryClassificationComparisonConfig.dataset_path must not be empty");
    }

    if (config.output_root.empty()) {
        throw std::invalid_argument("BinaryClassificationComparisonConfig.output_root must not be empty");
    }

    if (config.workflow_folder.empty()) {
        throw std::invalid_argument("BinaryClassificationComparisonConfig.workflow_folder must not be empty");
    }

    if (config.workflow_name.empty()) {
        throw std::invalid_argument("BinaryClassificationComparisonConfig.workflow_name must not be empty");
    }

    if (config.dataset_name.empty()) {
        throw std::invalid_argument("BinaryClassificationComparisonConfig.dataset_name must not be empty");
    }

    if (config.feature_columns.empty()) {
        throw std::invalid_argument("BinaryClassificationComparisonConfig.feature_columns must not be empty");
    }

    if (config.target_column.empty()) {
        throw std::invalid_argument("BinaryClassificationComparisonConfig.target_column must not be empty");
    }

    if (config.test_ratio <= 0.0 || config.test_ratio >= 1.0) {
        throw std::invalid_argument("BinaryClassificationComparisonConfig.test_ratio must be in (0, 1)");
    }
}

void validate_binary_labels(const Vector& y, const std::string& context) {
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (y(i) != 0.0 && y(i) != 1.0) {
            throw std::invalid_argument(
                context + ": expected binary labels encoded as 0/1"
            );
        }
    }
}

BinaryData take_rows(
    const Matrix& X,
    const Vector& y,
    const std::vector<Eigen::Index>& indices
) {
    BinaryData result;
    result.X.resize(static_cast<Eigen::Index>(indices.size()), X.cols());
    result.y.resize(static_cast<Eigen::Index>(indices.size()));

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(indices.size()); ++i) {
        const Eigen::Index source_index = indices[static_cast<std::size_t>(i)];
        result.X.row(i) = X.row(source_index);
        result.y(i) = y(source_index);
    }

    return result;
}

BinaryData maybe_limit_rows(
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

BinarySplit make_train_test_split(
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

Vector map_predictions_to_binary_zero_one(const Vector& predictions) {
    Vector mapped(predictions.size());

    for (Eigen::Index i = 0; i < predictions.size(); ++i) {
        if (predictions(i) == 0.0 || predictions(i) == 1.0) {
            mapped(i) = predictions(i);
        } else if (predictions(i) < 0.0) {
            mapped(i) = 0.0;
        } else {
            mapped(i) = 1.0;
        }
    }

    return mapped;
}

MetricsRow make_metric_row(
    const BinaryClassificationComparisonConfig& config,
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

std::vector<MetricsRow> build_binary_metric_rows(
    const BinaryClassificationComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const Vector& y_true,
    const Vector& y_pred
) {
    return {
        make_metric_row(config, run_id, model, "accuracy", accuracy_score(y_pred, y_true)),
        make_metric_row(config, run_id, model, "precision", precision_score(y_pred, y_true)),
        make_metric_row(config, run_id, model, "recall", recall_score(y_pred, y_true)),
        make_metric_row(config, run_id, model, "f1", f1_score(y_pred, y_true))
    };
}

std::vector<ClassificationPredictionRow> build_prediction_rows(
    const BinaryClassificationComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const Vector& y_true,
    const Vector& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("build_prediction_rows: y_true and y_pred size mismatch");
    }

    std::vector<ClassificationPredictionRow> rows;
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
            y_true(i) == y_pred(i) ? 1 : 0
        });
    }

    return rows;
}

std::vector<BinaryProbabilityRow> build_probability_rows_from_positive_class(
    const BinaryClassificationComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const Vector& y_true,
    const Vector& probability_class_1
) {
    if (y_true.size() != probability_class_1.size()) {
        throw std::invalid_argument("build_probability_rows_from_positive_class: size mismatch");
    }

    std::vector<BinaryProbabilityRow> rows;
    rows.reserve(static_cast<std::size_t>(y_true.size()));

    for (Eigen::Index i = 0; i < y_true.size(); ++i) {
        const double p1 = probability_class_1(i);

        rows.push_back({
            run_id,
            static_cast<std::size_t>(i),
            config.workflow_name,
            config.dataset_name,
            model,
            config.split_name,
            y_true(i),
            1.0 - p1,
            p1
        });
    }

    return rows;
}

std::vector<BinaryProbabilityRow> build_probability_rows_from_matrix(
    const BinaryClassificationComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const Vector& y_true,
    const Matrix& probabilities
) {
    if (probabilities.rows() != y_true.size()) {
        throw std::invalid_argument("build_probability_rows_from_matrix: row count mismatch");
    }

    if (probabilities.cols() < 2) {
        throw std::invalid_argument("build_probability_rows_from_matrix: expected at least 2 probability columns");
    }

    std::vector<BinaryProbabilityRow> rows;
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
            probabilities(i, 0),
            probabilities(i, 1)
        });
    }

    return rows;
}

std::vector<DecisionScoreRow> build_decision_score_rows(
    const BinaryClassificationComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const Vector& y_true,
    const Vector& scores
) {
    if (y_true.size() != scores.size()) {
        throw std::invalid_argument("build_decision_score_rows: y_true and scores size mismatch");
    }

    std::vector<DecisionScoreRow> rows;
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
            scores(i)
        });
    }

    return rows;
}

std::vector<LossHistoryRow> build_loss_history_rows(
    const BinaryClassificationComparisonConfig& config,
    const std::string& run_id,
    const std::string& model,
    const std::vector<double>& losses
) {
    std::vector<LossHistoryRow> rows;
    rows.reserve(losses.size());

    for (std::size_t i = 0; i < losses.size(); ++i) {
        rows.push_back({
            run_id,
            config.workflow_name,
            config.dataset_name,
            model,
            "train",
            i,
            losses[i]
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
    std::vector<ClassificationPredictionRow>& destination,
    const std::vector<ClassificationPredictionRow>& source
) {
    destination.insert(destination.end(), source.begin(), source.end());
}

void append_rows(
    std::vector<BinaryProbabilityRow>& destination,
    const std::vector<BinaryProbabilityRow>& source
) {
    destination.insert(destination.end(), source.begin(), source.end());
}

void append_rows(
    std::vector<DecisionScoreRow>& destination,
    const std::vector<DecisionScoreRow>& source
) {
    destination.insert(destination.end(), source.begin(), source.end());
}

void append_rows(
    std::vector<LossHistoryRow>& destination,
    const std::vector<LossHistoryRow>& source
) {
    destination.insert(destination.end(), source.begin(), source.end());
}

}  // namespace

BinaryClassificationComparisonSummary run_binary_classification_comparison(
    const BinaryClassificationComparisonConfig& config
) {
    validate_config(config);

    common::CsvDatasetLoader loader;

    const auto loaded = loader.load_supervised(
        config.dataset_path.string(),
        config.feature_columns,
        config.target_column
    );

    validate_binary_labels(loaded.y, "run_binary_classification_comparison");

    const BinaryData limited = maybe_limit_rows(
        loaded.X,
        loaded.y,
        config.max_rows,
        config.seed
    );

    BinarySplit split = make_train_test_split(
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
    std::vector<ClassificationPredictionRow> all_predictions;
    std::vector<BinaryProbabilityRow> all_probabilities;
    std::vector<DecisionScoreRow> all_decision_scores;
    std::vector<LossHistoryRow> all_loss_rows;

    {
        const std::string run_id = "logistic_regression_baseline";
        const std::string model_name = "LogisticRegression";

        LogisticRegression model(config.logistic_regression_options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict_classes(split.test.X);
        const Vector probabilities = model.predict_proba(split.test.X);
        const Vector scores = model.logits(split.test.X);

        append_rows(
            all_metrics,
            build_binary_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );

        if (config.export_probabilities) {
            append_rows(
                all_probabilities,
                build_probability_rows_from_positive_class(config, run_id, model_name, split.test.y, probabilities)
            );
        }

        if (config.export_decision_scores) {
            append_rows(
                all_decision_scores,
                build_decision_score_rows(config, run_id, model_name, split.test.y, scores)
            );
        }

        if (config.export_loss_history) {
            append_rows(
                all_loss_rows,
                build_loss_history_rows(
                    config,
                    run_id,
                    model_name,
                    model.training_history().losses
                )
            );
        }
    }

    {
        const std::string run_id = "linear_svm_baseline";
        const std::string model_name = "LinearSVM";

        LinearSVM model(config.linear_svm_options);
        model.fit(split.train.X, split.train.y);

        const Vector raw_predictions = model.predict(split.test.X);
        const Vector predictions = map_predictions_to_binary_zero_one(raw_predictions);
        const Vector scores = model.decision_function(split.test.X);

        append_rows(
            all_metrics,
            build_binary_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );

        if (config.export_decision_scores) {
            append_rows(
                all_decision_scores,
                build_decision_score_rows(config, run_id, model_name, split.test.y, scores)
            );
        }

        if (config.export_loss_history) {
            append_rows(
                all_loss_rows,
                build_loss_history_rows(
                    config,
                    run_id,
                    model_name,
                    model.training_loss_history()
                )
            );
        }
    }

    {
        const std::string run_id = "gaussian_naive_bayes_baseline";
        const std::string model_name = "GaussianNaiveBayes";

        GaussianNaiveBayes model(config.gaussian_naive_bayes_options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict(split.test.X);
        const Matrix probabilities = model.predict_proba(split.test.X);

        append_rows(
            all_metrics,
            build_binary_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );

        if (config.export_probabilities) {
            append_rows(
                all_probabilities,
                build_probability_rows_from_matrix(config, run_id, model_name, split.test.y, probabilities)
            );
        }
    }

    {
        const std::string run_id = "decision_tree_classifier_baseline";
        const std::string model_name = "DecisionTreeClassifier";

        DecisionTreeClassifier model(config.decision_tree_options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict(split.test.X);

        append_rows(
            all_metrics,
            build_binary_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );
    }

    {
        const std::string run_id = "random_forest_classifier_baseline";
        const std::string model_name = "RandomForestClassifier";

        RandomForestClassifier model(config.random_forest_options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict(split.test.X);
        const Matrix probabilities = model.predict_proba(split.test.X);

        append_rows(
            all_metrics,
            build_binary_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );

        if (config.export_probabilities) {
            append_rows(
                all_probabilities,
                build_probability_rows_from_matrix(config, run_id, model_name, split.test.y, probabilities)
            );
        }
    }

    {
        const std::string run_id = "tiny_mlp_binary_classifier_baseline";
        const std::string model_name = "TinyMLPBinaryClassifier";

        TinyMLPBinaryClassifier model(config.tiny_mlp_options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict(split.test.X);
        const Vector probabilities = model.predict_proba(split.test.X);

        append_rows(
            all_metrics,
            build_binary_metric_rows(config, run_id, model_name, split.test.y, predictions)
        );

        append_rows(
            all_predictions,
            build_prediction_rows(config, run_id, model_name, split.test.y, predictions)
        );

        if (config.export_probabilities) {
            append_rows(
                all_probabilities,
                build_probability_rows_from_positive_class(
                    config,
                    run_id,
                    model_name,
                    split.test.y,
                    probabilities
                )
            );
        }

        if (config.export_loss_history) {
            append_rows(
                all_loss_rows,
                build_loss_history_rows(
                    config,
                    run_id,
                    model_name,
                    model.loss_history()
                )
            );
        }
    }

    writer.write_metrics(config.workflow_folder, all_metrics, false);

    if (config.export_predictions) {
        writer.write_classification_predictions(config.workflow_folder, all_predictions, false);
    }

    if (config.export_probabilities && !all_probabilities.empty()) {
        writer.write_binary_probabilities(config.workflow_folder, all_probabilities, false);
    }

    if (config.export_decision_scores && !all_decision_scores.empty()) {
        writer.write_decision_scores(config.workflow_folder, all_decision_scores, false);
    }

    if (config.export_loss_history && !all_loss_rows.empty()) {
        writer.write_loss_history(config.workflow_folder, all_loss_rows, false);
    }

    std::cout << "\n[Binary classification comparison workflow]\n";
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