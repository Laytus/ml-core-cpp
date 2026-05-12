#include "ml/workflows/hyperparameter_sweeps.hpp"

#include "ml/common/classification_metrics.hpp"
#include "ml/common/csv_dataset_loader.hpp"
#include "ml/common/math_ops.hpp"
#include "ml/common/multiclass_metrics.hpp"
#include "ml/common/regression_metrics.hpp"

#include "ml/distance/knn_classifier.hpp"
#include "ml/dl_bridge/mlp.hpp"
#include "ml/linear_models/logistic_regression.hpp"
#include "ml/linear_models/regularization.hpp"
#include "ml/trees/decision_tree.hpp"
#include "ml/trees/gradient_boosting.hpp"
#include "ml/trees/random_forest.hpp"
#include "ml/unsupervised/kmeans.hpp"
#include "ml/unsupervised/pca.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ml::workflows {
namespace {

struct SupervisedData {
    Matrix X;
    Vector y;
};

struct SupervisedSplit {
    SupervisedData train;
    SupervisedData test;
};

struct StandardizationStats {
    Vector means;
    Vector stddevs;
};

std::string double_to_string(double value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

std::string size_to_string(std::size_t value) {
    return std::to_string(value);
}

void validate_config(const HyperparameterSweepConfig& config) {
    if (config.output_root.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.output_root must not be empty");
    }

    if (config.regression_dataset_path.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.regression_dataset_path must not be empty");
    }

    if (config.binary_dataset_path.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.binary_dataset_path must not be empty");
    }

    if (config.multiclass_dataset_path.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.multiclass_dataset_path must not be empty");
    }

    if (config.unsupervised_dataset_path.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.unsupervised_dataset_path must not be empty");
    }

    if (config.regression_feature_columns.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.regression_feature_columns must not be empty");
    }

    if (config.binary_feature_columns.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.binary_feature_columns must not be empty");
    }

    if (config.multiclass_feature_columns.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.multiclass_feature_columns must not be empty");
    }

    if (config.unsupervised_feature_columns.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.unsupervised_feature_columns must not be empty");
    }

    if (config.regression_target_column.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.regression_target_column must not be empty");
    }

    if (config.binary_target_column.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.binary_target_column must not be empty");
    }

    if (config.multiclass_target_column.empty()) {
        throw std::invalid_argument("HyperparameterSweepConfig.multiclass_target_column must not be empty");
    }

    if (config.test_ratio <= 0.0 || config.test_ratio >= 1.0) {
        throw std::invalid_argument("HyperparameterSweepConfig.test_ratio must be in (0, 1)");
    }

    if (config.multiclass_num_classes < 2) {
        throw std::invalid_argument("HyperparameterSweepConfig.multiclass_num_classes must be at least 2");
    }
}

void validate_binary_labels(const Vector& y, const std::string& context) {
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (y(i) != 0.0 && y(i) != 1.0) {
            throw std::invalid_argument(context + ": expected binary labels encoded as 0/1");
        }
    }
}

void validate_multiclass_labels(
    const Vector& y,
    std::size_t num_classes,
    const std::string& context
) {
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double label = y(i);

        if (label < 0.0 || label >= static_cast<double>(num_classes)) {
            throw std::invalid_argument(context + ": label outside [0, num_classes)");
        }

        if (std::floor(label) != label) {
            throw std::invalid_argument(context + ": labels must be integer-valued");
        }
    }
}

SupervisedData take_rows(
    const Matrix& X,
    const Vector& y,
    const std::vector<Eigen::Index>& indices
) {
    SupervisedData result;
    result.X.resize(static_cast<Eigen::Index>(indices.size()), X.cols());
    result.y.resize(static_cast<Eigen::Index>(indices.size()));

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(indices.size()); ++i) {
        const Eigen::Index source_index = indices[static_cast<std::size_t>(i)];
        result.X.row(i) = X.row(source_index);
        result.y(i) = y(source_index);
    }

    return result;
}

Matrix take_rows(
    const Matrix& X,
    const std::vector<Eigen::Index>& indices
) {
    Matrix result;
    result.resize(static_cast<Eigen::Index>(indices.size()), X.cols());

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(indices.size()); ++i) {
        result.row(i) = X.row(indices[static_cast<std::size_t>(i)]);
    }

    return result;
}

SupervisedData maybe_limit_rows(
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

Matrix maybe_limit_rows(
    const Matrix& X,
    std::size_t max_rows,
    std::uint32_t seed
) {
    const std::size_t n_rows = static_cast<std::size_t>(X.rows());

    if (max_rows == 0 || n_rows <= max_rows) {
        return X;
    }

    std::vector<Eigen::Index> indices(n_rows);
    std::iota(indices.begin(), indices.end(), static_cast<Eigen::Index>(0));

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    indices.resize(max_rows);
    std::sort(indices.begin(), indices.end());

    return take_rows(X, indices);
}

SupervisedSplit make_train_test_split(
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

StandardizationStats fit_standardizer(const Matrix& X) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("fit_standardizer: X must be non-empty");
    }

    StandardizationStats stats;
    stats.means = X.colwise().mean();
    stats.stddevs.resize(X.cols());

    for (Eigen::Index col = 0; col < X.cols(); ++col) {
        const Vector centered = X.col(col).array() - stats.means(col);
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

void standardize_split_if_needed(
    SupervisedSplit& split,
    bool standardize_features
) {
    if (!standardize_features) {
        return;
    }

    const StandardizationStats stats = fit_standardizer(split.train.X);
    split.train.X = apply_standardizer(split.train.X, stats);
    split.test.X = apply_standardizer(split.test.X, stats);
}

Matrix standardize_matrix_if_needed(
    const Matrix& X,
    bool standardize_features
) {
    if (!standardize_features) {
        return X;
    }

    const StandardizationStats stats = fit_standardizer(X);
    return apply_standardizer(X, stats);
}

std::vector<std::pair<std::string, double>> regression_metrics(
    const Vector& predictions,
    const Vector& targets
) {
    return {
        {"mse", mean_squared_error(predictions, targets)},
        {"rmse", root_mean_squared_error(predictions, targets)},
        {"mae", mean_absolute_error(predictions, targets)},
        {"r2", r2_score(predictions, targets)}
    };
}

std::vector<std::pair<std::string, double>> binary_metrics(
    const Vector& predictions,
    const Vector& targets
) {
    return {
        {"accuracy", accuracy_score(predictions, targets)},
        {"precision", precision_score(predictions, targets)},
        {"recall", recall_score(predictions, targets)},
        {"f1", f1_score(predictions, targets)}
    };
}

std::vector<std::pair<std::string, double>> multiclass_metrics(
    const Vector& predictions,
    const Vector& targets,
    std::size_t num_classes
) {
    const Eigen::Index class_count = static_cast<Eigen::Index>(num_classes);

    return {
        {"accuracy", multiclass_accuracy_score(predictions, targets, class_count)},
        {"macro_precision", macro_precision(predictions, targets, class_count)},
        {"macro_recall", macro_recall(predictions, targets, class_count)},
        {"macro_f1", macro_f1(predictions, targets, class_count)}
    };
}

void append_sweep_rows(
    std::vector<HyperparameterSweepRow>& rows,
    const std::string& run_id,
    const std::string& workflow,
    const std::string& dataset,
    const std::string& model,
    const std::string& split,
    const std::vector<std::pair<std::string, std::string>>& params,
    const std::vector<std::pair<std::string, double>>& metrics
) {
    for (const auto& metric : metrics) {
        for (const auto& param : params) {
            rows.push_back({
                run_id,
                workflow,
                dataset,
                model,
                split,
                param.first,
                param.second,
                metric.first,
                metric.second
            });
        }
    }
}

std::vector<HyperparameterSweepRow> run_regression_gradient_boosting_sweep(
    const HyperparameterSweepConfig& config,
    const SupervisedSplit& split
) {
    std::vector<HyperparameterSweepRow> rows;

    for (const std::size_t n_estimators : config.gradient_boosting_n_estimators_values) {
        for (const double learning_rate : config.gradient_boosting_learning_rate_values) {
            GradientBoostingRegressorOptions options;
            options.n_estimators = n_estimators;
            options.learning_rate = learning_rate;
            options.max_depth = 2;
            options.min_samples_split = 20;
            options.min_samples_leaf = 10;
            options.random_seed = config.seed;

            GradientBoostingRegressor model(options);
            model.fit(split.train.X, split.train.y);

            const Vector predictions = model.predict(split.test.X);

            const std::string run_id =
                "gb_n" + size_to_string(n_estimators)
                + "_lr" + double_to_string(learning_rate);

            append_sweep_rows(
                rows,
                run_id,
                "regression",
                config.regression_dataset_name,
                "GradientBoostingRegressor",
                "test",
                {
                    {"n_estimators", size_to_string(n_estimators)},
                    {"learning_rate", double_to_string(learning_rate)}
                },
                regression_metrics(predictions, split.test.y)
            );
        }
    }

    return rows;
}

std::vector<HyperparameterSweepRow> run_binary_logistic_regression_sweep(
    const HyperparameterSweepConfig& config,
    const SupervisedSplit& split
) {
    std::vector<HyperparameterSweepRow> rows;

    for (const double learning_rate : config.logistic_regression_learning_rate_values) {
        for (const double lambda : config.logistic_regression_ridge_lambda_values) {
            LogisticRegressionOptions options;
            options.learning_rate = learning_rate;
            options.max_iterations = 300;
            options.tolerance = 1e-8;
            options.store_loss_history = false;

            if (lambda > 0.0) {
                options.regularization = RegularizationConfig::ridge(lambda);
            } else {
                options.regularization = RegularizationConfig::none();
            }

            LogisticRegression model(options);
            model.fit(split.train.X, split.train.y);

            const Vector predictions = model.predict_classes(split.test.X);

            const std::string run_id =
                "logistic_lr" + double_to_string(learning_rate)
                + "_ridge" + double_to_string(lambda);

            append_sweep_rows(
                rows,
                run_id,
                "binary_classification",
                config.binary_dataset_name,
                "LogisticRegression",
                "test",
                {
                    {"learning_rate", double_to_string(learning_rate)},
                    {"ridge_lambda", double_to_string(lambda)}
                },
                binary_metrics(predictions, split.test.y)
            );
        }
    }

    return rows;
}

std::vector<HyperparameterSweepRow> run_binary_decision_tree_sweep(
    const HyperparameterSweepConfig& config,
    const SupervisedSplit& split
) {
    std::vector<HyperparameterSweepRow> rows;

    for (const std::size_t max_depth : config.decision_tree_max_depth_values) {
        for (const std::size_t min_samples_leaf : config.decision_tree_min_samples_leaf_values) {
            DecisionTreeOptions options;
            options.max_depth = max_depth;
            options.min_samples_split = 20;
            options.min_samples_leaf = min_samples_leaf;
            options.min_impurity_decrease = 0.0;
            options.use_balanced_class_weight = true;
            options.random_seed = config.seed;

            DecisionTreeClassifier model(options);
            model.fit(split.train.X, split.train.y);

            const Vector predictions = model.predict(split.test.X);

            const std::string run_id =
                "decision_tree_depth" + size_to_string(max_depth)
                + "_leaf" + size_to_string(min_samples_leaf);

            append_sweep_rows(
                rows,
                run_id,
                "binary_classification",
                config.binary_dataset_name,
                "DecisionTreeClassifier",
                "test",
                {
                    {"max_depth", size_to_string(max_depth)},
                    {"min_samples_leaf", size_to_string(min_samples_leaf)}
                },
                binary_metrics(predictions, split.test.y)
            );
        }
    }

    return rows;
}

std::vector<HyperparameterSweepRow> run_binary_random_forest_sweep(
    const HyperparameterSweepConfig& config,
    const SupervisedSplit& split
) {
    std::vector<HyperparameterSweepRow> rows;

    for (const std::size_t n_estimators : config.random_forest_n_estimators_values) {
        for (const std::size_t max_features : config.random_forest_max_features_values) {
            DecisionTreeOptions tree_options;
            tree_options.max_depth = 4;
            tree_options.min_samples_split = 20;
            tree_options.min_samples_leaf = 10;
            tree_options.min_impurity_decrease = 0.0;
            tree_options.use_balanced_class_weight = true;
            tree_options.random_seed = config.seed;

            RandomForestOptions options;
            options.n_estimators = n_estimators;
            options.bootstrap = true;
            options.max_features = max_features;
            options.random_seed = config.seed;
            options.tree_options = tree_options;

            RandomForestClassifier model(options);
            model.fit(split.train.X, split.train.y);

            const Vector predictions = model.predict(split.test.X);

            const std::string run_id =
                "random_forest_trees" + size_to_string(n_estimators)
                + "_maxfeat" + size_to_string(max_features);

            append_sweep_rows(
                rows,
                run_id,
                "binary_classification",
                config.binary_dataset_name,
                "RandomForestClassifier",
                "test",
                {
                    {"n_estimators", size_to_string(n_estimators)},
                    {"max_features", size_to_string(max_features)}
                },
                binary_metrics(predictions, split.test.y)
            );
        }
    }

    return rows;
}

std::vector<HyperparameterSweepRow> run_binary_tiny_mlp_sweep(
    const HyperparameterSweepConfig& config,
    const SupervisedSplit& split
) {
    std::vector<HyperparameterSweepRow> rows;

    for (const std::size_t hidden_units : config.tiny_mlp_hidden_units_values) {
        for (const double learning_rate : config.tiny_mlp_learning_rate_values) {
            for (const std::size_t max_epochs : config.tiny_mlp_max_epochs_values) {
                TinyMLPBinaryClassifierOptions options;
                options.hidden_units = hidden_units;
                options.learning_rate = learning_rate;
                options.max_epochs = max_epochs;
                options.batch_size = 32;
                options.random_seed = config.seed;

                TinyMLPBinaryClassifier model(options);
                model.fit(split.train.X, split.train.y);

                const Vector predictions = model.predict(split.test.X);

                const std::string run_id =
                    "tiny_mlp_h" + size_to_string(hidden_units)
                    + "_lr" + double_to_string(learning_rate)
                    + "_e" + size_to_string(max_epochs);

                append_sweep_rows(
                    rows,
                    run_id,
                    "binary_classification",
                    config.binary_dataset_name,
                    "TinyMLPBinaryClassifier",
                    "test",
                    {
                        {"hidden_units", size_to_string(hidden_units)},
                        {"learning_rate", double_to_string(learning_rate)},
                        {"max_epochs", size_to_string(max_epochs)}
                    },
                    binary_metrics(predictions, split.test.y)
                );
            }
        }
    }

    return rows;
}

std::vector<HyperparameterSweepRow> run_multiclass_knn_sweep(
    const HyperparameterSweepConfig& config,
    const SupervisedSplit& split
) {
    std::vector<HyperparameterSweepRow> rows;

    for (const std::size_t k : config.knn_k_values) {
        KNNClassifierOptions options;
        options.k = k;
        options.distance_metric = DistanceMetric::Euclidean;

        KNNClassifier model(options);
        model.fit(split.train.X, split.train.y);

        const Vector predictions = model.predict(split.test.X);

        const std::string run_id = "knn_k" + size_to_string(k);

        append_sweep_rows(
            rows,
            run_id,
            "multiclass_classification",
            config.multiclass_dataset_name,
            "KNNClassifier",
            "test",
            {
                {"k", size_to_string(k)}
            },
            multiclass_metrics(predictions, split.test.y, config.multiclass_num_classes)
        );
    }

    return rows;
}

std::vector<HyperparameterSweepRow> run_unsupervised_pca_kmeans_sweep(
    const HyperparameterSweepConfig& config,
    const Matrix& X
) {
    std::vector<HyperparameterSweepRow> rows;

    for (const std::size_t num_components : config.pca_num_components_values) {
        PCAOptions options;
        options.num_components = num_components;

        PCA pca(options);
        pca.fit(X);

        std::vector<std::pair<std::string, double>> metrics;

        for (Eigen::Index i = 0; i < pca.explained_variance_ratio().size(); ++i) {
            metrics.push_back({
                "explained_variance_ratio_" + std::to_string(i + 1),
                pca.explained_variance_ratio()(i)
            });
        }

        double cumulative = 0.0;
        for (Eigen::Index i = 0; i < pca.explained_variance_ratio().size(); ++i) {
            cumulative += pca.explained_variance_ratio()(i);
        }

        metrics.push_back({"cumulative_explained_variance_ratio", cumulative});

        append_sweep_rows(
            rows,
            "pca_components_" + size_to_string(num_components),
            "unsupervised",
            config.unsupervised_dataset_name,
            "PCA",
            "full",
            {
                {"num_components", size_to_string(num_components)}
            },
            metrics
        );
    }

    for (const std::size_t num_clusters : config.kmeans_num_clusters_values) {
        KMeansOptions options;
        options.num_clusters = num_clusters;
        options.max_iterations = 50;
        options.tolerance = 1e-6;

        KMeans model(options);
        model.fit(X);

        append_sweep_rows(
            rows,
            "kmeans_k" + size_to_string(num_clusters),
            "unsupervised",
            config.unsupervised_dataset_name,
            "KMeans",
            "full",
            {
                {"num_clusters", size_to_string(num_clusters)}
            },
            {
                {"inertia", model.inertia()},
                {"iterations", static_cast<double>(model.num_iterations())}
            }
        );
    }

    return rows;
}

}  // namespace

HyperparameterSweepSummary run_hyperparameter_sweeps(
    const HyperparameterSweepConfig& config
) {
    validate_config(config);

    common::CsvDatasetLoader loader;
    OutputWriter writer(config.output_root);

    const auto regression_loaded = loader.load_supervised(
        config.regression_dataset_path.string(),
        config.regression_feature_columns,
        config.regression_target_column
    );

    const SupervisedData regression_limited = maybe_limit_rows(
        regression_loaded.X,
        regression_loaded.y,
        config.regression_max_rows,
        config.seed
    );

    SupervisedSplit regression_split = make_train_test_split(
        regression_limited.X,
        regression_limited.y,
        config.test_ratio,
        config.shuffle,
        config.seed
    );

    standardize_split_if_needed(regression_split, config.standardize_features);

    const std::vector<HyperparameterSweepRow> regression_rows =
        run_regression_gradient_boosting_sweep(config, regression_split);

    writer.write_hyperparameter_sweep("regression", regression_rows, false);

    const auto binary_loaded = loader.load_supervised(
        config.binary_dataset_path.string(),
        config.binary_feature_columns,
        config.binary_target_column
    );

    validate_binary_labels(binary_loaded.y, "run_hyperparameter_sweeps");

    const SupervisedData binary_limited = maybe_limit_rows(
        binary_loaded.X,
        binary_loaded.y,
        config.binary_max_rows,
        config.seed
    );

    SupervisedSplit binary_split = make_train_test_split(
        binary_limited.X,
        binary_limited.y,
        config.test_ratio,
        config.shuffle,
        config.seed
    );

    standardize_split_if_needed(binary_split, config.standardize_features);

    std::vector<HyperparameterSweepRow> binary_rows;

    const std::vector<HyperparameterSweepRow> logistic_rows =
        run_binary_logistic_regression_sweep(config, binary_split);

    const std::vector<HyperparameterSweepRow> decision_tree_rows =
        run_binary_decision_tree_sweep(config, binary_split);

    const std::vector<HyperparameterSweepRow> random_forest_rows =
        run_binary_random_forest_sweep(config, binary_split);

    const std::vector<HyperparameterSweepRow> tiny_mlp_rows =
        run_binary_tiny_mlp_sweep(config, binary_split);

    binary_rows.insert(binary_rows.end(), logistic_rows.begin(), logistic_rows.end());
    binary_rows.insert(binary_rows.end(), decision_tree_rows.begin(), decision_tree_rows.end());
    binary_rows.insert(binary_rows.end(), random_forest_rows.begin(), random_forest_rows.end());
    binary_rows.insert(binary_rows.end(), tiny_mlp_rows.begin(), tiny_mlp_rows.end());

    writer.write_hyperparameter_sweep("binary-classification", binary_rows, false);

    const auto multiclass_loaded = loader.load_supervised(
        config.multiclass_dataset_path.string(),
        config.multiclass_feature_columns,
        config.multiclass_target_column
    );

    validate_multiclass_labels(
        multiclass_loaded.y,
        config.multiclass_num_classes,
        "run_hyperparameter_sweeps"
    );

    const SupervisedData multiclass_limited = maybe_limit_rows(
        multiclass_loaded.X,
        multiclass_loaded.y,
        config.multiclass_max_rows,
        config.seed
    );

    SupervisedSplit multiclass_split = make_train_test_split(
        multiclass_limited.X,
        multiclass_limited.y,
        config.test_ratio,
        config.shuffle,
        config.seed
    );

    standardize_split_if_needed(multiclass_split, config.standardize_features);

    const std::vector<HyperparameterSweepRow> multiclass_rows =
        run_multiclass_knn_sweep(config, multiclass_split);

    writer.write_hyperparameter_sweep("multiclass-classification", multiclass_rows, false);

    const auto unsupervised_loaded = loader.load_unsupervised(
        config.unsupervised_dataset_path.string(),
        config.unsupervised_feature_columns
    );

    Matrix X_unsupervised = maybe_limit_rows(
        unsupervised_loaded.X,
        config.unsupervised_max_rows,
        config.seed
    );

    X_unsupervised = standardize_matrix_if_needed(
        X_unsupervised,
        config.standardize_features
    );

    const std::vector<HyperparameterSweepRow> unsupervised_rows =
        run_unsupervised_pca_kmeans_sweep(config, X_unsupervised);

    writer.write_hyperparameter_sweep("unsupervised", unsupervised_rows, false);

    std::cout << "\n[Hyperparameter sweep workflows]\n";
    std::cout << "Regression sweep rows: " << regression_rows.size() << "\n";
    std::cout << "Binary sweep rows: " << binary_rows.size() << "\n";
    std::cout << "Multiclass sweep rows: " << multiclass_rows.size() << "\n";
    std::cout << "Unsupervised sweep rows: " << unsupervised_rows.size() << "\n";

    return {
        static_cast<std::size_t>(regression_limited.X.rows()),
        static_cast<std::size_t>(binary_limited.X.rows()),
        static_cast<std::size_t>(multiclass_limited.X.rows()),
        static_cast<std::size_t>(X_unsupervised.rows()),
        regression_rows.size(),
        binary_rows.size(),
        multiclass_rows.size(),
        unsupervised_rows.size()
    };
}

}  // namespace ml::workflows