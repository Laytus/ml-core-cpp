#include "ml/trees/gradient_boosting.hpp"

#include "ml/common/classification_metrics.hpp"
#include "ml/common/shape_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

namespace {

void validate_finite_matrix_values(
    const Matrix& X,
    const std::string& context
) {
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            if (!std::isfinite(X(i, j))) {
                throw std::invalid_argument(
                    context + ": X values must be finite"
                );
            }
        }
    }
}

void validate_finite_vector_values(
    const Vector& y,
    const std::string& context
) {
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y(i))) {
            throw std::invalid_argument(
                context + ": y values must be finite"
            );
        }
    }
}

double mean_value(
    const Vector& y
) {
    validate_non_empty_vector(y, "mean_value");

    return y.sum() / static_cast<double>(y.size());
}

double mean_squared_error_local(
    const Vector& predictions,
    const Vector& targets
) {
    validate_same_size(predictions, targets, "mean_squared_error_local");

    double total = 0.0;

    for (Eigen::Index i = 0; i < predictions.size(); ++i) {
        const double residual = predictions(i) - targets(i);
        total += residual * residual;
    }

    return total / static_cast<double>(predictions.size());
}

}  // namespace

void validate_gradient_boosting_regressor_options(
    const GradientBoostingRegressorOptions& options,
    const std::string& context
) {
    if (options.n_estimators == 0) {
        throw std::invalid_argument(
            context + ": n_estimators must be at least 1"
        );
    }

    if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0) {
        throw std::invalid_argument(
            context + ": learning_rate must be finite and positive"
        );
    }

    if (options.max_depth == 0) {
        throw std::invalid_argument(
            context + ": max_depth must be at least 1"
        );
    }

    if (options.min_samples_split < 2) {
        throw std::invalid_argument(
            context + ": min_samples_split must be at least 2"
        );
    }

    if (options.min_samples_leaf == 0) {
        throw std::invalid_argument(
            context + ": min_samples_leaf must be at least 1"
        );
    }
}

GradientBoostingRegressor::GradientBoostingRegressor(
    GradientBoostingRegressorOptions options
)
    : options_{options} {
    validate_gradient_boosting_regressor_options(
        options_,
        "GradientBoostingRegressor"
    );
}

void GradientBoostingRegressor::fit(
    const Matrix& X,
    const Vector& y
) {
    validate_gradient_boosting_regressor_options(
        options_,
        "GradientBoostingRegressor::fit"
    );

    validate_non_empty_matrix(X, "GradientBoostingRegressor::fit");
    validate_non_empty_vector(y, "GradientBoostingRegressor::fit");
    validate_same_number_of_rows(X, y, "GradientBoostingRegressor::fit");

    validate_finite_matrix_values(X, "GradientBoostingRegressor::fit");
    validate_finite_vector_values(y, "GradientBoostingRegressor::fit");

    num_features_ = X.cols();
    initial_prediction_ = mean_value(y);

    trees_.clear();
    trees_.reserve(options_.n_estimators);

    training_loss_history_.clear();
    training_loss_history_.reserve(options_.n_estimators);

    Vector current_predictions =
        Vector::Constant(
            y.size(),
            initial_prediction_
        );

    for (
        std::size_t estimator_index = 0;
        estimator_index < options_.n_estimators;
        ++estimator_index
    ) {
        const Vector residuals = y - current_predictions;

        RegressionTreeOptions tree_options;
        tree_options.max_depth = options_.max_depth;
        tree_options.min_samples_split = options_.min_samples_split;
        tree_options.min_samples_leaf = options_.min_samples_leaf;
        tree_options.min_error_decrease = 0.0;

        DecisionTreeRegressor tree(tree_options);
        tree.fit(X, residuals);

        const Vector correction = tree.predict(X);

        current_predictions = current_predictions + options_.learning_rate * correction;

        training_loss_history_.push_back(
            mean_squared_error_local(
                current_predictions,
                y
            )
        );

        trees_.push_back(std::move(tree));
    }

    fitted_ = true;
}

Vector GradientBoostingRegressor::predict(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "GradientBoostingRegressor::predict: model must be fitted before prediction"
        );
    }

    validate_non_empty_matrix(X, "GradientBoostingRegressor::predict");

    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "GradientBoostingRegressor::predict: X feature count must match training feature count"
        );
    }

    validate_finite_matrix_values(X, "GradientBoostingRegressor::predict");

    Vector predictions =
        Vector::Constant(
            X.rows(),
            initial_prediction_
        );

    for (const DecisionTreeRegressor& tree : trees_) {
        predictions = predictions + options_.learning_rate * tree.predict(X);
    }

    return predictions;
}

bool GradientBoostingRegressor::is_fitted() const {
    return fitted_;
}

const GradientBoostingRegressorOptions& GradientBoostingRegressor::options() const {
    return options_;
}

double GradientBoostingRegressor::initial_prediction() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "GradientBoostingRegressor::initial_prediction: model must be fitted"
        );
    }

    return initial_prediction_;
}

std::size_t GradientBoostingRegressor::num_trees() const {
    return trees_.size();
}

const std::vector<double>& GradientBoostingRegressor::training_loss_history() const {
    return training_loss_history_;
}

}  // namespace ml