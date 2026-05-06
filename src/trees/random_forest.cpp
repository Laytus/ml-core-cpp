#include "ml/trees/random_forest.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/trees/bootstrap.hpp"
#include "ml/trees/tree_builder.hpp"

#include <cmath>
#include <map>
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

std::size_t infer_num_classes_from_targets(
    const Vector& y
) {
    validate_non_empty_vector(y, "infer_num_classes_from_targets");

    int max_label = -1;

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double value = y(i);
        const double rounded = std::round(value);

        if (std::abs(value - rounded) > 1e-12) {
            throw std::invalid_argument(
                "infer_num_classes_from_targets: class labels must be integer-valued"
            );
        }
        
        if (rounded < 0.0) {
            throw std::invalid_argument(
                "infer_num_classes_from_targets: class labels must be non-negative"
            );
        }

        max_label = std::max(
            max_label,
            static_cast<int>(rounded)
        );
    }

    if (max_label < 0) {
        throw std::invalid_argument(
            "infer_num_classes_from_targets: at least one class is required"
        );
    }

    return static_cast<std::size_t>(max_label + 1);
}

}  // namespace

void validate_random_forest_options(
    const RandomForestOptions& options,
    const std::string& context
) {
    if (options.n_estimators == 0) {
        throw std::invalid_argument(
            context + ": n_estimators must be at least 1"
        );
    }
    
    if (
        options.max_features.has_value() &&
        options.max_features.value() == 0
    ) {
        throw std::invalid_argument(
            context + ": max_features must be at least 1 when provided"
        );
    }

    validate_decision_tree_options(
        options.tree_options,
        context + " tree_options"
    );
}

RandomForestClassifier::RandomForestClassifier(RandomForestOptions options)
    : options_{options} {
    validate_random_forest_options(
        options_,
        "RandomForestClassifier"
    );
}

void RandomForestClassifier::fit(
    const Matrix& X,
    const Vector& y
) {
    validate_random_forest_options(options_, "RandomForestClassifier::fit");

    validate_non_empty_matrix(X, "RandomForestClassifier::fit");
    validate_non_empty_vector(y, "RandomForestClassifier::fit");
    validate_same_number_of_rows(X, y, "RandomForestClassifier::fit");
    validate_finite_matrix_values(X, "RandomForestClassifier::fit");

    if (
        options_.max_features.has_value() &&
        options_.max_features.value() > static_cast<std::size_t>(X.cols())
    ) {
        throw std::invalid_argument(
            "RandomForestClassifier::fit: max_features must be less than or equal to the number of features"
        );
    }

    num_features_ = X.cols();
    num_classes_ = infer_num_classes_from_targets(y);

    trees_.clear();
    trees_.reserve(options_.n_estimators);

    for (std::size_t tree_index = 0; tree_index < options_.n_estimators; ++tree_index) {
        const unsigned int tree_seed = options_.random_seed + static_cast<unsigned int>(tree_index);

        const BootstrapSample sample = options_.bootstrap
            ? make_bootstrap_sample(X, y, tree_seed)
            : make_full_sample(X, y);

        DecisionTreeOptions tree_options = options_.tree_options;

        tree_options.random_seed = tree_seed;

        if (options_.max_features.has_value()) {
            tree_options.max_features = options_.max_features;
        }

        DecisionTreeClassifier tree(tree_options);
        tree.fit(
            sample.X,
            sample.y
        );

        trees_.push_back(std::move(tree));
    }

    fitted_ = true;
}

Vector RandomForestClassifier::predict(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "RandomForestClassifier::predict: model must be fitted before predictions"
        );
    }

    validate_non_empty_matrix(X, "RandomForestClassifier::predict");

    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "RandomForestClassifier::predict: X feature count must match training feature count"
        );
    }

    validate_finite_matrix_values(X, "RandomForestClassifier::predict");

    Vector predictions(X.rows());

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        Matrix samples_as_matrix(1, X.cols());
        samples_as_matrix.row(0) = X.row(i);

        const Vector tree_predictions = predict_tree_votes_for_sample(samples_as_matrix);

        predictions(i) = majority_class(tree_predictions);
    }

    return predictions;
}

Matrix RandomForestClassifier::predict_proba(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "RandomForestClassifier::predict_proba: model must be fitted before predictions"
        );
    }

    validate_non_empty_matrix(X, "RandomForestClassifier::predict_proba");

    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "RandomForestClassifier::predict_proba: X feature count must match training feature count"
        );
    }

    validate_finite_matrix_values(X, "RandomForestClassifier::predict_proba");

    Matrix probabilities(
        X.rows(),
        static_cast<Eigen::Index>(num_classes_)
    );

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        Matrix samples_as_matrix(1, X.cols());
        samples_as_matrix.row(0) = X.row(i);

        const Vector tree_predictions = predict_tree_votes_for_sample(samples_as_matrix);
        
        const Vector sample_probabilities = vote_probabilities(tree_predictions);

        probabilities.row(i) = sample_probabilities.transpose();
    }

    return probabilities;
}

bool RandomForestClassifier::is_fitted() const {
    return fitted_;
}

const RandomForestOptions& RandomForestClassifier::options() const {
    return options_;
}

std::size_t RandomForestClassifier::num_classes() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "RandomForestClassifier::num_classes: model must be fitted"
        );
    }

    return num_classes_;
}

std::size_t RandomForestClassifier::num_trees() const {
    return trees_.size();
}

Vector RandomForestClassifier::predict_tree_votes_for_sample(
    const Matrix& samples_as_matrix
) const {
    Vector tree_predictions(static_cast<Eigen::Index>(trees_.size()));

    for (std::size_t tree_index = 0; tree_index < trees_.size(); ++tree_index) {
        const Vector prediction = trees_[tree_index].predict(samples_as_matrix);

        tree_predictions(static_cast<Eigen::Index>(tree_index)) = prediction(0);
    }

    return tree_predictions;
}

double RandomForestClassifier::majority_vote(
    const Vector& tree_predictions
) const {
    validate_non_empty_vector(tree_predictions, "RandomForestClassifier::majority_vote");

    std::map<int, std::size_t> vote_counts;

    for (Eigen::Index i = 0; i < tree_predictions.size(); ++i) {
        const double value = tree_predictions(i);
        const double rounded = std::round(value);

        if (std::abs(value - rounded) > 1e-12 || rounded < 0.0) {
            throw std::runtime_error(
                "RandomForestClassifier::majority_vote: tree prediction must be a non-negative class index"
            );
        }

        const int label = static_cast<int>(rounded);

        if (static_cast<std::size_t>(label) >= num_classes_) {
            throw std::runtime_error(
                "RandomForestClassifier::majority_vote: tree prediction is out of class range"
            );
        }

        vote_counts[label] += 1;
    }

    int best_label = -1;
    std::size_t best_count = 0;

    for (const auto& [label, count] : vote_counts) {
        if (count > best_count) {
            best_label = label;
            best_count = count;
        }
    }

    return static_cast<double>(best_label);
}

Vector RandomForestClassifier::vote_probabilities(
    const Vector& tree_predictions
) const {
    validate_non_empty_vector(tree_predictions, "RandomForestClassifier::vote_probabilities");

    Vector probabilities = Vector::Zero(static_cast<Eigen::Index>(num_classes_));

    for (Eigen::Index i = 0; i < tree_predictions.size(); ++i) {
        const double value = tree_predictions(i);
        const double rounded = std::round(value);

        if (std::abs(value - rounded) > 1e-12 || rounded < 0.0) {
            throw std::runtime_error(
                "RandomForestClassifier::vote_probabilities: tree prediction must be a non-negative class index"
            );
        }

        const std::size_t label = static_cast<std::size_t>(rounded);

        if (label >= num_classes_) {
            throw std::runtime_error(
                "RandomForestClassifier::vote_probabilities: tree prediction is out of class range"
            );
        }

        probabilities(static_cast<Eigen::Index>(label)) += 1.0;
    }

    probabilities /= static_cast<double>(tree_predictions.size());

    return probabilities;
}


}  // namespace ml