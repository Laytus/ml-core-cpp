#include "ml/probabilistic/naive_bayes.hpp"

#include "ml/common/shape_validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace ml {

namespace {

constexpr double k_two_pi = 6.28318530717958647692;

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

void validate_class_targets(
    const Vector& y,
    const std::string& context
) {
    validate_non_empty_vector(y, context);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double value = y(i);
        const double rounded = std::round(value);

        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                context + ": target values must be finite"
            );
        }

        if (std::abs(value - rounded) > 1e-12) {
            throw std::invalid_argument(
                context + ": class labels must be integer-valued"
            );
        }

        if (rounded < 0.0) {
            throw std::invalid_argument(
                context + ": class labels must be non-negative"
            );
        }
    }
}

Vector sorted_unique_classes(
    const Vector& y
) {
    std::vector<double> values;

    values.reserve(
        static_cast<std::size_t>(y.size())
    );

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        values.push_back(
            std::round(y(i))
        );
    }

    std::sort(
        values.begin(),
        values.end()
    );

    values.erase(
        std::unique(
            values.begin(),
            values.end()
        ),

        values.end()
    );

    Vector classes(
        static_cast<Eigen::Index>(values.size())
    );

    for (Eigen::Index i = 0; i < classes.size(); ++i) {
        classes(i) = values[static_cast<std::size_t>(i)];
    }

    return classes;
}

Eigen::Index class_index_for_label(
    const Vector& classes,
    double label
) {
    const double rounded = std::round(label);

    for (Eigen::Index i = 0; i < classes.size(); ++i) {
        if (classes(i) == rounded) {
            return i;
        }
    }

    throw std::invalid_argument(
        "class_index_for_label: unknown class label"
    );
}

double stable_log_sum_exp_row(
    const Matrix& values,
    Eigen::Index row
) {
    double max_value = -std::numeric_limits<double>::infinity();

    for (Eigen::Index j = 0; j < values.cols(); ++j) {
        max_value = std::max(max_value, values(row, j));
    }

    double sum_exp = 0.0;

    for (Eigen::Index j = 0; j < values.cols(); ++j) {
        sum_exp += std::exp(values(row, j) - max_value);
    }

    return max_value + std::log(sum_exp);
}

}  // namespace

void validate_gaussian_naive_bayes_options(
    const GaussianNaiveBayesOptions& options,
    const std::string& context
) {
    if (
        !std::isfinite(options.variance_smoothing) ||
        options.variance_smoothing <= 0.0
    ) {
        throw std::invalid_argument(
            context + ": variance_smoothing must be finite and positive"
        );
    }
}

GaussianNaiveBayes::GaussianNaiveBayes(
    GaussianNaiveBayesOptions options
)
    : options_{options} {
    validate_gaussian_naive_bayes_options(
        options_,
        "GaussianNaiveBayes"
    );
}

void GaussianNaiveBayes::fit(
    const Matrix& X,
    const Vector& y
) {
    validate_gaussian_naive_bayes_options(
        options_,
        "GaussianNaiveBayes::fit"
    );
    
    validate_non_empty_matrix(X, "GaussianNaiveBayes::fit");
    validate_class_targets(y, "GaussianNaiveBayes::fit");
    validate_same_number_of_rows(X, y, "GaussianNaiveBayes::fit");
    validate_finite_matrix_values(X, "GaussianNaiveBayes::fit");

    num_features_ = X.cols();

    classes_ = sorted_unique_classes(y);

    const Eigen::Index num_classes = classes_.size();

    class_priors_ = Vector::Zero(num_classes);

    means_ = Matrix::Zero(num_classes, num_features_);
    
    variances_ = Matrix::Zero(num_classes, num_features_);

    Vector class_counts = Vector::Zero(num_classes);

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const Eigen::Index class_index = class_index_for_label(classes_, y(i));

        class_counts(class_index) += 1.0;
        means_.row(class_index) += X.row(i);
    }

    for (
        Eigen::Index class_index = 0;
        class_index < num_classes;
        ++class_index
    ) {
        if (class_counts(class_index) <= 0.0) {
            throw std::invalid_argument(
                "GaussianNaiveBayes::fit: class count must be positive"
            );
        }

        means_.row(class_index) /= class_counts(class_index);

        class_priors_(class_index) = 
            class_counts(class_index) / static_cast<double>(X.rows());
    }

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const Eigen::Index class_index = class_index_for_label(classes_, y(i));

        const Eigen::RowVectorXd residual = X.row(i) - means_.row(class_index);

        variances_.row(class_index) += residual.array().square().matrix();
    }

    for (
        Eigen::Index class_index = 0;
        class_index < num_classes;
        ++class_index
    ) {
        variances_.row(class_index) /= class_counts(class_index);

        for (
            Eigen::Index feature_index = 0;
            feature_index < num_features_;
            ++feature_index
        ) {
            variances_(class_index, feature_index) += options_.variance_smoothing;
        }
    }

    fitted_ = true;
}

Vector GaussianNaiveBayes::predict(
    const Matrix& X
) const {
    const Matrix log_proba = predict_log_proba(X);

    Vector predictions(X.rows());

    for (Eigen::Index i = 0; i < log_proba.rows(); ++i) {
        Eigen::Index best_class_index = 0;
        double best_log_probability = log_proba(i, 0);

        for (
            Eigen::Index class_index = 1;
            class_index < log_proba.cols();
            ++class_index
        ) {
            if (log_proba(i, class_index) > best_log_probability) {
                best_log_probability = log_proba(i, class_index);
                best_class_index = class_index;
            }
        }

        predictions(i) = classes_(best_class_index);
    }

    return predictions;
}

Matrix GaussianNaiveBayes::predict_proba(
    const Matrix& X
) const {
    const Matrix log_proba = predict_log_proba(X);

    Matrix probabilities(log_proba.rows(), log_proba.cols());

    for (Eigen::Index i = 0; i < log_proba.rows(); ++i) {
        for (Eigen::Index j = 0; j < log_proba.cols(); ++j) {
            probabilities(i, j) = std::exp(log_proba(i, j));
        }
    }

    return probabilities;
}

Matrix GaussianNaiveBayes::predict_log_proba(
    const Matrix& X
) const {
    const Matrix joint_log_likelihood = compute_joint_log_likelihood(X);

    Matrix log_proba(
        joint_log_likelihood.rows(),
        joint_log_likelihood.cols()
    );

    for (Eigen::Index i = 0; i < joint_log_likelihood.rows(); ++i) {
        const double normalizer = stable_log_sum_exp_row(joint_log_likelihood, i);

        for (
            Eigen::Index class_index = 0;
            class_index < joint_log_likelihood.cols();
            ++class_index
        ) {
            log_proba(i, class_index) = joint_log_likelihood(i, class_index) - normalizer;
        }
    }

    return log_proba;
}

bool GaussianNaiveBayes::is_fitted() const {
    return fitted_;
}

const GaussianNaiveBayesOptions& GaussianNaiveBayes::options() const {
    return options_;
}

const Vector& GaussianNaiveBayes::classes() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "GaussianNaiveBayes::classes: model must be fitted"
        );
    }

    return classes_;
}

const Vector& GaussianNaiveBayes::class_priors() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "GaussianNaiveBayes::class_priors: model must be fitted"
        );
    }

    return class_priors_;
}

const Matrix& GaussianNaiveBayes::means() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "GaussianNaiveBayes::means: model must be fitted"
        );
    }

    return means_;
}

const Matrix& GaussianNaiveBayes::variances() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "GaussianNaiveBayes::variances: model must be fitted"
        );
    }

    return variances_;
}

Eigen::Index GaussianNaiveBayes::num_features() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "GaussianNaiveBayes::num_features: model must be fitted"
        );
    }

    return num_features_;
}

Matrix GaussianNaiveBayes::compute_joint_log_likelihood(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "GaussianNaiveBayes::compute_joint_log_likelihood: model must be fitted before prediction"
        );
    }

    validate_non_empty_matrix(
        X,
        "GaussianNaiveBayes::compute_joint_log_likelihood"
    );
    
    validate_finite_matrix_values(
        X,
        "GaussianNaiveBayes::compute_joint_log_likelihood"
    );

    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "GaussianNaiveBayes::compute_joint_log_likelihood: X feature count must match training feature count"
        );
    }

    Matrix joint_log_likelihood(X.rows(), classes_.size());

    for (
        Eigen::Index sample_index = 0;
        sample_index < X.rows();
        ++sample_index
    ) {
        for (
            Eigen::Index class_index = 0;
            class_index < classes_.size();
            ++class_index
        ) {
            double score = std::log(class_priors_(class_index));

            for (
                Eigen::Index feature_index = 0;
                feature_index < num_features_;
                ++feature_index
            ) {
                const double mean = means_(class_index, feature_index);
                const double variance = variances_(class_index, feature_index);
                const double residual = X(sample_index, feature_index) - mean;

                score += -0.5 * std::log(k_two_pi * variance) - (residual * residual) / (2.0 * variance);
            }

            joint_log_likelihood(sample_index, class_index) = score;
        }
    }
    
    return joint_log_likelihood;
}


}  // namespace ml