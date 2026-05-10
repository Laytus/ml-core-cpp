#include "ml/dl_bridge/mlp.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/dl_bridge/activation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

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

void validate_binary_targets(
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
                context + ": target values must be integer-valued"
            );
        }

        if (!(rounded == 0.0 || rounded == 1.0)) {
            throw std::invalid_argument(
                context + ": target values must be binary labels 0 or 1"
            );
        }
    }
}

Matrix rows_from_indices(
    const Matrix& X,
    const std::vector<Eigen::Index>& indices,
    std::size_t start,
    std::size_t count
) {
    Matrix result(
        static_cast<Eigen::Index>(count),
        X.cols()
    );

    for (std::size_t i = 0; i < count; ++i) {
        result.row(static_cast<Eigen::Index>(i)) = X.row(indices[start + i]);
    }

    return result;
}

Vector targets_from_indices(
    const Vector& y,
    const std::vector<Eigen::Index>& indices,
    std::size_t start,
    std::size_t count
) {
    Vector result(static_cast<Eigen::Index>(count));

    for (std::size_t i = 0; i < count; ++i) {
        result(static_cast<Eigen::Index>(i)) = y(indices[start + i]);
    }

    return result;
}

}  // namespace

void validate_tiny_mlp_binary_classifier_options(
    const TinyMLPBinaryClassifierOptions& options,
    const std::string& context
) {
    if (options.hidden_units == 0) {
        throw std::invalid_argument(
            context + ": hidden_units must be at least 1"
        );
    }
    
    if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0) {
        throw std::invalid_argument(
            context + ": learning_rate must be finite and positive"
        );
    }
    
    if (options.max_epochs == 0) {
        throw std::invalid_argument(
            context + ": max_epochs must be at least 1"
        );
    }
    
    if (options.batch_size == 0) {
        throw std::invalid_argument(
            context + ": batch_size must be at least 1"
        );
    }
}

TinyMLPBinaryClassifier::TinyMLPBinaryClassifier(
    TinyMLPBinaryClassifierOptions options
) 
    : options_{options} {
    validate_tiny_mlp_binary_classifier_options(options_, "TinyMLPBinaryClassifier");
}

void TinyMLPBinaryClassifier::fit(
    const Matrix& X,
    const Vector& y
) {
    validate_tiny_mlp_binary_classifier_options(options_, "TinyMLPBinaryClassifier::fit");
    
    validate_non_empty_matrix(X, "TinyMLPBinaryClassifier::fit");
    validate_binary_targets(y, "TinyMLPBinaryClassifier::fit");
    validate_same_number_of_rows(X, y, "TinyMLPBinaryClassifier::fit");
    validate_finite_matrix_values(X, "TinyMLPBinaryClassifier::fit");

    initialize_parameters(X.cols());

    loss_history_.clear();
    loss_history_.reserve(options_.max_epochs);

    std::vector<Eigen::Index> indices(static_cast<std::size_t>(X.rows()));

    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 generator(options_.random_seed);

    for (std::size_t epoch = 0; epoch < options_.max_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), generator);

        for (
            std::size_t start = 0;
            start < indices.size();
            start += options_.batch_size
        ) {
            const std::size_t current_batch_size = std::min(options_.batch_size, indices.size() - start);

            const Matrix X_batch = rows_from_indices(X, indices, start, current_batch_size);
            const Vector y_batch = targets_from_indices(y, indices, start, current_batch_size);

            train_on_batch(X_batch, y_batch);
        }

        const Vector probabilities = predict_proba(X);

        loss_history_.push_back(binary_cross_entropy(probabilities, y));
    }

    fitted_ = true;
}

Vector TinyMLPBinaryClassifier::predict_proba(
    const Matrix& X
) const {
    const TinyMLPForwardCache cache = forward(X);

    return cache.A2.col(0);
}

Vector TinyMLPBinaryClassifier::predict(
    const Matrix& X
) const {
    const Vector probabilities = predict_proba(X);

    Vector predictions(probabilities.size());

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        predictions(i) = probabilities(i) >= 0.5 ? 1.0 : 0.0;
    }

    return predictions;
}

TinyMLPForwardCache TinyMLPBinaryClassifier::forward(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "TinyMLPBinaryClassifier::forward: model must be fitted before prediction"
        );
    }
    
    validate_non_empty_matrix(X, "TinyMLPBinaryClassifier::forward");
    validate_finite_matrix_values(X, "TinyMLPBinaryClassifier::forward");
    
    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "TinyMLPBinaryClassifier::forward: X feature count must match training feature count"
        );
    }

    TinyMLPForwardCache cache;
    cache.X = X;

    cache.Z1 = X * W1_;

    for (Eigen::Index i = 0; i < cache.Z1.rows(); ++i) {
        cache.Z1.row(i) += b1_.transpose();
    }

    cache.A1 = relu(cache.Z1);

    cache.Z2 = cache.A1 * W2_ + Matrix::Constant(cache.A1.rows(), 1, b2_);

    cache.A2 = sigmoid(cache.Z2);

    return cache;
}

bool TinyMLPBinaryClassifier::is_fitted() const {
    return fitted_;
}

const TinyMLPBinaryClassifierOptions& TinyMLPBinaryClassifier::options() const {
    return options_;
}

const Matrix& TinyMLPBinaryClassifier::W1() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "TinyMLPBinaryClassifier::W1: model must be fitted"
        );
    }

    return W1_;
}

const Vector& TinyMLPBinaryClassifier::b1() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "TinyMLPBinaryClassifier::b1: model must be fitted"
        );
    }

    return b1_;
}

const Matrix& TinyMLPBinaryClassifier::W2() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "TinyMLPBinaryClassifier::W2: model must be fitted"
        );
    }

    return W2_;
}

double TinyMLPBinaryClassifier::b2() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "TinyMLPBinaryClassifier::b2: model must be fitted"
        );
    }

    return b2_;
}

const std::vector<double> TinyMLPBinaryClassifier::loss_history() const {
    return loss_history_;
}

Eigen::Index TinyMLPBinaryClassifier::num_features() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "TinyMLPBinaryClassifier::num_features: model must be fitted"
        );
    }

    return num_features_;
}

void TinyMLPBinaryClassifier::initialize_parameters(
    Eigen::Index num_features
) {
    num_features_ = num_features;

    const double he_stddev = std::sqrt(2.0 / static_cast<double>(num_features_));

    std::mt19937 generator(options_.random_seed);
    std::normal_distribution<double> distribution(0.0, he_stddev);

    W1_ = Matrix(
        num_features_,
        static_cast<Eigen::Index>(options_.hidden_units)
    );

    b1_ = Vector::Constant(
        static_cast<Eigen::Index>(options_.hidden_units), 0.01
    );

    W2_ = Matrix(
        static_cast<Eigen::Index>(options_.hidden_units),
        1
    );

    for (Eigen::Index i = 0; i < W1_.rows(); ++i) {
        for (Eigen::Index j = 0; j < W1_.cols(); ++j) {
            W1_(i, j) = distribution(generator);
        }
    }

    for (Eigen::Index i = 0; i < W2_.rows(); ++i) {
        W2_(i, 0) = distribution(generator);
    }

    b2_ = 0.0;
    fitted_ = true;
}

double TinyMLPBinaryClassifier::binary_cross_entropy(
    const Vector& probabilities,
    const Vector& y
) const {
    if (probabilities.size() != y.size()) {
        throw std::invalid_argument(
            "TinyMLPBinaryClassifier::binary_cross_entropy: probabilities and y must have same size"
        );
    }

    constexpr double epsilon = 1e-12;

    double total = 0.0;

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        const double p = std::clamp(probabilities(i), epsilon, 1.0 - epsilon);

        total += -(y(i) * std::log(p) + (1.0 - y(i)) * std::log(1.0 - p));
    }

    return total / static_cast<double>(probabilities.size());
}

void TinyMLPBinaryClassifier::train_on_batch(
    const Matrix& X_batch,
    const Vector& y_batch
) {
    const TinyMLPForwardCache cache = forward(X_batch);

    const double n = static_cast<double>(X_batch.rows());

    Matrix dZ2 = cache.A2;

    for (Eigen::Index i = 0; i < dZ2.rows(); ++i) {
        dZ2(i, 0) -= y_batch(i);
    }

    dZ2 /= n;

    const Matrix dW2 = cache.A1.transpose() * dZ2;

    const double db2 = dZ2.sum();

    const Matrix dA1 = dZ2 * W2_.transpose();

    const Matrix dZ1 = dA1.array() * relu_derivative(cache.Z1).array();
    
    const Matrix dW1 = cache.X.transpose() * dZ1;

    const Vector db1 = dZ1.colwise().sum().transpose();

    W2_ -= options_.learning_rate * dW2;
    
    b2_ -= options_.learning_rate * db2;
    
    W1_ -= options_.learning_rate * dW1;
    
    b1_ -= options_.learning_rate * db1;
}

Matrix TinyMLPBinaryClassifier::select_batch_rows(
    const Matrix& X,
    std::size_t start_index,
    std::size_t batch_size
) const {
    Matrix result(static_cast<Eigen::Index>(batch_size), X.cols());

    for (std::size_t i = 0; i < batch_size; ++i) {
        result.row(static_cast<Eigen::Index>(i)) =
            X.row(static_cast<Eigen::Index>(start_index + i));
    }

    return result;
}

Vector TinyMLPBinaryClassifier::select_batch_targets(
    const Vector& y,
    std::size_t start_index,
    std::size_t batch_size
) const {
    Vector result(static_cast<Eigen::Index>(batch_size));

    for (std::size_t i = 0; i < batch_size; ++i) {
        result(static_cast<Eigen::Index>(i)) = 
            y(static_cast<Eigen::Index>(start_index + i));
    }

    return result;
}

}  // namespace ml