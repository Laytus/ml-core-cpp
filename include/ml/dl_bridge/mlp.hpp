#pragma once

#include "ml/common/types.hpp"

#include <cstddef>
#include <string>
#include <vector>



namespace ml {

struct TinyMLPBinaryClassifierOptions {
    std::size_t hidden_units{4};
    double learning_rate{0.1};
    std::size_t max_epochs{500};
    std::size_t batch_size{4};
    unsigned int random_seed{42};
};

void validate_tiny_mlp_binary_classifier_options(
    const TinyMLPBinaryClassifierOptions& options,
    const std::string& context
);

struct TinyMLPForwardCache {
    Matrix X;
    Matrix Z1;
    Matrix A1;
    Matrix Z2;
    Matrix A2;
};

class TinyMLPBinaryClassifier {
public:
    TinyMLPBinaryClassifier() = default;

    explicit TinyMLPBinaryClassifier(
        TinyMLPBinaryClassifierOptions options
    );

    void fit(
        const Matrix& X,
        const Vector& y
    );

    Vector predict_proba(
        const Matrix& X
    ) const;

    Vector predict(
        const Matrix& X
    ) const;

    TinyMLPForwardCache forward(
        const Matrix& X
    ) const;

    bool is_fitted() const;

    const TinyMLPBinaryClassifierOptions& options() const;

    const Matrix& W1() const;

    const Vector& b1() const;

    const Matrix& W2() const;

    double b2() const;

    const std::vector<double> loss_history() const;

    Eigen::Index num_features() const;

private:
    void initialize_parameters(
        Eigen::Index num_features
    );

    double binary_cross_entropy(
        const Vector& probabilities,
        const Vector& y
    ) const;

    void train_on_batch(
        const Matrix& X_batch,
        const Vector& y_batch
    );

    Matrix select_batch_rows(
        const Matrix& X,
        std::size_t start_index,
        std::size_t batch_size
    ) const;
    
    Vector select_batch_targets(
        const Vector& y,
        std::size_t start_index,
        std::size_t batch_size
    ) const;

    TinyMLPBinaryClassifierOptions options_{};

    Matrix W1_{};
    Vector b1_{};
    Matrix W2_{};
    double b2_{0.0};

    std::vector<double> loss_history_{};

    Eigen::Index num_features_{0};
    bool fitted_{false};
};

}  // namespace ml