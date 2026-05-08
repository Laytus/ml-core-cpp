#include "ml/unsupervised/pca.hpp"

#include "ml/common/shape_validation.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <numeric>
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

Vector column_means(
    const Matrix& X
) {
    validate_non_empty_matrix(X, "column_means PCA");

    Vector means(X.cols());

    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        means(j) = X.col(j).mean();
    }

    return means;
}

Matrix center_with_mean(
    const Matrix& X,
    const Vector& mean
) {
    Matrix centered = X;

    for (Eigen::Index i = 0; i < centered.rows(); ++i) {
        centered.row(i) -= mean.transpose();
    }

    return centered;
}

Matrix covariance_matrix_from_centered(
    const Matrix& X_centered
) {
    if (X_centered.rows() < 2) {
        throw std::invalid_argument(
            "covariance_matrix_from_centered: PCA requires at least 2 samples"
        );
    }

    return 
        (X_centered.transpose() * X_centered) /
        static_cast<double>(X_centered.rows() - 1); 
}

}  // namespace

void validate_pca_options(
    const PCAOptions& options,
    const std::string& context
) {
    if (options.num_components == 0) {
        throw std::invalid_argument(
            "validate_pca_options: num_components must be at least 1"
        );
    }
}

PCA::PCA(PCAOptions options)
    : options_{options} {
    validate_pca_options(options_, "PCA");
}

void PCA::fit(
    const Matrix& X
) {
    validate_pca_options(options_, "PCA::fit");
    
    validate_non_empty_matrix(X, "PCA::fit");
    validate_finite_matrix_values(X, "PCA::fit");

    if (X.rows() < 2) {
        throw std::invalid_argument(
            "PCA::fit: PCA requires at least 2 samples"
        );
    }

    if (options_.num_components > static_cast<std::size_t>(X.cols())) {
        throw std::invalid_argument(
            "PCA::fit: num_components must be less than or equal to the number of features"
        );
    }

    num_features_ = X.cols();

    mean_ = column_means(X);

    const Matrix X_centered = center_with_mean(X, mean_);

    const Matrix covariance = covariance_matrix_from_centered(X_centered);

    Eigen::SelfAdjointEigenSolver<Matrix> solver(covariance);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "PCA::fit: eigendecomposition failed"
        );
    }

    const Vector eigenvalues_ascending = solver.eigenvalues();
    const Matrix eigenvectors_ascending = solver.eigenvectors();

    std::vector<Eigen::Index> indices (
        static_cast<std::size_t>(eigenvalues_ascending.size())
    );

    std::iota(indices.begin(), indices.end(), 0);

    std::sort(
        indices.begin(),
        indices.end(),
        [&](Eigen::Index left, Eigen::Index right) {
            constexpr double epsilon = 1e-12;

            const double left_value = eigenvalues_ascending(left);
            const double right_value = eigenvalues_ascending(right);

            if (std::abs(left_value - right_value) > epsilon) {
                return left_value > right_value;
            }

            return left < right;
        }
    );

    const Eigen::Index num_components =
        static_cast<Eigen::Index>(options_.num_components);

    components_ = Matrix(X.cols(), num_components);

    explained_variance_ = Vector(num_components);

    for (
        Eigen::Index component_index = 0;
        component_index < num_components;
        ++component_index
    ) {
        const Eigen::Index source_index = 
            indices[static_cast<std::size_t>(component_index)];

        components_.col(component_index) = eigenvectors_ascending.col(source_index);

        explained_variance_(component_index) = std::max(0.0, eigenvalues_ascending(source_index));
    }

    const double total_variance = std::max(0.0, eigenvalues_ascending.sum());

    explained_variance_ratio_ = Vector(num_components);

    if (total_variance <= 0.0) {
        explained_variance_ratio_.setZero();
    } else {
        for (Eigen::Index i = 0; i < num_components; ++i) {
            explained_variance_ratio_(i) = explained_variance_(i) / total_variance;
        }
    }

    fitted_ = true;
}

Matrix PCA::transform(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "PCA::transform: model must be fitted before transform"
        );
    }
    
    validate_non_empty_matrix(X, "PCA::transform");
    validate_finite_matrix_values(X, "PCA::transform");
    
    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "PCA::transform: X feature count must match training feature count"
        );
    }

    return center_matrix(X) * components_;
}

Matrix PCA::fit_transform(
    const Matrix& X
) {
    fit(X);
    return transform(X);
}

Matrix PCA::inverse_transform(
    const Matrix& Z
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "PCA::inverse_transform: model must be fitted before inverse_transform"
        );
    }
    
    validate_non_empty_matrix(Z, "PCA::inverse_transform");
    validate_finite_matrix_values(Z, "PCA::inverse_transform");
    
    if (Z.cols() != components_.cols()) {
        throw std::invalid_argument(
            "PCA::inverse_transform: Z component count must match fitted num_components"
        );
    }

    Matrix reconstructed = Z * components_.transpose();

    for (Eigen::Index i = 0; i < reconstructed.rows(); ++i) {
        reconstructed.row(i) += mean_.transpose();
    }

    return reconstructed;
}

bool PCA::is_fitted() const {
    return fitted_;
}

const PCAOptions& PCA::options() const {
    return options_;
}

const Vector& PCA::mean() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "PCA::mean: model must be fitted"
        );
    }

    return mean_;
}

const Matrix& PCA::components() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "PCA::components: model must be fitted"
        );
    }

    return components_;
}

const Vector& PCA::explained_variance() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "PCA::explained_variance: model must be fitted"
        );
    }

    return explained_variance_;
}

const Vector& PCA::explained_variance_ratio() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "PCA::explained_variance_ratio: model must be fitted"
        );
    }

    return explained_variance_ratio_;
}

Eigen::Index PCA::num_features() const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "PCA::num_features: model must be fitted"
        );
    }

    return num_features_;
}

Matrix PCA::center_matrix(
    const Matrix& X
) const {
    return center_with_mean(X, mean_);
}

}  // namespace ml