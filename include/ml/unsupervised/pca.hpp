#pragma once

#include "ml/common/types.hpp"

#include <cstddef>
#include <string>

namespace ml {

struct PCAOptions {
    std::size_t num_components{2};
};

void validate_pca_options(
    const PCAOptions& options,
    const std::string& context
);

class PCA {
public:
    PCA() = default;

    explicit PCA(PCAOptions options);

    void fit(
        const Matrix& X
    );

    Matrix transform(
        const Matrix& X
    ) const;

    Matrix fit_transform(
        const Matrix& X
    );

    Matrix inverse_transform(
        const Matrix& Z
    ) const;

    bool is_fitted() const;

    const PCAOptions& options() const;

    const Vector& mean() const;

    const Matrix& components() const;

    const Vector& explained_variance() const;
    
    const Vector& explained_variance_ratio() const;

    Eigen::Index num_features() const;

private:
    Matrix center_matrix(
        const Matrix& X
    ) const;

    PCAOptions options_{};
    Vector mean_{};
    Matrix components_{};
    Vector explained_variance_{};
    Vector explained_variance_ratio_{};

    Eigen::Index num_features_{0};
    bool fitted_{false};
};

}  // namespace ml