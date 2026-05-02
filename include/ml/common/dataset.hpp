#pragma once

#include "ml/common/shape_validation.hpp"
#include "ml/common/types.hpp"

namespace ml {

struct SupervisedDataset {
    Matrix X;
    Vector y;

    SupervisedDataset(const Matrix& features, const Vector& targets)
        : X(features), y(targets) {
        validate_non_empty_matrix(X, "SupervisedDataset");
        validate_non_empty_vector(y, "SupervisedDataset");
        validate_same_number_of_rows(X, y, "SupervisedDataset");
    }

    Eigen::Index num_samples() const {
        return X.rows();
    }

    Eigen::Index num_features() const {
        return X.cols();
    }
};

}  // namespace ml