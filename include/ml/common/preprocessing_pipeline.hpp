#pragma once

#include "ml/common/types.hpp"

namespace ml {

struct StandardScaler {
    Vector means;
    Vector standard_deviations;

    static StandardScaler fit(const Matrix& X_train);

    Matrix transform(const Matrix& X) const;

    Matrix fit_transform(const Matrix& X_train);
};

struct MinMaxScaler {
    Vector mins;
    Vector ranges;

    static MinMaxScaler fit(const Matrix& X_train);

    Matrix transform(const Matrix& X) const;

    Matrix fit_transform(const Matrix& X_train);
};

}  // namespace ml