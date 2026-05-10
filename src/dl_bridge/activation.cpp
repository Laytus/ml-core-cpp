#include "ml/dl_bridge/activation.hpp"

#include <cmath>

namespace ml {

Matrix relu(
    const Matrix& Z
) {
    return Z.array().max(0.0).matrix();
}

Matrix relu_derivative(
    const Matrix& Z
) {
    Matrix derivative = Matrix::Zero(Z.rows(), Z.cols());

    for (Eigen::Index i = 0; i < Z.rows(); ++i) {
        for (Eigen::Index j = 0; j < Z.cols(); ++j) {
            derivative(i, j) = Z(i, j) > 0.0 ? 1.0 : 0.0;
        }
    }

    return derivative;
}

Matrix sigmoid(
    const Matrix& Z
) {
    Matrix result(Z.rows(), Z.cols());

    for (Eigen::Index i = 0; i < Z.rows(); ++i) {
        for (Eigen::Index j = 0; j < Z.cols(); ++j) {
            const double value = Z(i, j);

            if (value >= 0.0) {
                const double exp_neg = std::exp(-value);
                result(i, j) = 1.0 / (1.0 + exp_neg);
            } else {
                const double exp_pos = std::exp(value);
                result(i, j) = exp_pos / (1.0 + exp_pos);
            }
        }
    }

    return result;
}

// Vector sigmoid(
//     const Vector& Z
// ) {
//     Vector result(Z.size());

//     for (Eigen::Index i = 0; i < Z.size(); ++i) {
//         const double value = Z(i);

//         if (value >= 0.0) {
//             const double exp_neg = std::exp(-value);
//             result(i) = 1.0 / (1.0 + exp_neg);
//         } else {
//             const double exp_pos = std::exp(value);
//             result(i) = exp_pos / (1.0 + exp_pos);
//         }
//     }

//     return result;
// }

}  // namespace ml