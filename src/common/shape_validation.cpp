#include "ml/common/shape_validation.hpp"

#include <sstream>
#include <stdexcept>

namespace ml {

void validate_same_number_of_rows(
    const Matrix& X,
    const Vector& y,
    const std::string& context
) {
    if (X.rows() != y.size()) {
        std::ostringstream oss;
        oss << context
            << ": X rows must match y size. Got X.rows() = "
            << X.rows()
            << ", y.size() = "
            << y.size()
            << ".";
        throw std::invalid_argument(oss.str());
    }
}

void validate_same_number_of_rows(
    const Matrix& X,
    const Matrix& y,
    const std::string& context
) {
    if (X.rows() != y.rows()) {
        std::ostringstream oss;
        oss << context
            << ": X rows must match y rows. Got X.rows() = "
            << X.rows()
            << ", y.rows() = "
            << y.rows()
            << ".";
        throw std::invalid_argument(oss.str());
    }
}

void validate_same_size(
    const Vector& a,
    const Vector& b,
    const std::string& context
) {
    if (a.size() != b.size()) {
        std::ostringstream oss;
        oss << context
            << ": vectors must have the same size. Got a.size() = "
            << a.size()
            << ", b.size() = "
            << b.size()
            << ".";
        throw std::invalid_argument(oss.str());
    }
}

void validate_feature_count(
    const Matrix& X,
    const Vector& weights,
    const std::string& context
) {
    if (X.cols() != weights.size()) {
        std::ostringstream oss;
        oss << context
            << ": X columns must match weights size. Got X.cols() = "
            << X.cols()
            << ", weights.size() = "
            << weights.size()
            << ".";
        throw std::invalid_argument(oss.str());
    }
}

void validate_non_empty_matrix(
    const Matrix& X,
    const std::string& context
) {
    if (X.rows() == 0 || X.cols() == 0) {
        std::ostringstream oss;
        oss << context
            << ": matrix must be non-empty. Got X.rows() = "
            << X.rows()
            << ", X.cols() = "
            << X.cols()
            << ".";
        throw std::invalid_argument(oss.str());
    }
}

void validate_non_empty_vector(
    const Vector& v,
    const std::string& context
) {
    if (v.size() == 0) {
        std::ostringstream oss;
        oss << context
            << ": vector must be non-empty. Got v.size() = 0.";
        throw std::invalid_argument(oss.str());
    }
}

void validate_min_vector_size(
    const Vector& v,
    Eigen::Index min_size,
    const std::string& context
) {
    if (min_size < 0) {
        std::ostringstream oss;
        oss << context
            << ": min_size must be non-negative. Got min_size = "
            << min_size
            << ".";
        throw std::invalid_argument(oss.str());
    }
    
    if (v.size() < min_size) {
        std::ostringstream oss;
        oss << context
            << ": vector size must be at least "
            << min_size
            << ". Got v.size() = "
            << v.size()
            << ".";
        throw std::invalid_argument(oss.str());
    }
}

void validate_min_matrix_rows(
    const Matrix& X,
    Eigen::Index min_rows,
    const std::string& context
) {
    if (min_rows < 0) {
        std::ostringstream oss;
        oss << context
            << ": min_rows must be non-negative. Got min_rows = "
            << min_rows
            << ".";
        throw std::invalid_argument(oss.str());
    }

    if (X.rows() < min_rows) {
        std::ostringstream oss;
        oss << context
            << ": matrix number of rows must be at least "
            << min_rows
            << ". Got X.rows() = "
            << X.rows()
            << ".";
        throw std::invalid_argument(oss.str());
    }
}

}  // namespace ml