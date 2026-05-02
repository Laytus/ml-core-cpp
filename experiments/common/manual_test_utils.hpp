#pragma once

#include "ml/common/types.hpp"

#include <string>

namespace ml::experiments::test {

void print_pass(const std::string& test_name);

void print_fail(const std::string& test_name, const std::string& reason);

void expect_no_throw(
    const std::string& test_name,
    void (*test_fn)()
);

void expect_invalid_argument(
    const std::string& test_name,
    void (*test_fn)()
);

bool almost_equal(
    double actual,
    double expected,
    double tolerance = 1e-9
);

void assert_almost_equal(
    double actual,
    double expected,
    const std::string& context,
    double tolerance = 1e-9
);

void assert_vector_almost_equal(
    const ml::Vector& actual,
    const ml::Vector& expected,
    const std::string& context,
    double tolerance = 1e-9
);

void assert_matrix_almost_equal(
    const ml::Matrix& actual,
    const ml::Matrix& expected,
    const std::string& context,
    double tolerance = 1e-9
);

}  // namespace ml::experiments::test