#include "manual_test_utils.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ml::experiments::test {

void print_pass(const std::string& test_name) {
    std::cout << "[PASS] " << test_name << "\n";
}

void print_fail(const std::string& test_name, const std::string& reason) {
    std::cout << "[FAIL] " << test_name << " — " << reason << "\n";
}

void expect_no_throw(
    const std::string& test_name,
    void (*test_fn)()
) {
    try {
        test_fn();
        print_pass(test_name);
    } catch (const std::exception& e) {
        print_fail(test_name, e.what());
    } catch (...) {
        print_fail(test_name, "unknown exception");
    }
}

void expect_invalid_argument(
    const std::string& test_name,
    void (*test_fn)()
) {
    try {
        test_fn();
        print_fail(test_name, "expected std::invalid_argument but no exception was thrown");
    } catch (const std::invalid_argument&) {
        print_pass(test_name);
    } catch (const std::exception& e) {
        print_fail(
            test_name,
            std::string("expected std::invalid_argument but got: ") + e.what()
        );
    } catch (...) {
        print_fail(test_name, "expected std::invalid_argument but got unknown exception");
    }
}

bool almost_equal(
    double actual,
    double expected,
    double tolerance
) {
    return std::abs(actual - expected) <= tolerance;
}

void assert_almost_equal(
    double actual,
    double expected,
    const std::string& context,
    double tolerance
) {
    if (!almost_equal(actual, expected, tolerance)) {
        throw std::runtime_error(
            context + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void assert_vector_almost_equal(
    const ml::Vector& actual,
    const ml::Vector& expected,
    const std::string& context,
    double tolerance
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            context + ": vector sizes differ. expected size " +
            std::to_string(expected.size()) + ", got " +
            std::to_string(actual.size())
        );
    }

    for (Eigen::Index i = 0; i < actual.size(); ++i) {
        if (!almost_equal(actual(i), expected(i), tolerance)) {
            throw std::runtime_error(
                context + ": vectors differ at index " + std::to_string(i) +
                ". expected " + std::to_string(expected(i)) +
                ", got " + std::to_string(actual(i))
            );
        }
    }
}

void assert_matrix_almost_equal(
    const ml::Matrix& actual,
    const ml::Matrix& expected,
    const std::string& context,
    double tolerance
) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        throw std::runtime_error(
            context + ": matrix shapes differ. expected " +
            std::to_string(expected.rows()) + "x" + std::to_string(expected.cols()) +
            ", got " + std::to_string(actual.rows()) + "x" +
            std::to_string(actual.cols())
        );
    }

    for (Eigen::Index i = 0; i < actual.rows(); ++i) {
        for (Eigen::Index j = 0; j < actual.cols(); ++j) {
            if (!almost_equal(actual(i, j), expected(i, j), tolerance)) {
                throw std::runtime_error(
                    context + ": matrices differ at (" + std::to_string(i) +
                    ", " + std::to_string(j) + "). expected " +
                    std::to_string(expected(i, j)) + ", got " +
                    std::to_string(actual(i, j))
                );
            }
        }
    }
}

}  // namespace ml::experiments::test