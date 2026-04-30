#include "ml/common/shape_validation.hpp"
#include "ml/common/types.hpp"
#include "ml/common/math_ops.hpp"
#include "ml/common/statistics.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_pass(const std::string& test_name) {
    std::cout << "[PASS] " << test_name << "\n";
}

void print_fail(const std::string& test_name, const std::string& reason) {
    std::cout << "[FAIL] " << test_name << " — " << reason << "\n";
}

void expect_no_throw(const std::string& test_name, void (*test_fn)()) {
    try {
        test_fn();
        print_pass(test_name);
    } catch (const std::exception& e) {
        print_fail(test_name, e.what());
    } catch (...) {
        print_fail(test_name, "unknown exception");
    }
}

void expect_invalid_argument(const std::string& test_name, void (*test_fn)()) {
    try {
        test_fn();
        print_fail(test_name, "expected std::invalid_argument but no exception was thrown");
    } catch (const std::invalid_argument&) {
        print_pass(test_name);
    } catch (const std::exception& e) {
        print_fail(test_name, std::string("expected std::invalid_argument but got: ") + e.what());
    } catch (...) {
        print_fail(test_name, "expected std::invalid_argument but got unknown exception");
    }
}

void test_same_number_of_rows_accepts_matching_shapes() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector y(3);
    y << 10.0, 20.0, 30.0;

    ml::validate_same_number_of_rows(X, y, "test_same_number_of_rows_accepts_matching_shapes");
}

void test_same_number_of_rows_rejects_mismatched_shapes() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector y(2);
    y << 10.0, 20.0;

    ml::validate_same_number_of_rows(X, y, "test_same_number_of_rows_rejects_mismatched_shapes");
}

void test_same_size_accepts_equal_vectors() {
    ml::Vector a(3);
    a << 1.0, 2.0, 3.0;

    ml::Vector b(3);
    b << 4.0, 5.0, 6.0;

    ml::validate_same_size(a, b, "test_same_size_accepts_equal_vectors");
}

void test_same_size_rejects_unequal_vectors() {
    ml::Vector a(3);
    a << 1.0, 2.0, 3.0;

    ml::Vector b(2);
    b << 4.0, 5.0;

    ml::validate_same_size(a, b, "test_same_size_rejects_unequal_vectors");
}

void test_feature_count_accepts_matching_shapes() {
    ml::Matrix X(4, 3);
    X << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0,
         10.0, 11.0, 12.0;

    ml::Vector weights(3);
    weights << 0.1, 0.2, 0.3;

    ml::validate_feature_count(X, weights, "test_feature_count_accepts_matching_shapes");
}

void test_feature_count_rejects_mismatched_shapes() {
    ml::Matrix X(4, 3);
    X << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0,
         10.0, 11.0, 12.0;

    ml::Vector weights(2);
    weights << 0.1, 0.2;

    ml::validate_feature_count(X, weights, "test_feature_count_rejects_mismatched_shapes");
}

void test_non_empty_matrix_accepts_valid_matrix() {
    ml::Matrix X(2, 2);
    X << 1.0, 2.0,
         3.0, 4.0;

    ml::validate_non_empty_matrix(X, "test_non_empty_matrix_accepts_valid_matrix");
}

void test_non_empty_matrix_rejects_empty_matrix() {
    ml::Matrix X(0, 0);

    ml::validate_non_empty_matrix(X, "test_non_empty_matrix_rejects_empty_matrix");
}

void test_non_empty_vector_accepts_valid_vector() {
    ml::Vector v(3);
    v << 1.0, 2.0, 3.0;

    ml::validate_non_empty_vector(v, "test_non_empty_vector_accepts_valid_vector");
}

void test_non_empty_vector_rejects_empty_vector() {
    ml::Vector v(0);

    ml::validate_non_empty_vector(v, "test_non_empty_vector_rejects_empty_vector");
}

bool almost_equal(double actual, double expected, double tolerance = 1e-9) {
    return std::abs(actual - expected) <= tolerance;
}

void assert_almost_equal(double actual, double expected, const std::string& context) {
    if (!almost_equal(actual, expected)) {
        throw std::runtime_error(
            context + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void assert_vector_almost_equal(
    const ml::Vector& actual,
    const ml::Vector& expected,
    const std::string& context
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            context + ": vector sizes differ. expected size " +
            std::to_string(expected.size()) + ", got " +
            std::to_string(actual.size())
        );
    }

    for (Eigen::Index i = 0; i < actual.size(); ++i) {
        if (!almost_equal(actual(i), expected(i))) {
            throw std::runtime_error(
                context + ": vectors differ at index " + std::to_string(i) +
                ". expected " + std::to_string(expected(i)) +
                ", got " + std::to_string(actual(i))
            );
        }
    }
}

void run_shape_validation_tests() {
    std::cout << "[Phase 1.1] Shape validation tests\n\n";

    expect_no_throw(
        "validate_same_number_of_rows accepts matching X/y",
        test_same_number_of_rows_accepts_matching_shapes
    );

    expect_invalid_argument(
        "validate_same_number_of_rows rejects mismatched X/y",
        test_same_number_of_rows_rejects_mismatched_shapes
    );

    expect_no_throw(
        "validate_same_size accepts equal-size vectors",
        test_same_size_accepts_equal_vectors
    );

    expect_invalid_argument(
        "validate_same_size rejects unequal-size vectors",
        test_same_size_rejects_unequal_vectors
    );

    expect_no_throw(
        "validate_feature_count accepts matching X/weights",
        test_feature_count_accepts_matching_shapes
    );

    expect_invalid_argument(
        "validate_feature_count rejects mismatched X/weights",
        test_feature_count_rejects_mismatched_shapes
    );

    expect_no_throw(
        "validate_non_empty_matrix accepts non-empty matrix",
        test_non_empty_matrix_accepts_valid_matrix
    );

    expect_invalid_argument(
        "validate_non_empty_matrix rejects empty matrix",
        test_non_empty_matrix_rejects_empty_matrix
    );

    expect_no_throw(
        "validate_non_empty_vector accepts non-empty vector",
        test_non_empty_vector_accepts_valid_vector
    );

    expect_invalid_argument(
        "validate_non_empty_vector rejects empty vector",
        test_non_empty_vector_rejects_empty_vector
    );
}

void test_dot_product_computes_expected_scalar() {
    ml::Vector a(3);
    a << 1.0, 2.0, 3.0;

    ml::Vector b(3);
    b << 4.0, 5.0, 6.0;

    const double result = ml::dot_product(a, b);

    assert_almost_equal(result, 32.0, "test_dot_product_computes_expected_scalar");
}

void test_dot_product_rejects_mismatched_vectors() {
    ml::Vector a(3);
    a << 1.0, 2.0, 3.0;

    ml::Vector b(2);
    b << 4.0, 5.0;

    static_cast<void>(ml::dot_product(a, b));
}

void test_matvec_computes_expected_vector() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector weights(2);
    weights << 0.5, 1.0;

    const ml::Vector result = ml::matvec(X, weights);

    ml::Vector expected(3);
    expected << 2.5, 5.5, 8.5;

    assert_vector_almost_equal(result, expected, "test_matvec_computes_expected_vector");
}

void test_matvec_rejects_mismatched_shapes() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector weights(3);
    weights << 0.5, 1.0, 1.5;

    static_cast<void>(ml::matvec(X, weights));
}

void test_linear_prediction_computes_expected_vector() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector weights(2);
    weights << 0.5, 1.0;

    const double bias = 2.0;

    const ml::Vector result = ml::linear_prediction(X, weights, bias);

    ml::Vector expected(3);
    expected << 4.5, 7.5, 10.5;

    assert_vector_almost_equal(result, expected, "test_linear_prediction_computes_expected_vector");
}

void test_linear_prediction_rejects_mismatched_shapes() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector weights(3);
    weights << 0.5, 1.0, 1.5;

    static_cast<void>(ml::linear_prediction(X, weights, 2.0));
}

void test_residuals_computes_predictions_minus_targets() {
    ml::Vector predictions(3);
    predictions << 4.5, 7.5, 10.5;

    ml::Vector targets(3);
    targets << 5.0, 7.0, 11.0;

    const ml::Vector result = ml::residuals(predictions, targets);

    ml::Vector expected(3);
    expected << -0.5, 0.5, -0.5;

    assert_vector_almost_equal(result, expected, "test_residuals_computes_predictions_minus_targets");
}

void test_residuals_rejects_mismatched_vectors() {
    ml::Vector predictions(3);
    predictions << 4.5, 7.5, 10.5;

    ml::Vector targets(2);
    targets << 5.0, 7.0;

    static_cast<void>(ml::residuals(predictions, targets));
}

void test_mean_squared_error_computes_expected_value() {
    ml::Vector predictions(3);
    predictions << 4.5, 7.5, 10.5;

    ml::Vector targets(3);
    targets << 5.0, 7.0, 11.0;

    const double result = ml::mean_squared_error(predictions, targets);

    assert_almost_equal(result, 0.25, "test_mean_squared_error_computes_expected_value");
}

void test_mean_squared_error_rejects_mismatched_vectors() {
    ml::Vector predictions(3);
    predictions << 4.5, 7.5, 10.5;

    ml::Vector targets(2);
    targets << 5.0, 7.0;

    static_cast<void>(ml::mean_squared_error(predictions, targets));
}

void run_math_ops_tests() {
    std::cout << "\n[Phase 1.2] Matrix multiplication and math operations tests\n\n";

    expect_no_throw(
        "dot_product computes expected scalar",
        test_dot_product_computes_expected_scalar
    );

    expect_invalid_argument(
        "dot_product rejects mismatched vectors",
        test_dot_product_rejects_mismatched_vectors
    );

    expect_no_throw(
        "matvec computes Xw correctly",
        test_matvec_computes_expected_vector
    );

    expect_invalid_argument(
        "matvec rejects mismatched X/weights",
        test_matvec_rejects_mismatched_shapes
    );

    expect_no_throw(
        "linear_prediction computes Xw + b correctly",
        test_linear_prediction_computes_expected_vector
    );

    expect_invalid_argument(
        "linear_prediction rejects mismatched X/weights",
        test_linear_prediction_rejects_mismatched_shapes
    );

    expect_no_throw(
        "residuals computes predictions - targets",
        test_residuals_computes_predictions_minus_targets
    );

    expect_invalid_argument(
        "residuals rejects mismatched vectors",
        test_residuals_rejects_mismatched_vectors
    );

    expect_no_throw(
        "mean_squared_error computes expected value",
        test_mean_squared_error_computes_expected_value
    );

    expect_invalid_argument(
        "mean_squared_error rejects mismatched vectors",
        test_mean_squared_error_rejects_mismatched_vectors
    );
}

void test_mean_computes_expected_value() {
    ml::Vector values(4);
    values << 1.0, 2.0, 3.0, 4.0;

    const double result = ml::mean(values);

    assert_almost_equal(result, 2.5, "test_mean_computes_expected_value");
}

void test_mean_rejects_empty_vector() {
    ml::Vector values(0);

    static_cast<void>(ml::mean(values));
}

void test_variance_population_computes_expected_value() {
    ml::Vector values(4);
    values << 1.0, 2.0, 3.0, 4.0;

    const double result = ml::variance_population(values);

    assert_almost_equal(result, 1.25, "test_variance_population_computes_expected_value");
}

void test_variance_sample_computes_expected_value() {
    ml::Vector values(4);
    values << 1.0, 2.0, 3.0, 4.0;

    const double result = ml::variance_sample(values);

    assert_almost_equal(result, 5.0 / 3.0, "test_variance_sample_computes_expected_value");
}

void test_variance_sample_rejects_single_value_vector() {
    ml::Vector values(1);
    values << 1.0;

    static_cast<void>(ml::variance_sample(values));
}

void test_standard_deviation_population_computes_expected_value() {
    ml::Vector values(4);
    values << 1.0, 2.0, 3.0, 4.0;

    const double result = ml::standard_deviation_population(values);

    assert_almost_equal(
        result,
        std::sqrt(1.25),
        "test_standard_deviation_population_computes_expected_value"
    );
}

void test_standard_deviation_sample_computes_expected_value() {
    ml::Vector values(4);
    values << 1.0, 2.0, 3.0, 4.0;

    const double result = ml::standard_deviation_sample(values);

    assert_almost_equal(
        result,
        std::sqrt(5.0 / 3.0),
        "test_standard_deviation_sample_computes_expected_value"
    );
}

void test_column_means_computes_expected_vector() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    const ml::Vector result = ml::column_means(X);

    ml::Vector expected(2);
    expected << 3.0, 4.0;

    assert_vector_almost_equal(result, expected, "test_column_means_computes_expected_vector");
}

void test_column_variance_population_computes_expected_vector() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    const ml::Vector result = ml::column_variance_population(X);

    ml::Vector expected(2);
    expected << 8.0 / 3.0, 8.0 / 3.0;

    assert_vector_almost_equal(result, expected, "test_column_variance_population_computes_expected_vector");
}

void test_column_variance_sample_computes_expected_vector() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    const ml::Vector result = ml::column_variance_sample(X);

    ml::Vector expected(2);
    expected << 4.0, 4.0;

    assert_vector_almost_equal(result, expected, "test_column_variance_sample_computes_expected_vector");
}

void test_column_variance_sample_rejects_single_row_matrix() {
    ml::Matrix X(1, 2);
    X << 1.0, 2.0;

    static_cast<void>(ml::column_variance_sample(X));
}

void run_statistics_tests() {
    std::cout << "\n[Phase 1.3] Descriptive statistics tests\n\n";

    expect_no_throw(
        "mean computes expected value",
        test_mean_computes_expected_value
    );

    expect_invalid_argument(
        "mean rejects empty vector",
        test_mean_rejects_empty_vector
    );

    expect_no_throw(
        "variance_population computes expected value",
        test_variance_population_computes_expected_value
    );

    expect_no_throw(
        "variance_sample computes expected value",
        test_variance_sample_computes_expected_value
    );

    expect_invalid_argument(
        "variance_sample rejects vector with one element",
        test_variance_sample_rejects_single_value_vector
    );

    expect_no_throw(
        "standard_deviation_population computes expected value",
        test_standard_deviation_population_computes_expected_value
    );

    expect_no_throw(
        "standard_deviation_sample computes expected value",
        test_standard_deviation_sample_computes_expected_value
    );

    expect_no_throw(
        "column_means computes expected vector",
        test_column_means_computes_expected_vector
    );

    expect_no_throw(
        "column_variance_population computes expected vector",
        test_column_variance_population_computes_expected_vector
    );

    expect_no_throw(
        "column_variance_sample computes expected vector",
        test_column_variance_sample_computes_expected_vector
    );

    expect_invalid_argument(
        "column_variance_sample rejects matrix with one row",
        test_column_variance_sample_rejects_single_row_matrix
    );
}

}  // namespace

int main() {
    run_shape_validation_tests();
    run_math_ops_tests();
    run_statistics_tests();
    
    return 0;
}