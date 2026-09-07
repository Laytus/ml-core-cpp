#include <catch2/catch_test_macros.hpp>

#include "ml/common/shape_validation.hpp"

#include <stdexcept>

TEST_CASE(
    "Shape validation accepts compatible dimensions",
    "[common][shape_validation]"
) {
    ml::Matrix X(3, 2);
    X.setZero();

    ml::Vector y(3);
    y.setZero();

    ml::Vector weights(2);
    weights.setZero();

    REQUIRE_NOTHROW(
        ml::validate_same_number_of_rows(X, y, "test")
    );

    REQUIRE_NOTHROW(
        ml::validate_feature_count(X, weights, "test")
    );

    REQUIRE_NOTHROW(
        ml::validate_non_empty_matrix(X, "test")
    );

    REQUIRE_NOTHROW(
        ml::validate_non_empty_vector(y, "test")
    );
}

TEST_CASE(
    "Shape validation rejects incompatible dimensions",
    "[common][shape_validation][validation]"
) {
    ml::Matrix X(3, 2);
    X.setZero();

    ml::Vector wrong_y(2);
    wrong_y.setZero();

    ml::Vector wrong_weights(3);
    wrong_weights.setZero();

    REQUIRE_THROWS_AS(
        ml::validate_same_number_of_rows(X, wrong_y, "test"),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        ml::validate_feature_count(X, wrong_weights, "test"),
        std::invalid_argument
    );
}

TEST_CASE(
    "Shape validation rejects empty inputs",
    "[common][shape_validation][validation]"
) {
    ml::Matrix empty_matrix(0, 0);
    ml::Vector empty_vector(0);

    REQUIRE_THROWS_AS(
        ml::validate_non_empty_matrix(empty_matrix, "test"),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        ml::validate_non_empty_vector(empty_vector, "test"),
        std::invalid_argument
    );
}

TEST_CASE(
    "Shape validation enforces minimum sizes",
    "[common][shape_validation][validation]"
) {
    ml::Vector v(2);
    v.setZero();

    ml::Matrix X(2, 2);
    X.setZero();

    REQUIRE_NOTHROW(
        ml::validate_min_vector_size(v, 2, "test")
    );

    REQUIRE_THROWS_AS(
        ml::validate_min_vector_size(v, 3, "test"),
        std::invalid_argument
    );

    REQUIRE_NOTHROW(
        ml::validate_min_matrix_rows(X, 2, "test")
    );

    REQUIRE_THROWS_AS(
        ml::validate_min_matrix_rows(X, 3, "test"),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        ml::validate_min_vector_size(v, -1, "test"),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        ml::validate_min_matrix_rows(X, -1, "test"),
        std::invalid_argument
    );
}