#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/common/preprocessing.hpp"

#include <cmath>
#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "Standardization centers and scales variable columns",
    "[common][preprocessing]"
) {
    ml::Matrix X(3, 2);
    X << 1.0, 10.0,
         2.0, 10.0,
         3.0, 10.0;

    const ml::Matrix result =
        ml::standardize_columns(X);

    const double expected =
        std::sqrt(3.0 / 2.0);

    REQUIRE(result.rows() == 3);
    REQUIRE(result.cols() == 2);

    REQUIRE(result(0, 0) == Approx(-expected));
    REQUIRE(result(1, 0) == Approx(0.0));
    REQUIRE(result(2, 0) == Approx(expected));

    REQUIRE(result(0, 1) == Approx(0.0));
    REQUIRE(result(1, 1) == Approx(0.0));
    REQUIRE(result(2, 1) == Approx(0.0));
}

TEST_CASE(
    "Min-max normalization maps variable columns to zero and one",
    "[common][preprocessing]"
) {
    ml::Matrix X(3, 2);
    X << 1.0, 10.0,
         2.0, 10.0,
         3.0, 10.0;

    const ml::Matrix result =
        ml::normalize_min_max_columns(X);

    REQUIRE(result(0, 0) == Approx(0.0));
    REQUIRE(result(1, 0) == Approx(0.5));
    REQUIRE(result(2, 0) == Approx(1.0));

    REQUIRE(result(0, 1) == Approx(0.0));
    REQUIRE(result(1, 1) == Approx(0.0));
    REQUIRE(result(2, 1) == Approx(0.0));
}

TEST_CASE(
    "Preprocessing rejects empty matrices",
    "[common][preprocessing][validation]"
) {
    ml::Matrix empty(0, 0);

    REQUIRE_THROWS_AS(
        ml::standardize_columns(empty),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        ml::normalize_min_max_columns(empty),
        std::invalid_argument
    );
}