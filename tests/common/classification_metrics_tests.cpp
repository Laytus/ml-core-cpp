#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/common/classification_metrics.hpp"

#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "Binary classification metrics match known values",
    "[common][classification_metrics]"
) {
    ml::Vector predicted(4);
    predicted << 1.0,
                 1.0,
                 0.0,
                 0.0;

    ml::Vector targets(4);
    targets << 1.0,
               0.0,
               1.0,
               0.0;

    const ml::ConfusionMatrix matrix =
        ml::confusion_matrix(predicted, targets);

    REQUIRE(matrix.true_positive == 1);
    REQUIRE(matrix.true_negative == 1);
    REQUIRE(matrix.false_positive == 1);
    REQUIRE(matrix.false_negative == 1);

    REQUIRE(
        ml::accuracy_score(predicted, targets)
        == Approx(0.5)
    );

    REQUIRE(
        ml::precision_score(predicted, targets)
        == Approx(0.5)
    );

    REQUIRE(
        ml::recall_score(predicted, targets)
        == Approx(0.5)
    );

    REQUIRE(
        ml::f1_score(predicted, targets)
        == Approx(0.5)
    );
}

TEST_CASE(
    "Binary classification metrics handle zero positive predictions",
    "[common][classification_metrics]"
) {
    ml::Vector predicted(4);
    predicted << 0.0,
                 0.0,
                 0.0,
                 0.0;

    ml::Vector targets(4);
    targets << 0.0,
               0.0,
               1.0,
               1.0;

    REQUIRE(
        ml::precision_score(predicted, targets)
        == Approx(0.0)
    );

    REQUIRE(
        ml::recall_score(predicted, targets)
        == Approx(0.0)
    );

    REQUIRE(
        ml::f1_score(predicted, targets)
        == Approx(0.0)
    );
}

TEST_CASE(
    "Binary classification metrics reject incompatible sizes",
    "[common][classification_metrics][validation]"
) {
    ml::Vector predicted(3);
    predicted << 0.0, 1.0, 0.0;

    ml::Vector targets(2);
    targets << 0.0, 1.0;

    REQUIRE_THROWS_AS(
        ml::accuracy_score(predicted, targets),
        std::invalid_argument
    );
}

TEST_CASE(
    "Binary classification metrics reject non-binary values",
    "[common][classification_metrics][validation]"
) {
    ml::Vector predicted(3);
    predicted << 0.0, 2.0, 1.0;

    ml::Vector targets(3);
    targets << 0.0, 1.0, 1.0;

    REQUIRE_THROWS_AS(
        ml::confusion_matrix(predicted, targets),
        std::invalid_argument
    );
}