#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/linear_models/softmax_regression.hpp"

#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "SoftmaxRegression rejects prediction before fitting",
    "[softmax_regression]"
) {
    ml::SoftmaxRegression model;

    ml::Matrix X(2, 3);
    X.setZero();

    REQUIRE_THROWS_AS(
        model.predict_proba(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "SoftmaxRegression rejects invalid learning rate",
    "[softmax_regression][validation]"
) {
    ml::SoftmaxRegressionOptions options;
    options.learning_rate = 0.0;

    ml::SoftmaxRegression model(options);

    ml::Matrix X(3, 3);
    X << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0;

    ml::Vector y(3);
    y << 0.0, 1.0, 2.0;

    REQUIRE_THROWS_AS(
        model.fit(X, y, 3),
        std::invalid_argument
    );
}

TEST_CASE(
    "SoftmaxRegression learns a simple three-class dataset",
    "[softmax_regression]"
) {
    ml::Matrix X(6, 3);
    X << 2.0, 0.0, 0.0,
         3.0, 0.0, 0.0,
         0.0, 2.0, 0.0,
         0.0, 3.0, 0.0,
         0.0, 0.0, 2.0,
         0.0, 0.0, 3.0;

    ml::Vector y(6);
    y << 0.0, 0.0, 1.0, 1.0, 2.0, 2.0;

    ml::SoftmaxRegressionOptions options;
    options.learning_rate = 0.1;
    options.max_iterations = 3000;
    options.tolerance = 1e-12;

    ml::SoftmaxRegression model(options);
    model.fit(X, y, 3);

    REQUIRE(model.is_fitted());
    REQUIRE(model.num_classes() == 3);

    const ml::Matrix probabilities = model.predict_proba(X);
    const ml::Vector predictions = model.predict_classes(X);

    REQUIRE(probabilities.rows() == X.rows());
    REQUIRE(probabilities.cols() == 3);

    for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
        REQUIRE(probabilities.row(i).sum() == Approx(1.0).margin(1e-10));
        REQUIRE(predictions(i) == y(i));
    }
}
