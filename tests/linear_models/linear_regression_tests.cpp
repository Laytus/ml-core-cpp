#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/linear_models/linear_regression.hpp"

#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "LinearRegression rejects prediction before fitting",
    "[linear_regression]"
) {
    ml::LinearRegression model;

    ml::Matrix X(2, 1);
    X << 0.0,
         1.0;

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "LinearRegression rejects invalid learning rate",
    "[linear_regression][validation]"
) {
    ml::LinearRegressionOptions options;
    options.learning_rate = 0.0;

    ml::LinearRegression model(options);

    ml::Matrix X(2, 1);
    X << 0.0,
         1.0;

    ml::Vector y(2);
    y << 1.0,
         3.0;

    REQUIRE_THROWS_AS(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_CASE(
    "LinearRegression fits an exact linear relationship",
    "[linear_regression]"
) {
    ml::Matrix X(5, 1);
    X << -2.0,
         -1.0,
          0.0,
          1.0,
          2.0;

    ml::Vector y(5);
    y << -3.0,
         -1.0,
          1.0,
          3.0,
          5.0;

    ml::LinearRegressionOptions options;
    options.learning_rate = 0.05;
    options.max_iterations = 10000;
    options.tolerance = 1e-12;

    ml::LinearRegression model(options);

    REQUIRE_FALSE(model.is_fitted());

    model.fit(X, y);

    REQUIRE(model.is_fitted());

    const ml::Vector predictions = model.predict(X);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        REQUIRE(predictions(i) == Approx(y(i)).margin(1e-3));
    }

    REQUIRE(model.weights().size() == 1);
    REQUIRE(model.weights()(0) == Approx(2.0).margin(1e-3));
    REQUIRE(model.bias() == Approx(1.0).margin(1e-3));
    REQUIRE(model.score_mse(X, y) < 1e-6);
}