#include <catch2/catch_test_macros.hpp>

#include "ml/linear_models/logistic_regression.hpp"

#include <stdexcept>

TEST_CASE(
    "LogisticRegression rejects prediction before fitting",
    "[logistic_regression]"
) {
    ml::LogisticRegression model;

    ml::Matrix X(2, 1);
    X << -1.0,
          1.0;

    REQUIRE_THROWS_AS(
        model.predict_proba(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "LogisticRegression rejects invalid learning rate",
    "[logistic_regression][validation]"
) {
    ml::LogisticRegressionOptions options;
    options.learning_rate = 0.0;

    ml::LogisticRegression model(options);

    ml::Matrix X(2, 1);
    X << -1.0,
          1.0;

    ml::Vector y(2);
    y << 0.0,
         1.0;

    REQUIRE_THROWS_AS(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_CASE(
    "LogisticRegression classifies a simple separable dataset",
    "[logistic_regression]"
) {
    ml::Matrix X(6, 1);
    X << -3.0,
         -2.0,
         -1.0,
          1.0,
          2.0,
          3.0;

    ml::Vector y(6);
    y << 0.0,
         0.0,
         0.0,
         1.0,
         1.0,
         1.0;

    ml::LogisticRegressionOptions options;
    options.learning_rate = 0.1;
    options.max_iterations = 5000;
    options.tolerance = 1e-12;

    ml::LogisticRegression model(options);

    model.fit(X, y);

    REQUIRE(model.is_fitted());

    const ml::Vector probabilities = model.predict_proba(X);
    const ml::Vector predictions = model.predict_classes(X);

    REQUIRE(probabilities.size() == y.size());
    REQUIRE(predictions.size() == y.size());

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        REQUIRE(probabilities(i) >= 0.0);
        REQUIRE(probabilities(i) <= 1.0);
        REQUIRE(predictions(i) == y(i));
    }
}

TEST_CASE(
    "LogisticRegression rejects an invalid classification threshold",
    "[logistic_regression][validation]"
) {
    ml::Matrix X(4, 1);
    X << -2.0,
         -1.0,
          1.0,
          2.0;

    ml::Vector y(4);
    y << 0.0,
         0.0,
         1.0,
         1.0;

    ml::LogisticRegression model;
    model.fit(X, y);

    REQUIRE_THROWS_AS(
        model.predict_classes(X, 1.1),
        std::invalid_argument
    );
}