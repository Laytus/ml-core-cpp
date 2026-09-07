#include <catch2/catch_test_macros.hpp>

#include "ml/dl_bridge/perceptron.hpp"

#include <stdexcept>

TEST_CASE(
    "Perceptron rejects invalid learning rate",
    "[perceptron][validation]"
) {
    ml::PerceptronOptions options;
    options.learning_rate = 0.0;

    REQUIRE_THROWS_AS(
        ml::Perceptron(options),
        std::invalid_argument
    );
}

TEST_CASE(
    "Perceptron rejects prediction before fitting",
    "[perceptron]"
) {
    ml::Perceptron model;

    ml::Matrix X(2, 1);
    X << -1.0,
          1.0;

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "Perceptron learns a linearly separable dataset",
    "[perceptron]"
) {
    ml::Matrix X(6, 1);
    X << -3.0,
         -2.0,
         -1.0,
          1.0,
          2.0,
          3.0;

    ml::Vector y(6);
    y << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;

    ml::PerceptronOptions options;
    options.learning_rate = 0.1;
    options.max_epochs = 100;

    ml::Perceptron model(options);
    model.fit(X, y);

    REQUIRE(model.is_fitted());
    REQUIRE_FALSE(model.mistake_history().empty());
    REQUIRE(model.mistake_history().back() == 0.0);

    const ml::Vector predictions = model.predict(X);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        REQUIRE(predictions(i) == y(i));
    }
}
