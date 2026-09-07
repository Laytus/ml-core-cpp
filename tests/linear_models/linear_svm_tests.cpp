#include <catch2/catch_test_macros.hpp>

#include "ml/linear_models/linear_svm.hpp"

#include <stdexcept>

TEST_CASE(
    "LinearSVM rejects prediction before fitting",
    "[linear_svm]"
) {
    ml::LinearSVM model;

    ml::Matrix X(2, 1);
    X << -1.0,
          1.0;

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "LinearSVM rejects invalid options",
    "[linear_svm][validation]"
) {
    ml::LinearSVMOptions options;
    options.learning_rate = 0.0;

    REQUIRE_THROWS_AS(
        ml::LinearSVM(options),
        std::invalid_argument
    );
}

TEST_CASE(
    "LinearSVM separates a simple binary dataset",
    "[linear_svm]"
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

    ml::LinearSVMOptions options;
    options.learning_rate = 0.05;
    options.max_epochs = 200;
    options.l2_lambda = 0.01;

    ml::LinearSVM model(options);
    model.fit(X, y);

    REQUIRE(model.is_fitted());

    const ml::Vector scores = model.decision_function(X);
    const ml::Vector predictions = model.predict(X);

    REQUIRE(model.training_loss_history().size() == options.max_epochs);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        REQUIRE(predictions(i) == y(i));

        if (y(i) == 0.0) {
            REQUIRE(scores(i) < 0.0);
        } else {
            REQUIRE(scores(i) >= 0.0);
        }
    }
}
