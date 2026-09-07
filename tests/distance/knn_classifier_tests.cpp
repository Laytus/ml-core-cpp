#include <catch2/catch_test_macros.hpp>

#include "ml/distance/knn_classifier.hpp"

#include <stdexcept>

TEST_CASE(
    "KNNClassifier rejects zero neighbors",
    "[knn][validation]"
) {
    ml::KNNClassifierOptions options;
    options.k = 0;

    REQUIRE_THROWS_AS(
        ml::KNNClassifier(options),
        std::invalid_argument
    );
}

TEST_CASE(
    "KNNClassifier rejects prediction before fitting",
    "[knn]"
) {
    ml::KNNClassifier model;

    ml::Matrix X(1, 1);
    X << 0.0;

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "KNNClassifier predicts nearest classes with k equal to one",
    "[knn]"
) {
    ml::Matrix X_train(4, 1);
    X_train << 0.0,
               1.0,
              10.0,
              11.0;

    ml::Vector y_train(4);
    y_train << 0.0, 0.0, 1.0, 1.0;

    ml::KNNClassifierOptions options;
    options.k = 1;

    ml::KNNClassifier model(options);
    model.fit(X_train, y_train);

    REQUIRE(model.is_fitted());
    REQUIRE(model.num_train_samples() == 4);
    REQUIRE(model.num_features() == 1);

    ml::Matrix X_query(2, 1);
    X_query << 0.2,
              10.8;

    const ml::Vector predictions = model.predict(X_query);

    REQUIRE(predictions(0) == 0.0);
    REQUIRE(predictions(1) == 1.0);
}
