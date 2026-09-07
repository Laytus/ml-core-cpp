#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/trees/decision_tree.hpp"
#include "ml/trees/regression_tree.hpp"

#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "DecisionTreeClassifier rejects prediction before fitting",
    "[decision_tree][classification]"
) {
    ml::DecisionTreeClassifier model;

    ml::Matrix X(2, 1);
    X << 0.0,
         1.0;

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "DecisionTreeClassifier learns a simple threshold",
    "[decision_tree][classification]"
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

    ml::DecisionTreeOptions options;
    options.max_depth = 2;

    ml::DecisionTreeClassifier model(options);
    model.fit(X, y);

    REQUIRE(model.is_fitted());

    const ml::Vector predictions = model.predict(X);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        REQUIRE(predictions(i) == y(i));
    }
}

TEST_CASE(
    "DecisionTreeRegressor preserves a constant target",
    "[decision_tree][regression]"
) {
    ml::Matrix X(4, 1);
    X << 0.0,
         1.0,
         2.0,
         3.0;

    ml::Vector y(4);
    y << 2.5, 2.5, 2.5, 2.5;

    ml::DecisionTreeRegressor model;
    model.fit(X, y);

    REQUIRE(model.is_fitted());

    const ml::Vector predictions = model.predict(X);

    for (Eigen::Index i = 0; i < predictions.size(); ++i) {
        REQUIRE(predictions(i) == Approx(2.5).margin(1e-12));
    }
}
