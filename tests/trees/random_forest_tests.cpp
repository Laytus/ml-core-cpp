#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/trees/random_forest.hpp"

#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "RandomForestClassifier rejects zero estimators",
    "[random_forest][validation]"
) {
    ml::RandomForestOptions options;
    options.n_estimators = 0;

    REQUIRE_THROWS_AS(
        ml::RandomForestClassifier(options),
        std::invalid_argument
    );
}

TEST_CASE(
    "RandomForestClassifier fits a simple binary dataset",
    "[random_forest]"
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

    ml::RandomForestOptions options;
    options.n_estimators = 5;
    options.bootstrap = false;
    options.random_seed = 42;
    options.tree_options.max_depth = 2;

    ml::RandomForestClassifier model(options);
    model.fit(X, y);

    REQUIRE(model.is_fitted());
    REQUIRE(model.num_trees() == options.n_estimators);
    REQUIRE(model.num_classes() == 2);

    const ml::Vector predictions = model.predict(X);
    const ml::Matrix probabilities = model.predict_proba(X);

    REQUIRE(probabilities.rows() == X.rows());
    REQUIRE(probabilities.cols() == 2);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        REQUIRE(predictions(i) == y(i));
        REQUIRE(probabilities.row(i).sum() == Approx(1.0).margin(1e-12));
    }
}
