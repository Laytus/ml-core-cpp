#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/trees/gradient_boosting.hpp"

#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "GradientBoostingRegressor rejects prediction before fitting",
    "[gradient_boosting]"
) {
    ml::GradientBoostingRegressor model;

    ml::Matrix X(2, 1);
    X << 0.0,
         1.0;

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "GradientBoostingRegressor preserves a constant target",
    "[gradient_boosting]"
) {
    ml::Matrix X(5, 1);
    X << 0.0,
         1.0,
         2.0,
         3.0,
         4.0;

    ml::Vector y(5);
    y << 4.0, 4.0, 4.0, 4.0, 4.0;

    ml::GradientBoostingRegressorOptions options;
    options.n_estimators = 5;
    options.learning_rate = 0.1;
    options.max_depth = 1;

    ml::GradientBoostingRegressor model(options);
    model.fit(X, y);

    REQUIRE(model.is_fitted());
    REQUIRE(model.num_trees() == options.n_estimators);
    REQUIRE(model.initial_prediction() == Approx(4.0).margin(1e-12));
    REQUIRE(model.training_loss_history().size() == options.n_estimators);
    REQUIRE(model.training_loss_history().back() == Approx(0.0).margin(1e-12));

    const ml::Vector predictions = model.predict(X);

    for (Eigen::Index i = 0; i < predictions.size(); ++i) {
        REQUIRE(predictions(i) == Approx(4.0).margin(1e-12));
    }
}
