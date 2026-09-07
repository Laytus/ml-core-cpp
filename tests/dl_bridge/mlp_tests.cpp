#include <catch2/catch_test_macros.hpp>

#include "ml/dl_bridge/mlp.hpp"

#include <stdexcept>

TEST_CASE(
    "TinyMLPBinaryClassifier rejects zero hidden units",
    "[tiny_mlp][validation]"
) {
    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 0;

    REQUIRE_THROWS_AS(
        ml::TinyMLPBinaryClassifier(options),
        std::invalid_argument
    );
}

TEST_CASE(
    "TinyMLPBinaryClassifier rejects prediction before fitting",
    "[tiny_mlp]"
) {
    ml::TinyMLPBinaryClassifier model;

    ml::Matrix X(2, 1);
    X << -1.0,
          1.0;

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "TinyMLPBinaryClassifier trains deterministically on a simple dataset",
    "[tiny_mlp]"
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

    ml::TinyMLPBinaryClassifierOptions options;
    options.hidden_units = 4;
    options.learning_rate = 0.1;
    options.max_epochs = 500;
    options.batch_size = 6;
    options.random_seed = 42;

    ml::TinyMLPBinaryClassifier model(options);
    model.fit(X, y);

    REQUIRE(model.is_fitted());
    REQUIRE(model.num_features() == 1);

    const auto history = model.loss_history();
    REQUIRE(history.size() == options.max_epochs);
    REQUIRE(history.back() < history.front());

    const ml::Vector probabilities = model.predict_proba(X);
    const ml::TinyMLPForwardCache cache = model.forward(X);

    REQUIRE(cache.X.rows() == X.rows());
    REQUIRE(cache.Z1.rows() == X.rows());
    REQUIRE(cache.Z1.cols() == static_cast<Eigen::Index>(options.hidden_units));
    REQUIRE(cache.A2.rows() == X.rows());
    REQUIRE(cache.A2.cols() == 1);

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        REQUIRE(probabilities(i) >= 0.0);
        REQUIRE(probabilities(i) <= 1.0);
    }
}
