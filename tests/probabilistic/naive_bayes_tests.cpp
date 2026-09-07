#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/probabilistic/naive_bayes.hpp"

#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "GaussianNaiveBayes rejects non-positive variance smoothing",
    "[naive_bayes][validation]"
) {
    ml::GaussianNaiveBayesOptions options;
    options.variance_smoothing = 0.0;

    REQUIRE_THROWS_AS(
        ml::GaussianNaiveBayes(options),
        std::invalid_argument
    );
}

TEST_CASE(
    "GaussianNaiveBayes rejects prediction before fitting",
    "[naive_bayes]"
) {
    ml::GaussianNaiveBayes model;

    ml::Matrix X(1, 2);
    X << 0.0, 0.0;

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "GaussianNaiveBayes learns separated Gaussian classes",
    "[naive_bayes]"
) {
    ml::Matrix X(6, 2);
    X <<  0.0,  0.0,
          0.2, -0.1,
         -0.2,  0.1,
          5.0,  5.0,
          5.2,  4.9,
          4.8,  5.1;

    ml::Vector y(6);
    y << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;

    ml::GaussianNaiveBayes model;
    model.fit(X, y);

    REQUIRE(model.is_fitted());
    REQUIRE(model.classes().size() == 2);
    REQUIRE(model.class_priors()(0) == Approx(0.5).margin(1e-12));
    REQUIRE(model.class_priors()(1) == Approx(0.5).margin(1e-12));

    const ml::Vector predictions = model.predict(X);
    const ml::Matrix probabilities = model.predict_proba(X);

    REQUIRE(probabilities.rows() == X.rows());
    REQUIRE(probabilities.cols() == 2);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        REQUIRE(predictions(i) == y(i));
        REQUIRE(probabilities.row(i).sum() == Approx(1.0).margin(1e-10));
    }
}
