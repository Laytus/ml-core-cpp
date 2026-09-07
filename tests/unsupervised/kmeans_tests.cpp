#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/unsupervised/kmeans.hpp"

#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "KMeans rejects zero clusters",
    "[kmeans][validation]"
) {
    ml::KMeansOptions options;
    options.num_clusters = 0;

    REQUIRE_THROWS_AS(
        ml::KMeans(options),
        std::invalid_argument
    );
}

TEST_CASE(
    "KMeans rejects prediction before fitting",
    "[kmeans]"
) {
    ml::KMeans model;

    ml::Matrix X(1, 2);
    X << 0.0, 0.0;

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "KMeans separates two obvious clusters",
    "[kmeans]"
) {
    ml::Matrix X(4, 2);
    X << 0.0,  0.0,
        10.0, 10.0,
         0.1,  0.0,
        10.1, 10.0;

    ml::KMeansOptions options;
    options.num_clusters = 2;
    options.max_iterations = 100;
    options.tolerance = 1e-12;

    ml::KMeans model(options);
    const ml::Vector labels = model.fit_predict(X);

    REQUIRE(model.is_fitted());
    REQUIRE(model.centroids().rows() == 2);
    REQUIRE(model.centroids().cols() == 2);
    REQUIRE(model.inertia() >= 0.0);
    REQUIRE_FALSE(model.inertia_history().empty());

    REQUIRE(labels(0) == labels(2));
    REQUIRE(labels(1) == labels(3));
    REQUIRE(labels(0) != labels(1));

    REQUIRE(model.centroids()(0, 0) == Approx(0.05).margin(1e-10));
    REQUIRE(model.centroids()(0, 1) == Approx(0.0).margin(1e-10));
    REQUIRE(model.centroids()(1, 0) == Approx(10.05).margin(1e-10));
    REQUIRE(model.centroids()(1, 1) == Approx(10.0).margin(1e-10));
}
