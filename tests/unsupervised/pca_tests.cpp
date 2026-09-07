#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ml/unsupervised/pca.hpp"

#include <stdexcept>

using Catch::Approx;

TEST_CASE(
    "PCA rejects zero components",
    "[pca][validation]"
) {
    ml::PCAOptions options;
    options.num_components = 0;

    REQUIRE_THROWS_AS(
        ml::PCA(options),
        std::invalid_argument
    );
}

TEST_CASE(
    "PCA rejects transform before fitting",
    "[pca]"
) {
    ml::PCA model;

    ml::Matrix X(2, 2);
    X << 1.0, 1.0,
         2.0, 2.0;

    REQUIRE_THROWS_AS(
        model.transform(X),
        std::invalid_argument
    );
}

TEST_CASE(
    "PCA captures a one-dimensional perfectly correlated dataset",
    "[pca]"
) {
    ml::Matrix X(4, 2);
    X << 1.0, 1.0,
         2.0, 2.0,
         3.0, 3.0,
         4.0, 4.0;

    ml::PCAOptions options;
    options.num_components = 1;

    ml::PCA model(options);
    const ml::Matrix Z = model.fit_transform(X);

    REQUIRE(model.is_fitted());
    REQUIRE(model.num_features() == 2);
    REQUIRE(Z.rows() == X.rows());
    REQUIRE(Z.cols() == 1);
    REQUIRE(model.components().rows() == 2);
    REQUIRE(model.components().cols() == 1);
    REQUIRE(model.explained_variance().size() == 1);
    REQUIRE(model.explained_variance_ratio().size() == 1);
    REQUIRE(model.explained_variance_ratio()(0) == Approx(1.0).margin(1e-10));

    const ml::Matrix reconstructed = model.inverse_transform(Z);

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            REQUIRE(reconstructed(i, j) == Approx(X(i, j)).margin(1e-9));
        }
    }
}
