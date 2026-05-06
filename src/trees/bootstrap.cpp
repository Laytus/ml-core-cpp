#include "ml/trees/bootstrap.hpp"

#include "ml/common/shape_validation.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {

namespace {

void validate_bootstrap_input(
    const Matrix& X,
    const Vector& y,
    const std::string& context
) {
    validate_non_empty_matrix(X, context);
    validate_non_empty_vector(y, context);
    validate_same_number_of_rows(X, y, context);
}

std::vector<Eigen::Index> compute_out_of_bag_indices(
    Eigen::Index num_samples,
    const std::vector<Eigen::Index>& sampled_indices
) {
    if (num_samples <= 0) {
        throw std::invalid_argument(
            "compute_out_of_bag_indices: num_samples must be positive"
        );
    }

    std::vector<bool> was_sampled(
        static_cast<std::size_t>(num_samples),
        false
    );

    for (Eigen::Index index : sampled_indices) {
        if (index < 0 || index >= num_samples) {
            throw std::invalid_argument(
                "compute_out_of_bag_indices: sampled index is out of range"
            );
        }

        was_sampled[static_cast<std::size_t>(index)] = true;
    }

    std::vector<Eigen::Index> out_of_bag_indices;

    for (Eigen::Index i = 0; i < num_samples; ++i) {
        if (!was_sampled[static_cast<std::size_t>(i)]) {
            out_of_bag_indices.push_back(i);
        }
    }

    return out_of_bag_indices;
}

}  // namespace

BootstrapSample make_bootstrap_sample(
    const Matrix& X,
    const Vector& y,
    unsigned int random_seed
) {
    validate_bootstrap_input(X, y, "make_bootstrap_sample");

    const Eigen::Index num_samples = X.rows();
    const Eigen::Index num_features = X.cols();

    BootstrapSample sample;

    sample.X = Matrix(
        num_samples,
        num_features
    );

    sample.y = Vector(num_samples);

    sample.sampled_indices.reserve(
        static_cast<std::size_t>(num_samples)
    );

    std::mt19937 generator(random_seed);

    std::uniform_int_distribution<Eigen::Index> distribution(
        0,
        num_samples - 1
    );

    for (Eigen::Index row = 0; row < num_samples; ++row) {
        const Eigen::Index sampled_index = distribution(generator);

        sample.X.row(row) = X.row(sampled_index);
        sample.y(row) = y(sampled_index);

        sample.sampled_indices.push_back(sampled_index);
    }

    sample.out_of_bag_indices = compute_out_of_bag_indices(
        num_samples,
        sample.sampled_indices
    );

    return sample;
}

BootstrapSample make_full_sample(
    const Matrix& X,
    const Vector& y
) {
    validate_bootstrap_input(X, y, "make_full_sample");

    const Eigen::Index num_samples = X.rows();

    BootstrapSample sample;

    sample.X = X;
    sample.y = y;

    sample.sampled_indices.reserve(
        static_cast<std::size_t>(num_samples)
    );

    for (Eigen::Index i = 0; i < num_samples; ++i) {
        sample.sampled_indices.push_back(i);
    }

    sample.out_of_bag_indices.clear();

    return sample;
}

}  // namespace ml