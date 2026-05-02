#include "ml/common/cross_validation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ml {

std::vector<FoldSplit> k_fold_split(
    const SupervisedDataset& dataset,
    Eigen::Index k,
    bool shuffle,
    std::uint32_t seed
) {
    const Eigen::Index num_samples = dataset.num_samples();
    const Eigen::Index num_features = dataset.num_features();

    if (num_samples < 2) {
        throw std::invalid_argument(
            "k_fold_split: dataset must contain at least 2 samples"
        );
    }

    if (k < 2) {
        throw std::invalid_argument(
            "k_fold_split: k must be at least 2"
        );
    }
    
    if (k > num_samples) {
        throw std::invalid_argument(
            "k_fold_split: k must not be less than or equal to the number of samples"
        );
    }

    std::vector<Eigen::Index> indices(static_cast<std::size_t>(num_samples));
    std::iota(indices.begin(), indices.end(), Eigen::Index{0});

    if (shuffle) {
        std::mt19937 generator(seed);
        std::shuffle(indices.begin(), indices.end(), generator);
    }

    const Eigen::Index base_fold_size = num_samples / k;
    const Eigen::Index remainder = num_samples % k;

    std::vector<FoldSplit> folds;
    folds.reserve(static_cast<std::size_t>(k));

    Eigen::Index validation_start = 0;

    for (Eigen::Index fold_index = 0; fold_index < k; ++fold_index) {
        const Eigen::Index validation_size = base_fold_size + (fold_index < remainder ? 1 : 0);
        const Eigen::Index validation_end = validation_start + validation_size;
        const Eigen::Index train_size = num_samples - validation_size;

        Matrix X_train(train_size, num_features);
        Vector y_train(train_size);
        Matrix X_validation(validation_size, num_features);
        Vector y_validation(validation_size);

        Eigen::Index train_row = 0;
        Eigen::Index validation_row = 0;

        for (Eigen::Index i = 0; i < num_samples; ++i) {
            const Eigen::Index source_index = indices[static_cast<std::size_t>(i)];

            if (i >= validation_start && i < validation_end) {
                X_validation.row(validation_row) = dataset.X.row(source_index);
                y_validation(validation_row) = dataset.y(source_index);
                ++validation_row;
            } else {
                X_train.row(train_row) = dataset.X.row(source_index);
                y_train(train_row) = dataset.y(source_index);
                ++train_row;
            }
        }

        folds.push_back(
            FoldSplit{
                SupervisedDataset(X_train, y_train),
                SupervisedDataset(X_validation, y_validation)
            }
        );

        validation_start = validation_end;
    }

    return folds;
}

}  // namespace ml