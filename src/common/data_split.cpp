#include "ml/common/data_split.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {

TrainTestSplit train_test_split(
    const SupervisedDataset& dataset, 
    double test_ratio, 
    bool shuffle, 
    std::uint32_t seed
) {
    const Eigen::Index num_samples = dataset.num_samples();
    const Eigen::Index num_features = dataset.num_features();

    if (num_samples < 2) {
        throw std::invalid_argument(
            "train_test_split: dataset must contain at least 2 samples"
        );
    }
    
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        throw std::invalid_argument(
            "train_test_split: test_ratio must be strictly between 0.0 and 1.0"
        );
    }
    
    const auto raw_test_size = static_cast<Eigen::Index>(
        std::floor(static_cast<double>(num_samples) * test_ratio)
    );

    const Eigen::Index test_size = raw_test_size;
    const Eigen::Index train_size = num_samples - test_size;
    
    if (test_size < 1) {
        throw std::invalid_argument(
            "train_test_split: test_ratio produces an empty test set"
        );
    }
    
    if (train_size < 1) {
        throw std::invalid_argument(
            "train_test_split: test_ratio produces an empty training set"
        );
    }

    std::vector<Eigen::Index> indices(static_cast<std::size_t>(num_samples));
    std::iota(indices.begin(), indices.end(), Eigen::Index{0});

    if (shuffle) {
        std::mt19937 generator(seed);
        std::shuffle(indices.begin(), indices.end(), generator);
    }

    Matrix X_train(train_size, num_features);
    Vector y_train(train_size);
    Matrix X_test(test_size, num_features);
    Vector y_test(test_size);

    for (Eigen::Index i = 0; i < train_size; ++i) {
        const Eigen::Index source_index = indices[static_cast<std::size_t>(i)];
        X_train.row(i) = dataset.X.row(source_index);
        y_train(i) = dataset.y(source_index);
    }

    for (Eigen::Index i = 0; i < test_size; ++i) {
        const Eigen::Index source_index = indices[static_cast<std::size_t>(train_size + i)];
        X_test.row(i) = dataset.X.row(source_index);
        y_test(i) = dataset.y(source_index);
    }

    return TrainTestSplit{
        SupervisedDataset(X_train, y_train),
        SupervisedDataset(X_test, y_test),
    };
}

TrainValidationTestSplit train_validation_test_split(
    const SupervisedDataset& dataset, 
    double validation_ratio, 
    double test_ratio, 
    bool shuffle, 
    std::uint32_t seed
) {
    const Eigen::Index num_samples = dataset.num_samples();
    const Eigen::Index num_features = dataset.num_features();

    if (num_samples < 3) {
        throw std::invalid_argument(
            "train_validation_test_split: dataset must contain at least 3 samples"
        );
    }
    
    if (validation_ratio <= 0.0 || validation_ratio >= 1.0) {
        throw std::invalid_argument(
            "train_validation_test_split: validation_ratio must be strictly between 0.0 and 1.0"
        );
    }
    
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        throw std::invalid_argument(
            "train_validation_test_split: test_ratio must be strictly between 0.0 and 1.0"
        );
    }
    
    if (validation_ratio + test_ratio >= 1.0) {
        throw std::invalid_argument(
            "train_validation_test_split: validation_ratio + test_ratio must be strictly less than 1.0"
        );
    }
    
    const Eigen::Index validation_size = static_cast<Eigen::Index>(
        std::floor(static_cast<double>(num_samples) * validation_ratio)
    );
    
    const Eigen::Index test_size = static_cast<Eigen::Index>(
        std::floor(static_cast<double>(num_samples) * test_ratio)
    );

    const Eigen::Index train_size = num_samples - validation_size - test_size;
    
    if (train_size < 1) {
        throw std::invalid_argument(
            "train_validation_test_split: ratios produce an empty training set"
        );
    }
    
    if (validation_size < 1) {
        throw std::invalid_argument(
            "train_validation_test_split: validation_ratio produces an empty validation set"
        );
    }

    if (test_size < 1) {
        throw std::invalid_argument(
            "train_validation_test_split: test_ratio produces an empty test set"
        );
    }
    
    std::vector<Eigen::Index> indices(static_cast<std::size_t>(num_samples));
    std::iota(indices.begin(), indices.end(), Eigen::Index{0});

    if (shuffle) {
        std::mt19937 generator(seed);
        std::shuffle(indices.begin(), indices.end(), generator);
    }

    Matrix X_train(train_size, num_features);
    Vector y_train(train_size);

    Matrix X_validation(validation_size, num_features);
    Vector y_validation(validation_size);

    Matrix X_test(test_size, num_features);
    Vector y_test(test_size);

    for (Eigen::Index i = 0; i < train_size; ++i) {
        const Eigen::Index source_index = indices[static_cast<std::size_t>(i)];
        X_train.row(i) = dataset.X.row(source_index);
        y_train(i) = dataset.y(source_index);
    }

    for (Eigen::Index i = 0; i < validation_size; ++i) {
        const Eigen::Index source_index = indices[static_cast<std::size_t>(train_size + i)];
        X_validation.row(i) = dataset.X.row(source_index);
        y_validation(i) = dataset.y(source_index);
    }

    for (Eigen::Index i = 0; i < test_size; ++i) {
        const Eigen::Index source_index = indices[static_cast<std::size_t>(train_size + validation_size + i)];
        X_test.row(i) = dataset.X.row(source_index);
        y_test(i) = dataset.y(source_index);
    }

    return TrainValidationTestSplit{
        SupervisedDataset(X_train, y_train),
        SupervisedDataset(X_validation, y_validation),
        SupervisedDataset(X_test, y_test),
    };
}

}  // namespace ml