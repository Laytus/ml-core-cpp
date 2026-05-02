#pragma once

#include "ml/common/dataset.hpp"

#include <cstdint>

namespace ml {

struct TrainTestSplit {
    SupervisedDataset train;
    SupervisedDataset test;
};

struct TrainValidationTestSplit {
    SupervisedDataset train;
    SupervisedDataset validation;
    SupervisedDataset test;
};

TrainTestSplit train_test_split(
    const SupervisedDataset& dataset,
    double test_ratio,  // double in (0, 1)
    bool shuffle = true,  // controls wether indices are randomized
    std::uint32_t seed = 42  // makes randomized splitting reproducible
);

TrainValidationTestSplit train_validation_test_split(
    const SupervisedDataset& dataset,
    double validation_ratio,  // double in (0, 1)
    double test_ratio,  // double in (0, 1)
    bool shuffle = true,  // controls wether indices are randomized
    std::uint32_t seed = 42  // makes randomized splitting reproducible
);

}  // namespace ml