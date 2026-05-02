#pragma once

#include "ml/common/dataset.hpp"

#include <cstdint>
#include <vector>

namespace ml {

struct FoldSplit {
    SupervisedDataset train;
    SupervisedDataset validation;
};

std::vector<FoldSplit> k_fold_split(
    const SupervisedDataset& dataset,
    Eigen::Index k,
    bool shuffle = true,
    std::uint32_t seed = 42
);

}  // namespace ml