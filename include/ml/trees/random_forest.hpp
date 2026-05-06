#pragma once

#include "ml/common/types.hpp"
#include "ml/trees/decision_tree.hpp"
#include "ml/trees/split_scoring.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace ml {

struct RandomForestOptions {
    std::size_t n_estimators{100};
    bool bootstrap{true};
    std::optional<std::size_t> max_features{std::nullopt};
    unsigned int random_seed{42};

    DecisionTreeOptions tree_options{};
};

void validate_random_forest_options(
    const RandomForestOptions& options,
    const std::string& context
);

class RandomForestClassifier{
public:
    RandomForestClassifier() = default;

    explicit RandomForestClassifier(RandomForestOptions options);

    void fit(
        const Matrix& X,
        const Vector& y
    );

    Vector predict(
        const Matrix& X
    ) const;

    Matrix predict_proba(
        const Matrix& X
    ) const;

    bool is_fitted() const;

    const RandomForestOptions& options() const;

    std::size_t num_classes() const;
    
    std::size_t num_trees() const;

private:
    Vector predict_tree_votes_for_sample(
        const Matrix& samples_as_matrix
    ) const;

    double majority_vote(
        const Vector& tree_predictions
    ) const;

    Vector vote_probabilities(
        const Vector& tree_predictions
    ) const;

    RandomForestOptions options_{};
    std::vector<DecisionTreeClassifier> trees_{};

    Eigen::Index num_features_{0};
    std::size_t num_classes_{0};
    bool fitted_{false};
};

}  // namespace ml