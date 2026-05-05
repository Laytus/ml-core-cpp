#include "ml/trees/decision_tree.hpp"

#include "ml/common/shape_validation.hpp"
#include "ml/trees/tree_builder.hpp"

#include <stdexcept>

namespace ml {

DecisionTreeClassifier::DecisionTreeClassifier(DecisionTreeOptions options)
    : options_{options} {
        validate_decision_tree_options(options_, "DecisionTreeClassifier");
    }

void DecisionTreeClassifier::fit(
    const Matrix& X,
    const Vector& y
) {
    validate_decision_tree_options(options_, "DecisionTreeClassifier::fit");
    
    validate_non_empty_matrix(X, "DecisionTreeClassifier::fit");
    validate_non_empty_vector(y, "DecisionTreeClassifier::fit");
    validate_same_number_of_rows(X, y, "DecisionTreeClassifier::fit");

    root_ = build_tree(X, y, options_);

    num_features_ = X.cols();
}

Vector DecisionTreeClassifier::predict(
    const Matrix& X
) const {
    if (!is_fitted()) {
        throw std::invalid_argument(
            "DecisionTreeClassifier::predict: model must be fitted before prediction"
        );
    }
    
    validate_non_empty_matrix(X, "DecisionTreeClassifier::predict");
    
    if (X.cols() != num_features_) {
        throw std::invalid_argument(
            "DecisionTreeClassifier::predict: X feature count must match training feature count"
        );
    }

    Vector predictions(X.rows());

    for (Eigen::Index i = 0; i< X.rows(); ++i) {
        predictions(i) = predict_one(
            X.row(i).transpose(),
            *root_
        );
    }

    return predictions;
}

bool DecisionTreeClassifier::is_fitted() const {
    return root_ != nullptr;
}

const DecisionTreeOptions& DecisionTreeClassifier::options() const {
    return options_;
}

double DecisionTreeClassifier::predict_one(
    const Vector& sample,
    const TreeNode& node
) const {
    if (node.is_leaf) {
        return node.prediction;
    }

    if (!node.left || !node.right) {
        throw std::invalid_argument(
            "DecisionTreeClassifier::predict_one: internal node is missing children"
        );
    }

    if (node.feature_index < 0 || node.feature_index >= sample.size()) {
        throw std::invalid_argument(
            "DecisionTreeClassifier::predict_one: node feature_index is invalid"
        );
    }

    if (sample(node.feature_index) <= node.threshold) {
        return predict_one(
            sample,
            *node.left
        );
    }

    return predict_one(
        sample,
        *node.right
    );
}

}  // namespace ml