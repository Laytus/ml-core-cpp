add_executable(
    ml_core_unit_tests

    common/shape_validation_tests.cpp
    common/classification_metrics_tests.cpp
    common/preprocessing_tests.cpp

    linear_models/linear_regression_tests.cpp
    linear_models/logistic_regression_tests.cpp
    linear_models/softmax_regression_tests.cpp
    linear_models/linear_svm_tests.cpp

    trees/decision_tree_tests.cpp
    trees/random_forest_tests.cpp
    trees/gradient_boosting_tests.cpp

    distance/knn_classifier_tests.cpp

    unsupervised/pca_tests.cpp
    unsupervised/kmeans_tests.cpp

    probabilistic/naive_bayes_tests.cpp

    dl_bridge/perceptron_tests.cpp
    dl_bridge/mlp_tests.cpp
)

target_link_libraries(
    ml_core_unit_tests
    PRIVATE
    ml_core
    Catch2::Catch2WithMain
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

include(Catch)
catch_discover_tests(ml_core_unit_tests)
