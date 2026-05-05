cmake_minimum_required(VERSION 3.16)

project(ml_core_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(ml_core
    src/common/shape_validation.cpp
    src/common/math_ops.cpp
    src/common/statistics.cpp
    src/common/preprocessing.cpp
    src/common/data_split.cpp
    src/common/cross_validation.cpp
    src/common/preprocessing_pipeline.cpp
    src/common/numeric_utils.cpp
    src/common/regression_metrics.cpp
    src/common/baselines.cpp
    src/common/evaluation.cpp
    src/common/evaluation_harness.cpp
    src/common/experiment_summary.cpp
    src/linear_models/linear_regression.cpp
    src/common/classification_utils.cpp
    src/common/classification_metrics.cpp
    src/linear_models/logistic_regression.cpp
    src/common/classification_evaluation.cpp
    src/linear_models/softmax_regression.cpp
    src/common/multiclass_metrics.cpp
    src/optimization/model_optimization_adapters.cpp
    src/optimization/gradient_descent.cpp
    src/optimization/sgd.cpp
    src/optimization/mini_batch_gradient_descent.cpp
    src/optimization/training_history.cpp
    src/trees/split_scoring.cpp
    src/trees/tree_builder.cpp
    src/trees/decision_tree.cpp
)

target_include_directories(ml_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ml_core
    PUBLIC
        Eigen3::Eigen
)

add_executable(ml_core_app
    app/main.cpp
)

target_link_libraries(ml_core_app
    PRIVATE
        ml_core
)

add_executable(ml_core_tests
    app/test_runner.cpp
    experiments/common/manual_test_utils.cpp
    experiments/phase-1-math-sanity/phase1_math_sanity.cpp
    experiments/phase-2-evaluation-methodology/phase2_evaluation_sanity.cpp
    experiments/phase-3-linear-models/phase3_linear_models_sanity.cpp
    experiments/phase-4-logistic-regression/phase4_logistic_regression_sanity.cpp
    experiments/phase-4-logistic-regression/phase4_softmax_regression_sanity.cpp
    experiments/phase-5-optimization/phase5_optimization_sanity.cpp
    experiments/phase-6-trees-ensembles/phase6_trees_sanity.cpp
)

target_include_directories(ml_core_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/common
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/phase-1-math-sanity
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/phase-2-evaluation-methodology
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/phase-3-linear-models
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/phase-4-logistic-regression
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/phase-5-optimization
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/phase-6-trees-ensembles
)

target_link_libraries(ml_core_tests
    PRIVATE
        ml_core
)