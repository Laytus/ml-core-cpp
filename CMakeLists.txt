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
)

target_link_libraries(ml_core_tests
    PRIVATE
        ml_core
)