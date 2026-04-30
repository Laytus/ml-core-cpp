#pragma once

#include "ml/common/types.hpp"

namespace ml {

double mean(const Vector& values);

double variance_population(const Vector& values);

double variance_sample(const Vector& values);

double standard_deviation_population(const Vector& values);

double standard_deviation_sample(const Vector& values);

Vector column_means(const Matrix& X);

Vector column_variance_population(const Matrix& X);

Vector column_variance_sample(const Matrix& X);

Vector column_standard_deviation_population(const Matrix& X);

Vector column_standard_deviation_sample(const Matrix& X);

}  // namespace ml