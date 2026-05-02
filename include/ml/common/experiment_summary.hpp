#pragma once

#include "ml/common/evaluation_harness.hpp"

#include <string>

namespace ml {

struct RegressionExperimentSummary {
    std::string experiment_name;
    std::string dataset_name;
    std::string split_name;
    RegressionEvaluationReport report;
};

void export_regression_summary_csv(
    const RegressionExperimentSummary& summary,
    const std::string& output_path
);

void export_regression_summary_txt(
    const RegressionExperimentSummary& summary,
    const std::string& output_path
);

}  // namespace ml