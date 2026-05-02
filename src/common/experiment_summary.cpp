#include "ml/common/experiment_summary.hpp"

#include <fstream>
#include <stdexcept>

namespace ml {

namespace {

void validate_regression_experiment_summary(
    const RegressionExperimentSummary& summary,
    const std::string& context
) {
    if (summary.experiment_name.empty()) {
        throw std::invalid_argument(
            context + ": experiment_name must not be empty"
        );
    }
    
    if (summary.dataset_name.empty()) {
        throw std::invalid_argument(
            context + ": dataset_name must not be empty"
        );
    }
    
    if (summary.split_name.empty()) {
        throw std::invalid_argument(
            context + ": split_name must not be empty"
        );
    }
    
    if (summary.report.model_name.empty()) {
        throw std::invalid_argument(
            context + ": report.model_name must not be empty"
        );
    }
    
    if (summary.report.baseline_name.empty()) {
        throw std::invalid_argument(
            context + ": report.baseline_name must not be empty"
        );
    }
}

void validate_output_path(
    const std::string& output_path,
    const std::string& context
) {
    if (output_path.empty()) {
        throw std::invalid_argument(
            context + ": output_path must not be empty"
        );
    }
}

const char* bool_to_text(bool value) {
    return value ? "true" : "false";
}

}  // namespace

void export_regression_summary_csv(
    const RegressionExperimentSummary& summary,
    const std::string& output_path
) {
    const std::string context = "export_regression_summary_csv";
       
    validate_output_path(output_path, context);
    validate_regression_experiment_summary(summary, context);

    std::ofstream file(output_path);

    if(!file.is_open()) {
        throw std::runtime_error(
            context + ": could not open output file: " + output_path
        );
    }

    file << "experiment_name,"
         << "dataset_name,"
         << "split_name,"
         << "baseline_name,"
         << "model_name,"
         << "baseline_mse,"
         << "model_mse,"
         << "baseline_rmse,"
         << "model_rmse,"
         << "baseline_mae,"
         << "model_mae,"
         << "baseline_r2,"
         << "model_r2,"
         << "beats_mse,"
         << "beats_rmse,"
         << "beats_mae,"
         << "beats_r2\n";

    file << summary.experiment_name << ","
         << summary.dataset_name << ","
         << summary.split_name << ","
         << summary.report.baseline_name << ","
         << summary.report.model_name << ","
         << summary.report.comparison.baseline.mse << ","
         << summary.report.comparison.model.mse << ","
         << summary.report.comparison.baseline.rmse << ","
         << summary.report.comparison.model.rmse << ","
         << summary.report.comparison.baseline.mae << ","
         << summary.report.comparison.model.mae << ","
         << summary.report.comparison.baseline.r2 << ","
         << summary.report.comparison.model.r2 << ","
         << bool_to_text(summary.report.model_beats_baseline_mse()) << ","
         << bool_to_text(summary.report.model_beats_baseline_rmse()) << ","
         << bool_to_text(summary.report.model_beats_baseline_mae()) << ","
         << bool_to_text(summary.report.model_beats_baseline_r2()) << "\n";
}

void export_regression_summary_txt(
    const RegressionExperimentSummary& summary,
    const std::string& output_path
) {
    const std::string context = "export_regression_summary_txt";
       
    validate_output_path(output_path, context);
    validate_regression_experiment_summary(summary, context);

    std::ofstream file(output_path);

    if(!file.is_open()) {
        throw std::runtime_error(
            context + ": could not open output file: " + output_path
        );
    }

    file << "Experiment: " << summary.experiment_name << "\n"
         << "Dataset: " << summary.dataset_name << "\n"
         << "Split: " << summary.split_name << "\n\n";

    file << "Baseline: " << summary.report.baseline_name << "\n"
         << "Model: " << summary.report.model_name << "\n\n";

    file << "Baseline metrics:\n"
         << "  MSE: " << summary.report.comparison.baseline.mse << "\n"
         << "  RMSE: " << summary.report.comparison.baseline.rmse << "\n"
         << "  MAE: " << summary.report.comparison.baseline.mae << "\n"
         << "  R2: " << summary.report.comparison.baseline.r2 << "\n\n";

    file << "Model metrics:\n"
         << "  MSE: " << summary.report.comparison.model.mse << "\n"
         << "  RMSE: " << summary.report.comparison.model.rmse << "\n"
         << "  MAE: " << summary.report.comparison.model.mae << "\n"
         << "  R2: " << summary.report.comparison.model.r2 << "\n\n";

    file << "Comparison:\n"
         << "  Beats baseline on MSE: "
         << bool_to_text(summary.report.model_beats_baseline_mse()) << "\n"
         << "  Beats baseline on RMSE: "
         << bool_to_text(summary.report.model_beats_baseline_rmse()) << "\n"
         << "  Beats baseline on MAE: "
         << bool_to_text(summary.report.model_beats_baseline_mae()) << "\n"
         << "  Beats baseline on R2: "
         << bool_to_text(summary.report.model_beats_baseline_r2()) << "\n";
}

}  // namespace ml