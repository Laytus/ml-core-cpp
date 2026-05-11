#include "ml/workflows/output_writer.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml::workflows {
namespace {

std::string escape_csv_field(const std::string& value) {
    bool needs_quotes = false;

    for (const char ch : value) {
        if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
            needs_quotes = true;
            break;
        }
    }

    if (!needs_quotes) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');

    for (const char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
            escaped.push_back('"');
        } else {
            escaped.push_back(ch);
        }
    }

    escaped.push_back('"');
    return escaped;
}

std::string to_csv_number(const double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

std::string to_csv_int(const int value) {
    return std::to_string(value);
}

std::string to_csv_size(const std::size_t value) {
    return std::to_string(value);
}

void validate_workflow_folder(const std::string& workflow_folder) {
    if (workflow_folder.empty()) {
        throw std::invalid_argument("workflow_folder must not be empty");
    }

    if (workflow_folder.find("..") != std::string::npos) {
        throw std::invalid_argument("workflow_folder must not contain '..'");
    }

    if (workflow_folder.front() == '/' || workflow_folder.front() == '\\') {
        throw std::invalid_argument("workflow_folder must be relative");
    }
}

void ensure_parent_directory(const std::filesystem::path& path) {
    const auto parent = path.parent_path();

    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

bool file_exists_and_non_empty(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return false;
    }

    return std::filesystem::file_size(path) > 0;
}

void write_rows(
    const std::filesystem::path& path,
    const std::vector<std::string>& header,
    const std::vector<std::vector<std::string>>& rows,
    const bool append
) {
    ensure_parent_directory(path);

    const bool should_write_header = !append || !file_exists_and_non_empty(path);

    std::ofstream file;

    if (append) {
        file.open(path, std::ios::out | std::ios::app);
    } else {
        file.open(path, std::ios::out | std::ios::trunc);
    }

    if (!file.is_open()) {
        throw std::runtime_error("Could not open output CSV file: " + path.string());
    }

    const auto write_csv_line = [&file](const std::vector<std::string>& fields) {
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i > 0) {
                file << ',';
            }

            file << escape_csv_field(fields[i]);
        }

        file << '\n';
    };

    if (should_write_header) {
        write_csv_line(header);
    }

    for (const auto& row : rows) {
        if (row.size() != header.size()) {
            throw std::invalid_argument("CSV row size does not match header size");
        }

        write_csv_line(row);
    }
}

std::vector<std::string> metrics_header() {
    return {
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "metric",
        "value"
    };
}

std::vector<std::string> regression_predictions_header() {
    return {
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "y_pred",
        "error"
    };
}

std::vector<std::string> classification_predictions_header() {
    return {
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "y_pred",
        "correct"
    };
}

std::vector<std::string> binary_probabilities_header() {
    return {
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "probability_class_0",
        "probability_class_1"
    };
}

std::vector<std::string> multiclass_probabilities_header(const std::size_t class_count) {
    if (class_count == 0) {
        throw std::invalid_argument("class_count must be positive");
    }

    std::vector<std::string> header = {
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true"
    };

    for (std::size_t class_index = 0; class_index < class_count; ++class_index) {
        header.push_back("probability_class_" + std::to_string(class_index));
    }

    return header;
}

std::vector<std::string> decision_scores_header() {
    return {
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "decision_score"
    };
}

std::vector<std::string> loss_history_header() {
    return {
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "iteration",
        "loss"
    };
}

std::vector<std::string> hyperparameter_sweep_header() {
    return {
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "param_name",
        "param_value",
        "metric",
        "value"
    };
}

std::vector<std::string> projections_2d_header() {
    return {
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "method",
        "split",
        "component_1",
        "component_2",
        "label_reference"
    };
}

std::vector<std::string> projections_3d_header() {
    return {
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "method",
        "split",
        "component_1",
        "component_2",
        "component_3",
        "label_reference"
    };
}

std::vector<std::string> clustering_assignments_header() {
    return {
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "method",
        "split",
        "cluster",
        "label_reference"
    };
}

}  // namespace

OutputWriter::OutputWriter(std::filesystem::path output_root)
    : output_root_(std::move(output_root)) {
    if (output_root_.empty()) {
        throw std::invalid_argument("output_root must not be empty");
    }
}

const std::filesystem::path& OutputWriter::output_root() const {
    return output_root_;
}

std::filesystem::path OutputWriter::workflow_path(
    const std::string& workflow_folder
) const {
    validate_workflow_folder(workflow_folder);
    return output_root_ / workflow_folder;
}

std::filesystem::path OutputWriter::file_path(
    const std::string& workflow_folder,
    const std::string& file_name
) const {
    if (file_name.empty()) {
        throw std::invalid_argument("file_name must not be empty");
    }

    return workflow_path(workflow_folder) / file_name;
}

void OutputWriter::write_metrics(
    const std::string& workflow_folder,
    const std::vector<MetricsRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            row.workflow,
            row.dataset,
            row.model,
            row.split,
            row.metric,
            to_csv_number(row.value)
        });
    }

    write_rows(file_path(workflow_folder, "metrics.csv"), metrics_header(), csv_rows, append);
}

void OutputWriter::write_regression_predictions(
    const std::string& workflow_folder,
    const std::vector<RegressionPredictionRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            to_csv_size(row.row_id),
            row.workflow,
            row.dataset,
            row.model,
            row.split,
            to_csv_number(row.y_true),
            to_csv_number(row.y_pred),
            to_csv_number(row.error)
        });
    }

    write_rows(
        file_path(workflow_folder, "predictions.csv"),
        regression_predictions_header(),
        csv_rows,
        append
    );
}

void OutputWriter::write_classification_predictions(
    const std::string& workflow_folder,
    const std::vector<ClassificationPredictionRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            to_csv_size(row.row_id),
            row.workflow,
            row.dataset,
            row.model,
            row.split,
            to_csv_number(row.y_true),
            to_csv_number(row.y_pred),
            to_csv_int(row.correct)
        });
    }

    write_rows(
        file_path(workflow_folder, "predictions.csv"),
        classification_predictions_header(),
        csv_rows,
        append
    );
}

void OutputWriter::write_binary_probabilities(
    const std::string& workflow_folder,
    const std::vector<BinaryProbabilityRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            to_csv_size(row.row_id),
            row.workflow,
            row.dataset,
            row.model,
            row.split,
            to_csv_number(row.y_true),
            to_csv_number(row.probability_class_0),
            to_csv_number(row.probability_class_1)
        });
    }

    write_rows(
        file_path(workflow_folder, "probabilities.csv"),
        binary_probabilities_header(),
        csv_rows,
        append
    );
}

void OutputWriter::write_multiclass_probabilities(
    const std::string& workflow_folder,
    const std::vector<MulticlassProbabilityRow>& rows,
    const std::size_t class_count,
    const bool append
) const {
    const auto header = multiclass_probabilities_header(class_count);

    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        if (row.probabilities.size() != class_count) {
            throw std::invalid_argument(
                "MulticlassProbabilityRow probability count does not match class_count"
            );
        }

        std::vector<std::string> csv_row = {
            row.run_id,
            to_csv_size(row.row_id),
            row.workflow,
            row.dataset,
            row.model,
            row.split,
            to_csv_number(row.y_true)
        };

        for (const double probability : row.probabilities) {
            csv_row.push_back(to_csv_number(probability));
        }

        csv_rows.push_back(std::move(csv_row));
    }

    write_rows(file_path(workflow_folder, "probabilities.csv"), header, csv_rows, append);
}

void OutputWriter::write_decision_scores(
    const std::string& workflow_folder,
    const std::vector<DecisionScoreRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            to_csv_size(row.row_id),
            row.workflow,
            row.dataset,
            row.model,
            row.split,
            to_csv_number(row.y_true),
            to_csv_number(row.decision_score)
        });
    }

    write_rows(
        file_path(workflow_folder, "decision_scores.csv"),
        decision_scores_header(),
        csv_rows,
        append
    );
}

void OutputWriter::write_loss_history(
    const std::string& workflow_folder,
    const std::vector<LossHistoryRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            row.workflow,
            row.dataset,
            row.model,
            row.split,
            to_csv_size(row.iteration),
            to_csv_number(row.loss)
        });
    }

    write_rows(
        file_path(workflow_folder, "loss_history.csv"),
        loss_history_header(),
        csv_rows,
        append
    );
}

void OutputWriter::write_hyperparameter_sweep(
    const std::string& workflow_folder,
    const std::vector<HyperparameterSweepRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            row.workflow,
            row.dataset,
            row.model,
            row.split,
            row.param_name,
            row.param_value,
            row.metric,
            to_csv_number(row.value)
        });
    }

    write_rows(
        file_path(workflow_folder, "hyperparameter_sweep.csv"),
        hyperparameter_sweep_header(),
        csv_rows,
        append
    );
}

void OutputWriter::write_projections_2d(
    const std::string& workflow_folder,
    const std::vector<ProjectionRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            to_csv_size(row.row_id),
            row.workflow,
            row.dataset,
            row.method,
            row.split,
            to_csv_number(row.component_1),
            to_csv_number(row.component_2),
            row.label_reference
        });
    }

    write_rows(
        file_path(workflow_folder, "projections.csv"),
        projections_2d_header(),
        csv_rows,
        append
    );
}

void OutputWriter::write_projections_3d(
    const std::string& workflow_folder,
    const std::vector<Projection3DRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            to_csv_size(row.row_id),
            row.workflow,
            row.dataset,
            row.method,
            row.split,
            to_csv_number(row.component_1),
            to_csv_number(row.component_2),
            to_csv_number(row.component_3),
            row.label_reference
        });
    }

    write_rows(
        file_path(workflow_folder, "projections.csv"),
        projections_3d_header(),
        csv_rows,
        append
    );
}

void OutputWriter::write_clustering_assignments(
    const std::string& workflow_folder,
    const std::vector<ClusteringAssignmentRow>& rows,
    const bool append
) const {
    std::vector<std::vector<std::string>> csv_rows;
    csv_rows.reserve(rows.size());

    for (const auto& row : rows) {
        csv_rows.push_back({
            row.run_id,
            to_csv_size(row.row_id),
            row.workflow,
            row.dataset,
            row.method,
            row.split,
            to_csv_int(row.cluster),
            row.label_reference
        });
    }

    write_rows(
        file_path(workflow_folder, "clustering_assignments.csv"),
        clustering_assignments_header(),
        csv_rows,
        append
    );
}

}  // namespace ml::workflows