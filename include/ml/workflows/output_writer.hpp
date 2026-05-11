#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace ml::workflows {

struct MetricsRow {
    std::string run_id;
    std::string workflow;
    std::string dataset;
    std::string model;
    std::string split;
    std::string metric;
    double value = 0.0;
};

struct RegressionPredictionRow {
    std::string run_id;
    std::size_t row_id = 0;
    std::string workflow;
    std::string dataset;
    std::string model;
    std::string split;
    double y_true = 0.0;
    double y_pred = 0.0;
    double error = 0.0;
};

struct ClassificationPredictionRow {
    std::string run_id;
    std::size_t row_id = 0;
    std::string workflow;
    std::string dataset;
    std::string model;
    std::string split;
    double y_true = 0.0;
    double y_pred = 0.0;
    int correct = 0;
};

struct BinaryProbabilityRow {
    std::string run_id;
    std::size_t row_id = 0;
    std::string workflow;
    std::string dataset;
    std::string model;
    std::string split;
    double y_true = 0.0;
    double probability_class_0 = 0.0;
    double probability_class_1 = 0.0;
};

struct MulticlassProbabilityRow {
    std::string run_id;
    std::size_t row_id = 0;
    std::string workflow;
    std::string dataset;
    std::string model;
    std::string split;
    double y_true = 0.0;
    std::vector<double> probabilities;
};

struct DecisionScoreRow {
    std::string run_id;
    std::size_t row_id = 0;
    std::string workflow;
    std::string dataset;
    std::string model;
    std::string split;
    double y_true = 0.0;
    double decision_score = 0.0;
};

struct LossHistoryRow {
    std::string run_id;
    std::string workflow;
    std::string dataset;
    std::string model;
    std::string split;
    std::size_t iteration = 0;
    double loss = 0.0;
};

struct HyperparameterSweepRow {
    std::string run_id;
    std::string workflow;
    std::string dataset;
    std::string model;
    std::string split;
    std::string param_name;
    std::string param_value;
    std::string metric;
    double value = 0.0;
};

struct ProjectionRow {
    std::string run_id;
    std::size_t row_id = 0;
    std::string workflow;
    std::string dataset;
    std::string method;
    std::string split;
    double component_1 = 0.0;
    double component_2 = 0.0;
    std::string label_reference;
};

struct Projection3DRow {
    std::string run_id;
    std::size_t row_id = 0;
    std::string workflow;
    std::string dataset;
    std::string method;
    std::string split;
    double component_1 = 0.0;
    double component_2 = 0.0;
    double component_3 = 0.0;
    std::string label_reference;
};

struct ClusteringAssignmentRow {
    std::string run_id;
    std::size_t row_id = 0;
    std::string workflow;
    std::string dataset;
    std::string method;
    std::string split;
    int cluster = 0;
    std::string label_reference;
};

class OutputWriter {
public:
    explicit OutputWriter(std::filesystem::path output_root);

    const std::filesystem::path& output_root() const;

    void write_metrics(
        const std::string& workflow_folder,
        const std::vector<MetricsRow>& rows,
        bool append = false
    ) const;

    void write_regression_predictions(
        const std::string& workflow_folder,
        const std::vector<RegressionPredictionRow>& rows,
        bool append = false
    ) const;

    void write_classification_predictions(
        const std::string& workflow_folder,
        const std::vector<ClassificationPredictionRow>& rows,
        bool append = false
    ) const;

    void write_binary_probabilities(
        const std::string& workflow_folder,
        const std::vector<BinaryProbabilityRow>& rows,
        bool append = false
    ) const;

    void write_multiclass_probabilities(
        const std::string& workflow_folder,
        const std::vector<MulticlassProbabilityRow>& rows,
        std::size_t class_count,
        bool append = false
    ) const;

    void write_decision_scores(
        const std::string& workflow_folder,
        const std::vector<DecisionScoreRow>& rows,
        bool append = false
    ) const;

    void write_loss_history(
        const std::string& workflow_folder,
        const std::vector<LossHistoryRow>& rows,
        bool append = false
    ) const;

    void write_hyperparameter_sweep(
        const std::string& workflow_folder,
        const std::vector<HyperparameterSweepRow>& rows,
        bool append = false
    ) const;

    void write_projections_2d(
        const std::string& workflow_folder,
        const std::vector<ProjectionRow>& rows,
        bool append = false
    ) const;

    void write_projections_3d(
        const std::string& workflow_folder,
        const std::vector<Projection3DRow>& rows,
        bool append = false
    ) const;

    void write_clustering_assignments(
        const std::string& workflow_folder,
        const std::vector<ClusteringAssignmentRow>& rows,
        bool append = false
    ) const;

private:
    std::filesystem::path output_root_;

    std::filesystem::path workflow_path(const std::string& workflow_folder) const;
    std::filesystem::path file_path(
        const std::string& workflow_folder,
        const std::string& file_name
    ) const;
};

}  // namespace ml::workflows