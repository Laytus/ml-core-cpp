#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <string>
#include <vector>

namespace ml::common {

struct CsvDatasetLoaderOptions {
    bool has_header = true;
    char delimiter = ',';

    bool reject_missing = true;
    bool reject_non_finite = true;

    std::vector<std::string> missing_values = {
        "",
        "?",
        "NA",
        "NaN",
        "nan",
        "null"
    };
};

struct SupervisedDataset {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    std::vector<std::string> feature_names;
    std::string target_name;

    std::size_t rows() const;
    std::size_t features() const;
};

struct UnsupervisedDataset {
    Eigen::MatrixXd X;

    std::vector<std::string> feature_names;

    std::size_t rows() const;
    std::size_t features() const;
};

class CsvDatasetLoader {
public:
    explicit CsvDatasetLoader(CsvDatasetLoaderOptions options = {});

    SupervisedDataset load_supervised(
        const std::string& csv_path,
        const std::vector<std::string>& feature_columns,
        const std::string& target_column
    ) const;

    UnsupervisedDataset load_unsupervised(
        const std::string& csv_path,
        const std::vector<std::string>& feature_columns
    ) const;

private:
    CsvDatasetLoaderOptions options_;

    struct CsvTable {
        std::vector<std::string> header;
        std::vector<std::vector<std::string>> rows;
    };

    CsvTable read_csv_table(const std::string& csv_path) const;

    static std::vector<std::string> split_line(
        const std::string& line,
        char delimiter
    );

    static std::string trim(const std::string& value);

    bool is_missing_value(const std::string& value) const;

    double parse_numeric_value(
        const std::string& value,
        const std::string& column_name,
        std::size_t row_number
    ) const;

    static std::size_t find_column_index(
        const std::vector<std::string>& header,
        const std::string& column_name
    );

    static void validate_non_empty_columns(
        const std::vector<std::string>& columns,
        const std::string& context
    );

    static void validate_unique_columns(
        const std::vector<std::string>& columns,
        const std::string& context
    );
};

} // namespace ml::common