#include "ml/common/csv_dataset_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace ml::common {

std::size_t SupervisedDataset::rows() const {
    return static_cast<std::size_t>(X.rows());
}

std::size_t SupervisedDataset::features() const {
    return static_cast<std::size_t>(X.cols());
}

std::size_t UnsupervisedDataset::rows() const {
    return static_cast<std::size_t>(X.rows());
}

std::size_t UnsupervisedDataset::features() const {
    return static_cast<std::size_t>(X.cols());
}

CsvDatasetLoader::CsvDatasetLoader(CsvDatasetLoaderOptions options)
    : options_(std::move(options)) {}

SupervisedDataset CsvDatasetLoader::load_supervised(
    const std::string& csv_path,
    const std::vector<std::string>& feature_columns,
    const std::string& target_column
) const {
    validate_non_empty_columns(feature_columns, "feature_columns");
    validate_unique_columns(feature_columns, "feature_columns");

    if (trim(target_column).empty()) {
        throw std::invalid_argument("target_column must not be empty");
    }

    if (std::find(feature_columns.begin(), feature_columns.end(), target_column) != feature_columns.end()) {
        throw std::invalid_argument("target_column must not also appear in feature_columns");
    }

    const CsvTable table = read_csv_table(csv_path);

    const std::size_t n_rows = table.rows.size();
    const std::size_t n_features = feature_columns.size();

    if (n_rows == 0) {
        throw std::invalid_argument("CSV file contains no data rows: " + csv_path);
    }

    std::vector<std::size_t> feature_indices;
    feature_indices.reserve(n_features);

    for (const std::string& feature : feature_columns) {
        feature_indices.push_back(find_column_index(table.header, feature));
    }

    const std::size_t target_index = find_column_index(table.header, target_column);

    Eigen::MatrixXd X(static_cast<Eigen::Index>(n_rows), static_cast<Eigen::Index>(n_features));
    Eigen::VectorXd y(static_cast<Eigen::Index>(n_rows));

    for (std::size_t row_idx = 0; row_idx < n_rows; ++row_idx) {
        const std::vector<std::string>& row = table.rows[row_idx];

        for (std::size_t feature_idx = 0; feature_idx < n_features; ++feature_idx) {
            const std::size_t csv_col_idx = feature_indices[feature_idx];
            const std::string& column_name = feature_columns[feature_idx];

            X(
                static_cast<Eigen::Index>(row_idx),
                static_cast<Eigen::Index>(feature_idx)
            ) = parse_numeric_value(row[csv_col_idx], column_name, row_idx + 2);
        }

        y(static_cast<Eigen::Index>(row_idx)) =
            parse_numeric_value(row[target_index], target_column, row_idx + 2);
    }

    SupervisedDataset dataset;
    dataset.X = std::move(X);
    dataset.y = std::move(y);
    dataset.feature_names = feature_columns;
    dataset.target_name = target_column;

    return dataset;
}

UnsupervisedDataset CsvDatasetLoader::load_unsupervised(
    const std::string& csv_path,
    const std::vector<std::string>& feature_columns
) const {
    validate_non_empty_columns(feature_columns, "feature_columns");
    validate_unique_columns(feature_columns, "feature_columns");

    const CsvTable table = read_csv_table(csv_path);

    const std::size_t n_rows = table.rows.size();
    const std::size_t n_features = feature_columns.size();

    if (n_rows == 0) {
        throw std::invalid_argument("CSV file contains no data rows: " + csv_path);
    }

    std::vector<std::size_t> feature_indices;
    feature_indices.reserve(n_features);

    for (const std::string& feature : feature_columns) {
        feature_indices.push_back(find_column_index(table.header, feature));
    }

    Eigen::MatrixXd X(static_cast<Eigen::Index>(n_rows), static_cast<Eigen::Index>(n_features));

    for (std::size_t row_idx = 0; row_idx < n_rows; ++row_idx) {
        const std::vector<std::string>& row = table.rows[row_idx];

        for (std::size_t feature_idx = 0; feature_idx < n_features; ++feature_idx) {
            const std::size_t csv_col_idx = feature_indices[feature_idx];
            const std::string& column_name = feature_columns[feature_idx];

            X(
                static_cast<Eigen::Index>(row_idx),
                static_cast<Eigen::Index>(feature_idx)
            ) = parse_numeric_value(row[csv_col_idx], column_name, row_idx + 2);
        }
    }

    UnsupervisedDataset dataset;
    dataset.X = std::move(X);
    dataset.feature_names = feature_columns;

    return dataset;
}

CsvDatasetLoader::CsvTable CsvDatasetLoader::read_csv_table(
    const std::string& csv_path
) const {
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + csv_path);
    }

    std::string line;

    CsvTable table;

    if (!std::getline(file, line)) {
        throw std::invalid_argument("CSV file is empty: " + csv_path);
    }

    if (!options_.has_header) {
        throw std::invalid_argument(
            "CsvDatasetLoader currently requires has_header = true for real workflow loading"
        );
    }

    table.header = split_line(line, options_.delimiter);

    if (table.header.empty()) {
        throw std::invalid_argument("CSV header is empty: " + csv_path);
    }

    validate_unique_columns(table.header, "CSV header");

    std::size_t expected_columns = table.header.size();
    std::size_t line_number = 1;

    while (std::getline(file, line)) {
        ++line_number;

        if (trim(line).empty()) {
            continue;
        }

        std::vector<std::string> row = split_line(line, options_.delimiter);

        if (row.size() != expected_columns) {
            std::ostringstream oss;
            oss << "CSV row has wrong number of columns in file '" << csv_path
                << "' at line " << line_number
                << ": expected " << expected_columns
                << ", got " << row.size();

            throw std::invalid_argument(oss.str());
        }

        table.rows.push_back(std::move(row));
    }

    if (table.rows.empty()) {
        throw std::invalid_argument("CSV file contains header but no data rows: " + csv_path);
    }

    return table;
}

std::vector<std::string> CsvDatasetLoader::split_line(
    const std::string& line,
    char delimiter
) {
    std::vector<std::string> values;
    std::string current;
    bool in_quotes = false;

    for (char ch : line) {
        if (ch == '"') {
            in_quotes = !in_quotes;
            continue;
        }

        if (ch == delimiter && !in_quotes) {
            values.push_back(trim(current));
            current.clear();
            continue;
        }

        current.push_back(ch);
    }

    values.push_back(trim(current));

    return values;
}

std::string CsvDatasetLoader::trim(const std::string& value) {
    const auto first = std::find_if_not(
        value.begin(),
        value.end(),
        [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }
    );

    if (first == value.end()) {
        return "";
    }

    const auto last = std::find_if_not(
        value.rbegin(),
        value.rend(),
        [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }
    ).base();

    return std::string(first, last);
}

bool CsvDatasetLoader::is_missing_value(const std::string& value) const {
    const std::string cleaned = trim(value);

    return std::find(
        options_.missing_values.begin(),
        options_.missing_values.end(),
        cleaned
    ) != options_.missing_values.end();
}

double CsvDatasetLoader::parse_numeric_value(
    const std::string& value,
    const std::string& column_name,
    std::size_t row_number
) const {
    const std::string cleaned = trim(value);

    if (options_.reject_missing && is_missing_value(cleaned)) {
        std::ostringstream oss;
        oss << "Missing value found in column '" << column_name
            << "' at CSV row " << row_number;

        throw std::invalid_argument(oss.str());
    }

    std::size_t parsed_chars = 0;
    double result = 0.0;

    try {
        result = std::stod(cleaned, &parsed_chars);
    } catch (const std::exception&) {
        std::ostringstream oss;
        oss << "Could not parse numeric value in column '" << column_name
            << "' at CSV row " << row_number
            << ": " << value;

        throw std::invalid_argument(oss.str());
    }

    if (parsed_chars != cleaned.size()) {
        std::ostringstream oss;
        oss << "Unexpected trailing characters in numeric value for column '"
            << column_name << "' at CSV row " << row_number
            << ": " << value;

        throw std::invalid_argument(oss.str());
    }

    if (options_.reject_non_finite && !std::isfinite(result)) {
        std::ostringstream oss;
        oss << "Non-finite numeric value in column '" << column_name
            << "' at CSV row " << row_number
            << ": " << value;

        throw std::invalid_argument(oss.str());
    }

    return result;
}

std::size_t CsvDatasetLoader::find_column_index(
    const std::vector<std::string>& header,
    const std::string& column_name
) {
    const auto it = std::find(header.begin(), header.end(), column_name);

    if (it == header.end()) {
        std::ostringstream oss;
        oss << "Column not found: '" << column_name << "'. Available columns:";

        for (const std::string& column : header) {
            oss << " " << column;
        }

        throw std::invalid_argument(oss.str());
    }

    return static_cast<std::size_t>(std::distance(header.begin(), it));
}

void CsvDatasetLoader::validate_non_empty_columns(
    const std::vector<std::string>& columns,
    const std::string& context
) {
    if (columns.empty()) {
        throw std::invalid_argument(context + " must not be empty");
    }

    for (const std::string& column : columns) {
        if (trim(column).empty()) {
            throw std::invalid_argument(context + " must not contain empty column names");
        }
    }
}

void CsvDatasetLoader::validate_unique_columns(
    const std::vector<std::string>& columns,
    const std::string& context
) {
    std::unordered_set<std::string> seen;

    for (const std::string& column : columns) {
        if (seen.find(column) != seen.end()) {
            throw std::invalid_argument(context + " contains duplicate column: " + column);
        }

        seen.insert(column);
    }
}

} // namespace ml::common