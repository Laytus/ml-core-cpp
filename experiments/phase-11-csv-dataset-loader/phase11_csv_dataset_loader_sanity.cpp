#include "ml/common/csv_dataset_loader.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_true(const std::string& label, bool condition) {
    if (!condition) {
        throw std::runtime_error("FAILED: " + label);
    }

    std::cout << "[OK] " << label << "\n";
}

void expect_size_equal(
    const std::string& label,
    std::size_t actual,
    std::size_t expected
) {
    if (actual != expected) {
        throw std::runtime_error(
            "FAILED: " + label
            + " | expected " + std::to_string(expected)
            + ", got " + std::to_string(actual)
        );
    }

    std::cout << "[OK] " << label << "\n";
}

void run_test(const std::string& label, void (*test_fn)()) {
    try {
        test_fn();
        std::cout << "[PASS] " << label << "\n\n";
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << label << "\n";
        std::cerr << "Reason: " << error.what() << "\n";
        throw;
    }
}

void test_load_wine_supervised_dataset() {
    ml::common::CsvDatasetLoader loader;

    const auto wine = loader.load_supervised(
        "data/processed/wine.csv",
        {
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280_od315_of_diluted_wines",
            "proline"
        },
        "class"
    );

    expect_size_equal("Wine row count", wine.rows(), static_cast<std::size_t>(178));
    expect_size_equal("Wine feature count", wine.features(), static_cast<std::size_t>(13));
    expect_size_equal("Wine target size", static_cast<std::size_t>(wine.y.size()), wine.rows());

    std::cout << "Wine rows: " << wine.rows() << "\n";
    std::cout << "Wine features: " << wine.features() << "\n";
}

void test_load_nasa_kc1_supervised_dataset() {
    ml::common::CsvDatasetLoader loader;

    const auto kc1 = loader.load_supervised(
        "data/processed/nasa_kc1_software_defects.csv",
        {
            "loc",
            "v_g",
            "ev_g",
            "iv_g",
            "n",
            "v",
            "l",
            "d",
            "i",
            "e",
            "b",
            "t",
            "lOCode",
            "lOComment",
            "lOBlank",
            "locCodeAndComment",
            "uniq_Op",
            "uniq_Opnd",
            "total_Op",
            "total_Opnd",
            "branchCount"
        },
        "defects"
    );

    expect_size_equal("KC1 row count", kc1.rows(), static_cast<std::size_t>(2109));
    expect_size_equal("KC1 feature count", kc1.features(), static_cast<std::size_t>(21));
    expect_size_equal("KC1 target size", static_cast<std::size_t>(kc1.y.size()), kc1.rows());

    std::cout << "KC1 rows: " << kc1.rows() << "\n";
    std::cout << "KC1 features: " << kc1.features() << "\n";
}

void test_load_stock_supervised_dataset() {
    ml::common::CsvDatasetLoader loader;

    const auto stock = loader.load_supervised(
        "data/processed/stock_ohlcv_engineered.csv",
        {
            "return_1d",
            "return_5d",
            "volatility_5d",
            "range_pct",
            "volume_change_1d"
        },
        "target_next_return"
    );

    expect_true("Stock row count is positive", stock.rows() > 0);
    expect_size_equal("Stock feature count", stock.features(), static_cast<std::size_t>(5));
    expect_size_equal("Stock target size", static_cast<std::size_t>(stock.y.size()), stock.rows());

    std::cout << "Stock rows: " << stock.rows() << "\n";
    std::cout << "Stock features: " << stock.features() << "\n";
}

void test_load_stock_unsupervised_dataset() {
    ml::common::CsvDatasetLoader loader;

    const auto stock = loader.load_unsupervised(
        "data/processed/stock_ohlcv_engineered.csv",
        {
            "return_1d",
            "return_5d",
            "volatility_5d",
            "range_pct",
            "volume_change_1d"
        }
    );

    expect_true("Stock unsupervised row count is positive", stock.rows() > 0);
    expect_size_equal("Stock unsupervised feature count", stock.features(), static_cast<std::size_t>(5));

    std::cout << "Stock unsupervised rows: " << stock.rows() << "\n";
    std::cout << "Stock unsupervised features: " << stock.features() << "\n";
}

void run_csv_dataset_loader_tests() {
    std::cout << "\n[Phase 11.1] CSV Dataset Loader sanity tests\n\n";

    run_test(
        "CsvDatasetLoader loads Wine supervised dataset",
        test_load_wine_supervised_dataset
    );

    run_test(
        "CsvDatasetLoader loads NASA KC1 supervised dataset",
        test_load_nasa_kc1_supervised_dataset
    );

    run_test(
        "CsvDatasetLoader loads stock OHLCV supervised dataset",
        test_load_stock_supervised_dataset
    );

    run_test(
        "CsvDatasetLoader loads stock OHLCV unsupervised dataset",
        test_load_stock_unsupervised_dataset
    );
}

}  // namespace

namespace ml::experiments {

void run_phase11_csv_dataset_loader_sanity() {
    run_csv_dataset_loader_tests();
}

}  // namespace ml::experiments