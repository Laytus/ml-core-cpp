#include "phase2_evaluation_sanity.hpp"

#include "manual_test_utils.hpp"

#include "ml/common/dataset.hpp"
#include "ml/common/data_split.hpp"
#include "ml/common/cross_validation.hpp"
#include "ml/common/preprocessing_pipeline.hpp"
#include "ml/common/regression_metrics.hpp"
#include "ml/common/baselines.hpp"
#include "ml/common/evaluation.hpp"
#include "ml/common/evaluation_harness.hpp"
#include "ml/common/experiment_summary.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace test = ml::experiments::test;

namespace {

void test_supervised_dataset_accepts_valid_xy() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector y(3);
    y << 10.0, 20.0, 30.0;

    const ml::SupervisedDataset dataset(X, y);

    if (dataset.num_samples() != 3) {
        throw std::runtime_error("expected num_samples() == 3");
    }

    if (dataset.num_features() != 2) {
        throw std::runtime_error("expected num_features() == 2");
    }
}

void test_supervised_dataset_rejects_mismatched_xy() {
    ml::Matrix X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    ml::Vector y(2);
    y << 10.0, 20.0;

    const ml::SupervisedDataset dataset(X, y);

    static_cast<void>(dataset);
}

void test_supervised_dataset_rejects_empty_matrix() {
    ml::Matrix X(0, 0);
    ml::Vector y(0);
    
    const ml::SupervisedDataset dataset(X, y);
    static_cast<void>(dataset);
}

void test_supervised_dataset_reports_num_samples() {
    ml::Matrix X(4, 3);
    X << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0,
         10.0, 11.0, 12.0;

    ml::Vector y(4);
    y << 1.0, 0.0, 1.0, 0.0;

    const ml::SupervisedDataset dataset(X, y);

    if (dataset.num_samples() != 4) {
        throw std::runtime_error("expected num_samples() == 4");
    }
}

void test_supervised_dataset_reports_num_features() {
    ml::Matrix X(4, 3);
    X << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0,
         10.0, 11.0, 12.0;

    ml::Vector y(4);
    y << 1.0, 0.0, 1.0, 0.0;

    const ml::SupervisedDataset dataset(X, y);

    if (dataset.num_features() != 3) {
        throw std::runtime_error("expected num_features() == 3");
    }
}

ml::SupervisedDataset make_split_test_dataset() {
    ml::Matrix X(5, 2);
    X << 1.0, 10.0,
         2.0, 20.0,
         3.0, 30.0,
         4.0, 40.0,
         5.0, 50.0;

    ml::Vector y(5);
    y << 100.0, 200.0, 300.0, 400.0, 500.0;

    return ml::SupervisedDataset(X, y);
}

void test_train_test_split_creates_expected_sizes() {
    const ml::SupervisedDataset dataset = make_split_test_dataset();

    const ml::TrainTestSplit split = ml::train_test_split(
        dataset,
        0.4,
        false
    );

    if (split.train.num_samples() != 3) {
        throw std::runtime_error("expected train.num_samples() == 3");
    }

    if (split.test.num_samples() != 2) {
        throw std::runtime_error("expected test.num_samples() == 2");
    }
}

void test_train_test_split_preserves_feature_count() {
    const ml::SupervisedDataset dataset = make_split_test_dataset();

    const ml::TrainTestSplit split = ml::train_test_split(
        dataset,
        0.4,
        false
    );

    if (split.train.num_features() != 2) {
        throw std::runtime_error("expected train.num_features() == 2");
    }

    if (split.test.num_features() != 2) {
        throw std::runtime_error("expected test.num_features() == 2");
    }
}

void test_train_test_split_preserves_alignment_without_shuffle() {
    const ml::SupervisedDataset dataset = make_split_test_dataset();

    const ml::TrainTestSplit split = ml::train_test_split(
        dataset,
        0.4,
        false
    );

    ml::Matrix expected_X_train(3, 2);
    expected_X_train << 1.0, 10.0,
                        2.0, 20.0,
                        3.0, 30.0;

    ml::Vector expected_y_train(3);
    expected_y_train << 100.0, 200.0, 300.0;

    ml::Matrix expected_X_test(2, 2);
    expected_X_test << 4.0, 40.0,
                       5.0, 50.0;

    ml::Vector expected_y_test(2);
    expected_y_test << 400.0, 500.0;

    test::assert_matrix_almost_equal(
        split.train.X,
        expected_X_train,
        "test_train_test_split_preserves_alignment_without_shuffle train X"
    );

    test::assert_vector_almost_equal(
        split.train.y,
        expected_y_train,
        "test_train_test_split_preserves_alignment_without_shuffle train y"
    );

    test::assert_matrix_almost_equal(
        split.test.X,
        expected_X_test,
        "test_train_test_split_preserves_alignment_without_shuffle test X"
    );

    test::assert_vector_almost_equal(
        split.test.y,
        expected_y_test,
        "test_train_test_split_preserves_alignment_without_shuffle test y"
    );
}

void test_train_test_split_is_reproducible_with_same_seed() {
    const ml::SupervisedDataset dataset = make_split_test_dataset();

    const ml::TrainTestSplit split_a = ml::train_test_split(
        dataset,
        0.4,
        true,
        123
    );

    const ml::TrainTestSplit split_b = ml::train_test_split(
        dataset,
        0.4,
        true,
        123
    );

    test::assert_matrix_almost_equal(
        split_a.train.X,
        split_b.train.X,
        "test_train_test_split_is_reproducible_with_same_seed train X"
    );

    test::assert_vector_almost_equal(
        split_a.train.y,
        split_b.train.y,
        "test_train_test_split_is_reproducible_with_same_seed train y"
    );

    test::assert_matrix_almost_equal(
        split_a.test.X,
        split_b.test.X,
        "test_train_test_split_is_reproducible_with_same_seed test X"
    );

    test::assert_vector_almost_equal(
        split_a.test.y,
        split_b.test.y,
        "test_train_test_split_is_reproducible_with_same_seed test y"
    );
}

void test_train_test_split_rejects_zero_test_ratio() {
    const ml::SupervisedDataset dataset = make_split_test_dataset();

    static_cast<void>(ml::train_test_split(dataset, 0.0, false));
}

void test_train_test_split_rejects_one_test_ratio() {
    const ml::SupervisedDataset dataset = make_split_test_dataset();

    static_cast<void>(ml::train_test_split(dataset, 1.0, false));
}

void test_train_test_split_rejects_too_small_test_set() {
    const ml::SupervisedDataset dataset = make_split_test_dataset();

    static_cast<void>(ml::train_test_split(dataset, 0.1, false));
}

void test_train_test_split_rejects_dataset_with_fewer_than_two_samples() {
    ml::Matrix X(1, 2);
    X << 1.0, 10.0;

    ml::Vector y(1);
    y << 100.0;

    const ml::SupervisedDataset dataset(X, y);

    static_cast<void>(ml::train_test_split(dataset, 0.5, false));
}

void run_train_test_split_tests() {
    std::cout << "\n[Phase 2.2] Train/test split tests\n\n";

    test::expect_no_throw(
        "train_test_split creates expected sizes",
        test_train_test_split_creates_expected_sizes
    );

    test::expect_no_throw(
        "train_test_split preserves feature count",
        test_train_test_split_preserves_feature_count
    );

    test::expect_no_throw(
        "train_test_split preserves X/y alignment without shuffle",
        test_train_test_split_preserves_alignment_without_shuffle
    );

    test::expect_no_throw(
        "train_test_split is reproducible with same seed",
        test_train_test_split_is_reproducible_with_same_seed
    );

    test::expect_invalid_argument(
        "train_test_split rejects test_ratio 0",
        test_train_test_split_rejects_zero_test_ratio
    );

    test::expect_invalid_argument(
        "train_test_split rejects test_ratio 1",
        test_train_test_split_rejects_one_test_ratio
    );

    test::expect_invalid_argument(
        "train_test_split rejects split with empty test set",
        test_train_test_split_rejects_too_small_test_set
    );

    test::expect_invalid_argument(
        "train_test_split rejects dataset with fewer than 2 samples",
        test_train_test_split_rejects_dataset_with_fewer_than_two_samples
    );
}

ml::SupervisedDataset make_train_validation_test_split_dataset() {
    ml::Matrix X(10, 2);
    X << 1.0, 10.0,
         2.0, 20.0,
         3.0, 30.0,
         4.0, 40.0,
         5.0, 50.0,
         6.0, 60.0,
         7.0, 70.0,
         8.0, 80.0,
         9.0, 90.0,
         10.0, 100.0;

    ml::Vector y(10);
    y << 100.0, 200.0, 300.0, 400.0, 500.0,
         600.0, 700.0, 800.0, 900.0, 1000.0;

    return ml::SupervisedDataset(X, y);
}

void test_train_validation_test_split_creates_expected_sizes() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    const ml::TrainValidationTestSplit split = ml::train_validation_test_split(
        dataset,
        0.2,
        0.2,
        false
    );

    if (split.train.num_samples() != 6) {
        throw std::runtime_error("expected train.num_samples() == 6");
    }

    if (split.validation.num_samples() != 2) {
        throw std::runtime_error("expected validation.num_samples() == 2");
    }

    if (split.test.num_samples() != 2) {
        throw std::runtime_error("expected test.num_samples() == 2");
    }
}

void test_train_validation_test_split_preserves_feature_count() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    const ml::TrainValidationTestSplit split = ml::train_validation_test_split(
        dataset,
        0.2,
        0.2,
        false
    );

    if (split.train.num_features() != 2) {
        throw std::runtime_error("expected train.num_features() == 2");
    }

    if (split.validation.num_features() != 2) {
        throw std::runtime_error("expected validation.num_features() == 2");
    }

    if (split.test.num_features() != 2) {
        throw std::runtime_error("expected test.num_features() == 2");
    }
}

void test_train_validation_test_split_preserves_alignment_without_shuffle() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    const ml::TrainValidationTestSplit split = ml::train_validation_test_split(
        dataset,
        0.2,
        0.2,
        false
    );

    ml::Matrix expected_X_train(6, 2);
    expected_X_train << 1.0, 10.0,
                        2.0, 20.0,
                        3.0, 30.0,
                        4.0, 40.0,
                        5.0, 50.0,
                        6.0, 60.0;

    ml::Vector expected_y_train(6);
    expected_y_train << 100.0, 200.0, 300.0, 400.0, 500.0, 600.0;

    ml::Matrix expected_X_validation(2, 2);
    expected_X_validation << 7.0, 70.0,
                             8.0, 80.0;

    ml::Vector expected_y_validation(2);
    expected_y_validation << 700.0, 800.0;

    ml::Matrix expected_X_test(2, 2);
    expected_X_test << 9.0, 90.0,
                       10.0, 100.0;

    ml::Vector expected_y_test(2);
    expected_y_test << 900.0, 1000.0;

    test::assert_matrix_almost_equal(
        split.train.X,
        expected_X_train,
        "test_train_validation_test_split_preserves_alignment_without_shuffle train X"
    );

    test::assert_vector_almost_equal(
        split.train.y,
        expected_y_train,
        "test_train_validation_test_split_preserves_alignment_without_shuffle train y"
    );

    test::assert_matrix_almost_equal(
        split.validation.X,
        expected_X_validation,
        "test_train_validation_test_split_preserves_alignment_without_shuffle validation X"
    );

    test::assert_vector_almost_equal(
        split.validation.y,
        expected_y_validation,
        "test_train_validation_test_split_preserves_alignment_without_shuffle validation y"
    );

    test::assert_matrix_almost_equal(
        split.test.X,
        expected_X_test,
        "test_train_validation_test_split_preserves_alignment_without_shuffle test X"
    );

    test::assert_vector_almost_equal(
        split.test.y,
        expected_y_test,
        "test_train_validation_test_split_preserves_alignment_without_shuffle test y"
    );
}

void test_train_validation_test_split_is_reproducible_with_same_seed() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    const ml::TrainValidationTestSplit split_a = ml::train_validation_test_split(
        dataset,
        0.2,
        0.2,
        true,
        123
    );

    const ml::TrainValidationTestSplit split_b = ml::train_validation_test_split(
        dataset,
        0.2,
        0.2,
        true,
        123
    );

    test::assert_matrix_almost_equal(
        split_a.train.X,
        split_b.train.X,
        "test_train_validation_test_split_is_reproducible_with_same_seed train X"
    );

    test::assert_vector_almost_equal(
        split_a.train.y,
        split_b.train.y,
        "test_train_validation_test_split_is_reproducible_with_same_seed train y"
    );

    test::assert_matrix_almost_equal(
        split_a.validation.X,
        split_b.validation.X,
        "test_train_validation_test_split_is_reproducible_with_same_seed validation X"
    );

    test::assert_vector_almost_equal(
        split_a.validation.y,
        split_b.validation.y,
        "test_train_validation_test_split_is_reproducible_with_same_seed validation y"
    );

    test::assert_matrix_almost_equal(
        split_a.test.X,
        split_b.test.X,
        "test_train_validation_test_split_is_reproducible_with_same_seed test X"
    );

    test::assert_vector_almost_equal(
        split_a.test.y,
        split_b.test.y,
        "test_train_validation_test_split_is_reproducible_with_same_seed test y"
    );
}

void test_train_validation_test_split_rejects_zero_validation_ratio() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    static_cast<void>(ml::train_validation_test_split(dataset, 0.0, 0.2, false));
}

void test_train_validation_test_split_rejects_zero_test_ratio() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    static_cast<void>(ml::train_validation_test_split(dataset, 0.2, 0.0, false));
}

void test_train_validation_test_split_rejects_ratio_sum_one() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    static_cast<void>(ml::train_validation_test_split(dataset, 0.5, 0.5, false));
}

void test_train_validation_test_split_rejects_ratio_sum_greater_than_one() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    static_cast<void>(ml::train_validation_test_split(dataset, 0.6, 0.5, false));
}

void test_train_validation_test_split_rejects_empty_validation_set() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    static_cast<void>(ml::train_validation_test_split(dataset, 0.05, 0.2, false));
}

void test_train_validation_test_split_rejects_empty_test_set() {
    const ml::SupervisedDataset dataset = make_train_validation_test_split_dataset();

    static_cast<void>(ml::train_validation_test_split(dataset, 0.2, 0.05, false));
}

void test_train_validation_test_split_rejects_dataset_with_fewer_than_three_samples() {
    ml::Matrix X(2, 2);
    X << 1.0, 10.0,
         2.0, 20.0;

    ml::Vector y(2);
    y << 100.0, 200.0;

    const ml::SupervisedDataset dataset(X, y);

    static_cast<void>(ml::train_validation_test_split(dataset, 0.25, 0.25, false));
}

void run_train_validation_test_split_tests() {
    std::cout << "\n[Phase 2.2] Train/validation/test split tests\n\n";

    test::expect_no_throw(
        "train_validation_test_split creates expected sizes",
        test_train_validation_test_split_creates_expected_sizes
    );

    test::expect_no_throw(
        "train_validation_test_split preserves feature count",
        test_train_validation_test_split_preserves_feature_count
    );

    test::expect_no_throw(
        "train_validation_test_split preserves X/y alignment without shuffle",
        test_train_validation_test_split_preserves_alignment_without_shuffle
    );

    test::expect_no_throw(
        "train_validation_test_split is reproducible with same seed",
        test_train_validation_test_split_is_reproducible_with_same_seed
    );

    test::expect_invalid_argument(
        "train_validation_test_split rejects validation_ratio 0",
        test_train_validation_test_split_rejects_zero_validation_ratio
    );

    test::expect_invalid_argument(
        "train_validation_test_split rejects test_ratio 0",
        test_train_validation_test_split_rejects_zero_test_ratio
    );

    test::expect_invalid_argument(
        "train_validation_test_split rejects validation_ratio + test_ratio == 1",
        test_train_validation_test_split_rejects_ratio_sum_one
    );

    test::expect_invalid_argument(
        "train_validation_test_split rejects validation_ratio + test_ratio > 1",
        test_train_validation_test_split_rejects_ratio_sum_greater_than_one
    );

    test::expect_invalid_argument(
        "train_validation_test_split rejects split with empty validation set",
        test_train_validation_test_split_rejects_empty_validation_set
    );

    test::expect_invalid_argument(
        "train_validation_test_split rejects split with empty test set",
        test_train_validation_test_split_rejects_empty_test_set
    );

    test::expect_invalid_argument(
        "train_validation_test_split rejects dataset with fewer than 3 samples",
        test_train_validation_test_split_rejects_dataset_with_fewer_than_three_samples
    );
}

ml::SupervisedDataset make_k_fold_split_dataset() {
    ml::Matrix X(6, 2);
    X << 1.0, 10.0,
         2.0, 20.0,
         3.0, 30.0,
         4.0, 40.0,
         5.0, 50.0,
         6.0, 60.0;

    ml::Vector y(6);
    y << 100.0, 200.0, 300.0, 400.0, 500.0, 600.0;

    return ml::SupervisedDataset(X, y);
}

void test_k_fold_split_creates_k_folds() {
    const ml::SupervisedDataset dataset = make_k_fold_split_dataset();

    const std::vector<ml::FoldSplit> folds = ml::k_fold_split(
        dataset,
        3,
        false
    );

    if (folds.size() != 3) {
        throw std::runtime_error("expected folds.size() == 3");
    }
}

void test_k_fold_split_creates_expected_validation_sizes() {
    const ml::SupervisedDataset dataset = make_k_fold_split_dataset();

    const std::vector<ml::FoldSplit> folds = ml::k_fold_split(
        dataset,
        3,
        false
    );

    for (std::size_t i = 0; i < folds.size(); ++i) {
        if (folds[i].validation.num_samples() != 2) {
            throw std::runtime_error("expected each validation fold to contain 2 samples");
        }

        if (folds[i].train.num_samples() != 4) {
            throw std::runtime_error("expected each training fold to contain 4 samples");
        }
    }
}

void test_k_fold_split_preserves_feature_count() {
    const ml::SupervisedDataset dataset = make_k_fold_split_dataset();

    const std::vector<ml::FoldSplit> folds = ml::k_fold_split(
        dataset,
        3,
        false
    );

    for (const ml::FoldSplit& fold : folds) {
        if (fold.train.num_features() != 2) {
            throw std::runtime_error("expected train.num_features() == 2");
        }

        if (fold.validation.num_features() != 2) {
            throw std::runtime_error("expected validation.num_features() == 2");
        }
    }
}

void test_k_fold_split_preserves_alignment_without_shuffle() {
    const ml::SupervisedDataset dataset = make_k_fold_split_dataset();

    const std::vector<ml::FoldSplit> folds = ml::k_fold_split(
        dataset,
        3,
        false
    );

    ml::Matrix expected_fold0_X_validation(2, 2);
    expected_fold0_X_validation << 1.0, 10.0,
                                   2.0, 20.0;

    ml::Vector expected_fold0_y_validation(2);
    expected_fold0_y_validation << 100.0, 200.0;

    ml::Matrix expected_fold0_X_train(4, 2);
    expected_fold0_X_train << 3.0, 30.0,
                              4.0, 40.0,
                              5.0, 50.0,
                              6.0, 60.0;

    ml::Vector expected_fold0_y_train(4);
    expected_fold0_y_train << 300.0, 400.0, 500.0, 600.0;

    ml::Matrix expected_fold1_X_validation(2, 2);
    expected_fold1_X_validation << 3.0, 30.0,
                                   4.0, 40.0;

    ml::Vector expected_fold1_y_validation(2);
    expected_fold1_y_validation << 300.0, 400.0;

    ml::Matrix expected_fold2_X_validation(2, 2);
    expected_fold2_X_validation << 5.0, 50.0,
                                   6.0, 60.0;

    ml::Vector expected_fold2_y_validation(2);
    expected_fold2_y_validation << 500.0, 600.0;

    test::assert_matrix_almost_equal(
        folds[0].validation.X,
        expected_fold0_X_validation,
        "test_k_fold_split_preserves_alignment_without_shuffle fold 0 validation X"
    );

    test::assert_vector_almost_equal(
        folds[0].validation.y,
        expected_fold0_y_validation,
        "test_k_fold_split_preserves_alignment_without_shuffle fold 0 validation y"
    );

    test::assert_matrix_almost_equal(
        folds[0].train.X,
        expected_fold0_X_train,
        "test_k_fold_split_preserves_alignment_without_shuffle fold 0 train X"
    );

    test::assert_vector_almost_equal(
        folds[0].train.y,
        expected_fold0_y_train,
        "test_k_fold_split_preserves_alignment_without_shuffle fold 0 train y"
    );

    test::assert_matrix_almost_equal(
        folds[1].validation.X,
        expected_fold1_X_validation,
        "test_k_fold_split_preserves_alignment_without_shuffle fold 1 validation X"
    );

    test::assert_vector_almost_equal(
        folds[1].validation.y,
        expected_fold1_y_validation,
        "test_k_fold_split_preserves_alignment_without_shuffle fold 1 validation y"
    );

    test::assert_matrix_almost_equal(
        folds[2].validation.X,
        expected_fold2_X_validation,
        "test_k_fold_split_preserves_alignment_without_shuffle fold 2 validation X"
    );

    test::assert_vector_almost_equal(
        folds[2].validation.y,
        expected_fold2_y_validation,
        "test_k_fold_split_preserves_alignment_without_shuffle fold 2 validation y"
    );
}

void test_k_fold_split_handles_uneven_fold_sizes() {
    ml::Matrix X(5, 2);
    X << 1.0, 10.0,
         2.0, 20.0,
         3.0, 30.0,
         4.0, 40.0,
         5.0, 50.0;

    ml::Vector y(5);
    y << 100.0, 200.0, 300.0, 400.0, 500.0;

    const ml::SupervisedDataset dataset(X, y);

    const std::vector<ml::FoldSplit> folds = ml::k_fold_split(
        dataset,
        3,
        false
    );

    if (folds[0].validation.num_samples() != 2) {
        throw std::runtime_error("expected fold 0 validation size == 2");
    }

    if (folds[1].validation.num_samples() != 2) {
        throw std::runtime_error("expected fold 1 validation size == 2");
    }

    if (folds[2].validation.num_samples() != 1) {
        throw std::runtime_error("expected fold 2 validation size == 1");
    }
}

void test_k_fold_split_is_reproducible_with_same_seed() {
    const ml::SupervisedDataset dataset = make_k_fold_split_dataset();

    const std::vector<ml::FoldSplit> folds_a = ml::k_fold_split(
        dataset,
        3,
        true,
        123
    );

    const std::vector<ml::FoldSplit> folds_b = ml::k_fold_split(
        dataset,
        3,
        true,
        123
    );

    if (folds_a.size() != folds_b.size()) {
        throw std::runtime_error("expected reproducible folds to have the same size");
    }

    for (std::size_t i = 0; i < folds_a.size(); ++i) {
        test::assert_matrix_almost_equal(
            folds_a[i].train.X,
            folds_b[i].train.X,
            "test_k_fold_split_is_reproducible_with_same_seed train X"
        );

        test::assert_vector_almost_equal(
            folds_a[i].train.y,
            folds_b[i].train.y,
            "test_k_fold_split_is_reproducible_with_same_seed train y"
        );

        test::assert_matrix_almost_equal(
            folds_a[i].validation.X,
            folds_b[i].validation.X,
            "test_k_fold_split_is_reproducible_with_same_seed validation X"
        );

        test::assert_vector_almost_equal(
            folds_a[i].validation.y,
            folds_b[i].validation.y,
            "test_k_fold_split_is_reproducible_with_same_seed validation y"
        );
    }
}

void test_k_fold_split_rejects_k_less_than_two() {
    const ml::SupervisedDataset dataset = make_k_fold_split_dataset();

    static_cast<void>(ml::k_fold_split(dataset, 1, false));
}

void test_k_fold_split_rejects_k_greater_than_num_samples() {
    const ml::SupervisedDataset dataset = make_k_fold_split_dataset();

    static_cast<void>(ml::k_fold_split(dataset, 7, false));
}

void test_k_fold_split_rejects_dataset_with_fewer_than_two_samples() {
    ml::Matrix X(1, 2);
    X << 1.0, 10.0;

    ml::Vector y(1);
    y << 100.0;

    const ml::SupervisedDataset dataset(X, y);

    static_cast<void>(ml::k_fold_split(dataset, 2, false));
}

void run_k_fold_split_tests() {
    std::cout << "\n[Phase 2.3] K-fold cross-validation split tests\n\n";

    test::expect_no_throw(
        "k_fold_split creates k folds",
        test_k_fold_split_creates_k_folds
    );

    test::expect_no_throw(
        "k_fold_split creates expected validation sizes",
        test_k_fold_split_creates_expected_validation_sizes
    );

    test::expect_no_throw(
        "k_fold_split preserves feature count",
        test_k_fold_split_preserves_feature_count
    );

    test::expect_no_throw(
        "k_fold_split preserves X/y alignment without shuffle",
        test_k_fold_split_preserves_alignment_without_shuffle
    );

    test::expect_no_throw(
        "k_fold_split handles uneven fold sizes",
        test_k_fold_split_handles_uneven_fold_sizes
    );

    test::expect_no_throw(
        "k_fold_split is reproducible with same seed",
        test_k_fold_split_is_reproducible_with_same_seed
    );

    test::expect_invalid_argument(
        "k_fold_split rejects k < 2",
        test_k_fold_split_rejects_k_less_than_two
    );

    test::expect_invalid_argument(
        "k_fold_split rejects k > num_samples",
        test_k_fold_split_rejects_k_greater_than_num_samples
    );

    test::expect_invalid_argument(
        "k_fold_split rejects dataset with fewer than 2 samples",
        test_k_fold_split_rejects_dataset_with_fewer_than_two_samples
    );
}

// ---- Preprocessing pipeline tests ----

void test_standard_scaler_fit_learns_train_statistics() {
    ml::Matrix X_train(3, 2);
    X_train << 1.0, 10.0,
               2.0, 20.0,
               3.0, 30.0;

    const ml::StandardScaler scaler = ml::StandardScaler::fit(X_train);

    ml::Vector expected_means(2);
    expected_means << 2.0, 20.0;

    const double expected_std_col0 = std::sqrt(2.0 / 3.0);
    const double expected_std_col1 = std::sqrt(200.0 / 3.0);

    ml::Vector expected_stds(2);
    expected_stds << expected_std_col0, expected_std_col1;

    test::assert_vector_almost_equal(
        scaler.means,
        expected_means,
        "test_standard_scaler_fit_learns_train_statistics means"
    );

    test::assert_vector_almost_equal(
        scaler.standard_deviations,
        expected_stds,
        "test_standard_scaler_fit_learns_train_statistics standard deviations"
    );
}

void test_standard_scaler_transform_applies_train_statistics() {
    ml::Matrix X_train(3, 2);
    X_train << 1.0, 10.0,
               2.0, 20.0,
               3.0, 30.0;

    const ml::StandardScaler scaler = ml::StandardScaler::fit(X_train);

    ml::Matrix X_new(2, 2);
    X_new << 2.0, 20.0,
             4.0, 40.0;

    const ml::Matrix result = scaler.transform(X_new);

    const double z = std::sqrt(1.5);

    ml::Matrix expected(2, 2);
    expected << 0.0, 0.0,
                z * 2.0, z * 2.0;

    test::assert_matrix_almost_equal(
        result,
        expected,
        "test_standard_scaler_transform_applies_train_statistics"
    );
}

void test_standard_scaler_fit_transform_matches_transform_on_training_data() {
    ml::Matrix X_train(3, 2);
    X_train << 1.0, 10.0,
               2.0, 20.0,
               3.0, 30.0;

    ml::StandardScaler scaler;
    const ml::Matrix fit_transform_result = scaler.fit_transform(X_train);
    const ml::Matrix transform_result = scaler.transform(X_train);

    test::assert_matrix_almost_equal(
        fit_transform_result,
        transform_result,
        "test_standard_scaler_fit_transform_matches_transform_on_training_data"
    );
}

void test_standard_scaler_handles_constant_column() {
    ml::Matrix X_train(3, 2);
    X_train << 5.0, 1.0,
               5.0, 2.0,
               5.0, 3.0;

    const ml::StandardScaler scaler = ml::StandardScaler::fit(X_train);
    const ml::Matrix result = scaler.transform(X_train);

    const double z = std::sqrt(1.5);

    ml::Matrix expected(3, 2);
    expected << 0.0, -z,
                0.0, 0.0,
                0.0, z;

    test::assert_matrix_almost_equal(
        result,
        expected,
        "test_standard_scaler_handles_constant_column"
    );
}

void test_standard_scaler_rejects_mismatched_feature_count() {
    ml::Matrix X_train(3, 2);
    X_train << 1.0, 10.0,
               2.0, 20.0,
               3.0, 30.0;

    const ml::StandardScaler scaler = ml::StandardScaler::fit(X_train);

    ml::Matrix X_bad(2, 3);
    X_bad << 1.0, 2.0, 3.0,
             4.0, 5.0, 6.0;

    static_cast<void>(scaler.transform(X_bad));
}

void test_min_max_scaler_fit_learns_train_statistics() {
    ml::Matrix X_train(3, 2);
    X_train << 1.0, 10.0,
               2.0, 20.0,
               3.0, 30.0;

    const ml::MinMaxScaler scaler = ml::MinMaxScaler::fit(X_train);

    ml::Vector expected_mins(2);
    expected_mins << 1.0, 10.0;

    ml::Vector expected_ranges(2);
    expected_ranges << 2.0, 20.0;

    test::assert_vector_almost_equal(
        scaler.mins,
        expected_mins,
        "test_min_max_scaler_fit_learns_train_statistics mins"
    );

    test::assert_vector_almost_equal(
        scaler.ranges,
        expected_ranges,
        "test_min_max_scaler_fit_learns_train_statistics ranges"
    );
}

void test_min_max_scaler_transform_applies_train_statistics() {
    ml::Matrix X_train(3, 2);
    X_train << 1.0, 10.0,
               2.0, 20.0,
               3.0, 30.0;

    const ml::MinMaxScaler scaler = ml::MinMaxScaler::fit(X_train);

    ml::Matrix X_new(2, 2);
    X_new << 2.0, 20.0,
             4.0, 40.0;

    const ml::Matrix result = scaler.transform(X_new);

    ml::Matrix expected(2, 2);
    expected << 0.5, 0.5,
                1.5, 1.5;

    test::assert_matrix_almost_equal(
        result,
        expected,
        "test_min_max_scaler_transform_applies_train_statistics"
    );
}

void test_min_max_scaler_fit_transform_matches_transform_on_training_data() {
    ml::Matrix X_train(3, 2);
    X_train << 1.0, 10.0,
               2.0, 20.0,
               3.0, 30.0;

    ml::MinMaxScaler scaler;
    const ml::Matrix fit_transform_result = scaler.fit_transform(X_train);
    const ml::Matrix transform_result = scaler.transform(X_train);

    test::assert_matrix_almost_equal(
        fit_transform_result,
        transform_result,
        "test_min_max_scaler_fit_transform_matches_transform_on_training_data"
    );
}

void test_min_max_scaler_handles_constant_column() {
    ml::Matrix X_train(3, 2);
    X_train << 5.0, 1.0,
               5.0, 2.0,
               5.0, 3.0;

    const ml::MinMaxScaler scaler = ml::MinMaxScaler::fit(X_train);
    const ml::Matrix result = scaler.transform(X_train);

    ml::Matrix expected(3, 2);
    expected << 0.0, 0.0,
                0.0, 0.5,
                0.0, 1.0;

    test::assert_matrix_almost_equal(
        result,
        expected,
        "test_min_max_scaler_handles_constant_column"
    );
}

void test_min_max_scaler_rejects_mismatched_feature_count() {
    ml::Matrix X_train(3, 2);
    X_train << 1.0, 10.0,
               2.0, 20.0,
               3.0, 30.0;

    const ml::MinMaxScaler scaler = ml::MinMaxScaler::fit(X_train);

    ml::Matrix X_bad(2, 3);
    X_bad << 1.0, 2.0, 3.0,
             4.0, 5.0, 6.0;

    static_cast<void>(scaler.transform(X_bad));
}

void run_preprocessing_pipeline_tests() {
    std::cout << "\n[Phase 2.4] Preprocessing pipeline tests\n\n";

    test::expect_no_throw(
        "StandardScaler fit learns train statistics",
        test_standard_scaler_fit_learns_train_statistics
    );

    test::expect_no_throw(
        "StandardScaler transform applies train statistics",
        test_standard_scaler_transform_applies_train_statistics
    );

    test::expect_no_throw(
        "StandardScaler fit_transform matches transform on training data",
        test_standard_scaler_fit_transform_matches_transform_on_training_data
    );

    test::expect_no_throw(
        "StandardScaler handles constant column",
        test_standard_scaler_handles_constant_column
    );

    test::expect_invalid_argument(
        "StandardScaler rejects mismatched feature count",
        test_standard_scaler_rejects_mismatched_feature_count
    );

    test::expect_no_throw(
        "MinMaxScaler fit learns train statistics",
        test_min_max_scaler_fit_learns_train_statistics
    );

    test::expect_no_throw(
        "MinMaxScaler transform applies train statistics",
        test_min_max_scaler_transform_applies_train_statistics
    );

    test::expect_no_throw(
        "MinMaxScaler fit_transform matches transform on training data",
        test_min_max_scaler_fit_transform_matches_transform_on_training_data
    );

    test::expect_no_throw(
        "MinMaxScaler handles constant column",
        test_min_max_scaler_handles_constant_column
    );

    test::expect_invalid_argument(
        "MinMaxScaler rejects mismatched feature count",
        test_min_max_scaler_rejects_mismatched_feature_count
    );
}

// ---- Baseline evaluation flow tests ----

void test_mean_absolute_error_computes_expected_value() {
    ml::Vector predictions(3);
    predictions << 2.0, 5.0, 5.0;

    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    const double result = ml::mean_absolute_error(predictions, targets);

    test::assert_almost_equal(
        result,
        2.0 / 3.0,
        "test_mean_absolute_error_computes_expected_value"
    );
}

void test_root_mean_squared_error_computes_expected_value() {
    ml::Vector predictions(3);
    predictions << 2.0, 5.0, 5.0;

    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    const double result = ml::root_mean_squared_error(predictions, targets);

    test::assert_almost_equal(
        result,
        std::sqrt(2.0 / 3.0),
        "test_root_mean_squared_error_computes_expected_value"
    );
}

void test_r2_score_computes_expected_value() {
    ml::Vector predictions(3);
    predictions << 2.0, 5.0, 5.0;

    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    const double result = ml::r2_score(predictions, targets);

    test::assert_almost_equal(
        result,
        0.75,
        "test_r2_score_computes_expected_value"
    );
}

void test_r2_score_rejects_constant_targets() {
    ml::Vector predictions(3);
    predictions << 2.0, 2.0, 2.0;

    ml::Vector targets(3);
    targets << 4.0, 4.0, 4.0;

    static_cast<void>(ml::r2_score(predictions, targets));
}

void test_regression_metrics_reject_mismatched_vectors() {
    ml::Vector predictions(3);
    predictions << 2.0, 5.0, 5.0;

    ml::Vector targets(2);
    targets << 2.0, 4.0;

    static_cast<void>(ml::mean_absolute_error(predictions, targets));
}

void test_mean_regressor_fit_stores_target_mean() {
    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    ml::MeanRegressor baseline;
    baseline.fit(targets);

    test::assert_almost_equal(
        baseline.value(),
        4.0,
        "test_mean_regressor_fit_stores_target_mean"
    );

    if (!baseline.is_fitted()) {
        throw std::runtime_error("expected MeanRegressor to be fitted");
    }
}

void test_mean_regressor_predict_returns_constant_predictions() {
    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    ml::MeanRegressor baseline;
    baseline.fit(targets);

    const ml::Vector predictions = baseline.predict(3);

    ml::Vector expected(3);
    expected << 4.0, 4.0, 4.0;

    test::assert_vector_almost_equal(
        predictions,
        expected,
        "test_mean_regressor_predict_returns_constant_predictions"
    );
}

void test_mean_regressor_rejects_predict_before_fit() {
    ml::MeanRegressor baseline;

    static_cast<void>(baseline.predict(3));
}

void test_mean_regressor_rejects_non_positive_prediction_size() {
    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    ml::MeanRegressor baseline;
    baseline.fit(targets);

    static_cast<void>(baseline.predict(0));
}

void test_evaluate_regression_returns_expected_metrics() {
    ml::Vector predictions(3);
    predictions << 2.0, 5.0, 5.0;

    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    const ml::RegressionEvaluation result = ml::evaluate_regression(
        predictions,
        targets
    );

    test::assert_almost_equal(
        result.mse,
        2.0 / 3.0,
        "test_evaluate_regression_returns_expected_metrics mse"
    );

    test::assert_almost_equal(
        result.rmse,
        std::sqrt(2.0 / 3.0),
        "test_evaluate_regression_returns_expected_metrics rmse"
    );

    test::assert_almost_equal(
        result.mae,
        2.0 / 3.0,
        "test_evaluate_regression_returns_expected_metrics mae"
    );

    test::assert_almost_equal(
        result.r2,
        0.75,
        "test_evaluate_regression_returns_expected_metrics r2"
    );
}

void test_compare_regression_to_baseline_evaluates_both_predictors() {
    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    ml::Vector baseline_predictions(3);
    baseline_predictions << 4.0, 4.0, 4.0;

    ml::Vector model_predictions(3);
    model_predictions << 2.0, 5.0, 5.0;

    const ml::BaselineComparison comparison = ml::compare_regression_to_baseline(
        baseline_predictions,
        model_predictions,
        targets
    );

    test::assert_almost_equal(
        comparison.baseline.mse,
        8.0 / 3.0,
        "test_compare_regression_to_baseline_evaluates_both_predictors baseline mse"
    );

    test::assert_almost_equal(
        comparison.model.mse,
        2.0 / 3.0,
        "test_compare_regression_to_baseline_evaluates_both_predictors model mse"
    );

    test::assert_almost_equal(
        comparison.baseline.r2,
        0.0,
        "test_compare_regression_to_baseline_evaluates_both_predictors baseline r2"
    );

    test::assert_almost_equal(
        comparison.model.r2,
        0.75,
        "test_compare_regression_to_baseline_evaluates_both_predictors model r2"
    );
}

void test_compare_regression_to_baseline_rejects_mismatched_model_predictions() {
    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    ml::Vector baseline_predictions(3);
    baseline_predictions << 4.0, 4.0, 4.0;

    ml::Vector model_predictions(2);
    model_predictions << 2.0, 5.0;

    static_cast<void>(ml::compare_regression_to_baseline(
        baseline_predictions,
        model_predictions,
        targets
    ));
}

void run_baseline_evaluation_flow_tests() {
    std::cout << "\n[Phase 2.5] Baseline evaluation flow tests\n\n";

    test::expect_no_throw(
        "mean_absolute_error computes expected value",
        test_mean_absolute_error_computes_expected_value
    );

    test::expect_no_throw(
        "root_mean_squared_error computes expected value",
        test_root_mean_squared_error_computes_expected_value
    );

    test::expect_no_throw(
        "r2_score computes expected value",
        test_r2_score_computes_expected_value
    );

    test::expect_invalid_argument(
        "r2_score rejects constant targets",
        test_r2_score_rejects_constant_targets
    );

    test::expect_invalid_argument(
        "regression metrics reject mismatched vectors",
        test_regression_metrics_reject_mismatched_vectors
    );

    test::expect_no_throw(
        "MeanRegressor fit stores target mean",
        test_mean_regressor_fit_stores_target_mean
    );

    test::expect_no_throw(
        "MeanRegressor predict returns constant predictions",
        test_mean_regressor_predict_returns_constant_predictions
    );

    test::expect_invalid_argument(
        "MeanRegressor rejects predict before fit",
        test_mean_regressor_rejects_predict_before_fit
    );

    test::expect_invalid_argument(
        "MeanRegressor rejects non-positive prediction size",
        test_mean_regressor_rejects_non_positive_prediction_size
    );

    test::expect_no_throw(
        "evaluate_regression returns expected metrics",
        test_evaluate_regression_returns_expected_metrics
    );

    test::expect_no_throw(
        "compare_regression_to_baseline evaluates both predictors",
        test_compare_regression_to_baseline_evaluates_both_predictors
    );

    test::expect_invalid_argument(
        "compare_regression_to_baseline rejects mismatched model predictions",
        test_compare_regression_to_baseline_rejects_mismatched_model_predictions
    );

}

// ---- Reusable evaluation harness tests ----

ml::RegressionEvaluationInput make_regression_evaluation_input() {
    ml::Vector targets(3);
    targets << 2.0, 4.0, 6.0;

    ml::Vector baseline_predictions(3);
    baseline_predictions << 4.0, 4.0, 4.0;

    ml::Vector model_predictions(3);
    model_predictions << 2.0, 5.0, 5.0;

    return ml::RegressionEvaluationInput{
        targets,
        baseline_predictions,
        model_predictions,
        "TestModel",
        "MeanRegressor"
    };
}

void test_run_regression_evaluation_returns_names_and_metrics() {
    const ml::RegressionEvaluationInput input = make_regression_evaluation_input();

    const ml::RegressionEvaluationReport report = ml::run_regression_evaluation(input);

    if (report.model_name != "TestModel") {
        throw std::runtime_error("expected model_name == TestModel");
    }

    if (report.baseline_name != "MeanRegressor") {
        throw std::runtime_error("expected baseline_name == MeanRegressor");
    }

    test::assert_almost_equal(
        report.comparison.baseline.mse,
        8.0 / 3.0,
        "test_run_regression_evaluation_returns_names_and_metrics baseline mse"
    );

    test::assert_almost_equal(
        report.comparison.model.mse,
        2.0 / 3.0,
        "test_run_regression_evaluation_returns_names_and_metrics model mse"
    );

    test::assert_almost_equal(
        report.comparison.baseline.r2,
        0.0,
        "test_run_regression_evaluation_returns_names_and_metrics baseline r2"
    );

    test::assert_almost_equal(
        report.comparison.model.r2,
        0.75,
        "test_run_regression_evaluation_returns_names_and_metrics model r2"
    );
}

void test_run_regression_evaluation_detects_model_beating_baseline_mse() {
    const ml::RegressionEvaluationReport report = ml::run_regression_evaluation(
        make_regression_evaluation_input()
    );

    if (!report.model_beats_baseline_mse()) {
        throw std::runtime_error("expected model to beat baseline on MSE");
    }
}

void test_run_regression_evaluation_detects_model_beating_baseline_rmse() {
    const ml::RegressionEvaluationReport report = ml::run_regression_evaluation(
        make_regression_evaluation_input()
    );

    if (!report.model_beats_baseline_rmse()) {
        throw std::runtime_error("expected model to beat baseline on RMSE");
    }
}

void test_run_regression_evaluation_detects_model_beating_baseline_mae() {
    const ml::RegressionEvaluationReport report = ml::run_regression_evaluation(
        make_regression_evaluation_input()
    );

    if (!report.model_beats_baseline_mae()) {
        throw std::runtime_error("expected model to beat baseline on MAE");
    }
}

void test_run_regression_evaluation_detects_model_beating_baseline_r2() {
    const ml::RegressionEvaluationReport report = ml::run_regression_evaluation(
        make_regression_evaluation_input()
    );

    if (!report.model_beats_baseline_r2()) {
        throw std::runtime_error("expected model to beat baseline on R2");
    }
}

void test_run_regression_evaluation_rejects_empty_model_name() {
    ml::RegressionEvaluationInput input = make_regression_evaluation_input();
    input.model_name = "";

    static_cast<void>(ml::run_regression_evaluation(input));
}

void test_run_regression_evaluation_rejects_empty_baseline_name() {
    ml::RegressionEvaluationInput input = make_regression_evaluation_input();
    input.baseline_name = "";

    static_cast<void>(ml::run_regression_evaluation(input));
}

void test_run_regression_evaluation_rejects_mismatched_baseline_predictions() {
    ml::RegressionEvaluationInput input = make_regression_evaluation_input();

    ml::Vector bad_baseline_predictions(2);
    bad_baseline_predictions << 4.0, 4.0;
    input.baseline_predictions = bad_baseline_predictions;

    static_cast<void>(ml::run_regression_evaluation(input));
}

void test_run_regression_evaluation_rejects_mismatched_model_predictions() {
    ml::RegressionEvaluationInput input = make_regression_evaluation_input();

    ml::Vector bad_model_predictions(2);
    bad_model_predictions << 2.0, 5.0;
    input.model_predictions = bad_model_predictions;

    static_cast<void>(ml::run_regression_evaluation(input));
}

void run_evaluation_harness_tests() {
    std::cout << "\n[Phase 2.6] Reusable evaluation harness tests\n\n";

    test::expect_no_throw(
        "run_regression_evaluation returns names and metrics",
        test_run_regression_evaluation_returns_names_and_metrics
    );

    test::expect_no_throw(
        "run_regression_evaluation detects model beating baseline on MSE",
        test_run_regression_evaluation_detects_model_beating_baseline_mse
    );

    test::expect_no_throw(
        "run_regression_evaluation detects model beating baseline on RMSE",
        test_run_regression_evaluation_detects_model_beating_baseline_rmse
    );

    test::expect_no_throw(
        "run_regression_evaluation detects model beating baseline on MAE",
        test_run_regression_evaluation_detects_model_beating_baseline_mae
    );

    test::expect_no_throw(
        "run_regression_evaluation detects model beating baseline on R2",
        test_run_regression_evaluation_detects_model_beating_baseline_r2
    );

    test::expect_invalid_argument(
        "run_regression_evaluation rejects empty model name",
        test_run_regression_evaluation_rejects_empty_model_name
    );

    test::expect_invalid_argument(
        "run_regression_evaluation rejects empty baseline name",
        test_run_regression_evaluation_rejects_empty_baseline_name
    );

    test::expect_invalid_argument(
        "run_regression_evaluation rejects mismatched baseline predictions",
        test_run_regression_evaluation_rejects_mismatched_baseline_predictions
    );

    test::expect_invalid_argument(
        "run_regression_evaluation rejects mismatched model predictions",
        test_run_regression_evaluation_rejects_mismatched_model_predictions
    );
}

// ---- Experiment summary export tests ----

const std::string k_phase2_output_dir = "outputs/phase-2-evaluation-methodology";

std::string read_text_file(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file for reading: " + path);
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

ml::RegressionExperimentSummary make_regression_experiment_summary() {
    const ml::RegressionEvaluationReport report = ml::run_regression_evaluation(
        make_regression_evaluation_input()
    );

    return ml::RegressionExperimentSummary{
        "phase2_export_test",
        "synthetic_regression_dataset",
        "test_split",
        report
    };
}

void ensure_phase2_output_dir_exists() {
    std::filesystem::create_directories(k_phase2_output_dir);
}

void test_export_regression_summary_csv_creates_file() {
    ensure_phase2_output_dir_exists();

    const std::string output_path =
        k_phase2_output_dir + "/test_regression_summary.csv";

    const ml::RegressionExperimentSummary summary =
        make_regression_experiment_summary();

    ml::export_regression_summary_csv(summary, output_path);

    if (!std::filesystem::exists(output_path)) {
        throw std::runtime_error("expected CSV summary file to exist");
    }
}

void test_export_regression_summary_csv_writes_expected_content() {
    ensure_phase2_output_dir_exists();

    const std::string output_path =
        k_phase2_output_dir + "/test_regression_summary_content.csv";

    const ml::RegressionExperimentSummary summary =
        make_regression_experiment_summary();

    ml::export_regression_summary_csv(summary, output_path);

    const std::string content = read_text_file(output_path);

    if (content.find("experiment_name,dataset_name,split_name") == std::string::npos) {
        throw std::runtime_error("expected CSV header to contain metadata columns");
    }

    if (content.find("baseline_mse,model_mse") == std::string::npos) {
        throw std::runtime_error("expected CSV header to contain MSE columns");
    }

    if (content.find("beats_mse,beats_rmse,beats_mae,beats_r2") == std::string::npos) {
        throw std::runtime_error("expected CSV header to contain comparison columns");
    }

    if (content.find("phase2_export_test") == std::string::npos) {
        throw std::runtime_error("expected CSV content to contain experiment name");
    }

    if (content.find("synthetic_regression_dataset") == std::string::npos) {
        throw std::runtime_error("expected CSV content to contain dataset name");
    }

    if (content.find("TestModel") == std::string::npos) {
        throw std::runtime_error("expected CSV content to contain model name");
    }

    if (content.find("MeanRegressor") == std::string::npos) {
        throw std::runtime_error("expected CSV content to contain baseline name");
    }
}

void test_export_regression_summary_txt_creates_file() {
    ensure_phase2_output_dir_exists();

    const std::string output_path =
        k_phase2_output_dir + "/test_regression_summary.txt";

    const ml::RegressionExperimentSummary summary =
        make_regression_experiment_summary();

    ml::export_regression_summary_txt(summary, output_path);

    if (!std::filesystem::exists(output_path)) {
        throw std::runtime_error("expected TXT summary file to exist");
    }
}

void test_export_regression_summary_txt_writes_expected_content() {
    ensure_phase2_output_dir_exists();

    const std::string output_path =
        k_phase2_output_dir + "/test_regression_summary_content.txt";

    const ml::RegressionExperimentSummary summary =
        make_regression_experiment_summary();

    ml::export_regression_summary_txt(summary, output_path);

    const std::string content = read_text_file(output_path);

    if (content.find("Experiment: phase2_export_test") == std::string::npos) {
        throw std::runtime_error("expected TXT content to contain experiment name");
    }

    if (content.find("Dataset: synthetic_regression_dataset") == std::string::npos) {
        throw std::runtime_error("expected TXT content to contain dataset name");
    }

    if (content.find("Split: test_split") == std::string::npos) {
        throw std::runtime_error("expected TXT content to contain split name");
    }

    if (content.find("Baseline metrics:") == std::string::npos) {
        throw std::runtime_error("expected TXT content to contain baseline metrics section");
    }

    if (content.find("Model metrics:") == std::string::npos) {
        throw std::runtime_error("expected TXT content to contain model metrics section");
    }

    if (content.find("Comparison:") == std::string::npos) {
        throw std::runtime_error("expected TXT content to contain comparison section");
    }
}

void test_export_regression_summary_csv_rejects_empty_experiment_name() {
    ensure_phase2_output_dir_exists();

    ml::RegressionExperimentSummary summary = make_regression_experiment_summary();
    summary.experiment_name = "";

    const std::string output_path =
        k_phase2_output_dir + "/invalid_empty_experiment_name.csv";

    ml::export_regression_summary_csv(summary, output_path);
}

void test_export_regression_summary_txt_rejects_empty_output_path() {
    const ml::RegressionExperimentSummary summary = make_regression_experiment_summary();

    ml::export_regression_summary_txt(summary, "");
}

void run_experiment_summary_export_tests() {
    std::cout << "\n[Phase 2.7] Metrics and experiment summary export tests\n\n";

    test::expect_no_throw(
        "export_regression_summary_csv creates file",
        test_export_regression_summary_csv_creates_file
    );

    test::expect_no_throw(
        "export_regression_summary_csv writes expected content",
        test_export_regression_summary_csv_writes_expected_content
    );

    test::expect_no_throw(
        "export_regression_summary_txt creates file",
        test_export_regression_summary_txt_creates_file
    );

    test::expect_no_throw(
        "export_regression_summary_txt writes expected content",
        test_export_regression_summary_txt_writes_expected_content
    );

    test::expect_invalid_argument(
        "export_regression_summary_csv rejects empty experiment name",
        test_export_regression_summary_csv_rejects_empty_experiment_name
    );

    test::expect_invalid_argument(
        "export_regression_summary_txt rejects empty output path",
        test_export_regression_summary_txt_rejects_empty_output_path
    );
}

void run_dataset_tests() {
    std::cout << "\n[Phase 2.1] Supervised dataset abstraction tests\n\n";

    test::expect_no_throw(
        "SupervisedDataset accepts valid X/y",
        test_supervised_dataset_accepts_valid_xy
    );

    test::expect_invalid_argument(
        "SupervisedDataset rejects mismatched X/y",
        test_supervised_dataset_rejects_mismatched_xy
    );

    test::expect_invalid_argument(
        "SupervisedDataset rejects empty X/y",
        test_supervised_dataset_rejects_empty_matrix
    );

    test::expect_no_throw(
        "SupervisedDataset reports num_samples correctly",
        test_supervised_dataset_reports_num_samples
    );

    test::expect_no_throw(
        "SupervisedDataset reports num_features correctly",
        test_supervised_dataset_reports_num_features
    );
}

}  // namespace

namespace ml::experiments {

void run_phase2_evaluation_sanity() {
    run_dataset_tests();
    run_train_test_split_tests();
    run_train_validation_test_split_tests();
    run_k_fold_split_tests();
    run_preprocessing_pipeline_tests();
    run_baseline_evaluation_flow_tests();
    run_evaluation_harness_tests();
    run_experiment_summary_export_tests();
}

}  // namespace ml::experiments