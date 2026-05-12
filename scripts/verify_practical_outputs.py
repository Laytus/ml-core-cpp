#!/usr/bin/env python3
"""
Verify that Phase 11 practical workflow outputs are readable from Python/Pandas.

Usage:
    python3 scripts/verify_practical_outputs.py

Purpose:
    - Read generated practical output CSV files with pandas.
    - Validate that required files exist.
    - Validate that required columns are present.
    - Print a concise summary of row counts.

This script does not validate model quality.
It only validates output readability and schema compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


OUTPUT_ROOT = Path("outputs/practical-exercises")


EXPECTED_SCHEMAS: Dict[Path, List[str]] = {
    Path("regression/metrics.csv"): [
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "metric",
        "value",
    ],
    Path("regression/predictions.csv"): [
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "y_pred",
        "error",
    ],
    Path("regression/loss_history.csv"): [
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "iteration",
        "loss",
    ],
    Path("binary-classification/metrics.csv"): [
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "metric",
        "value",
    ],
    Path("binary-classification/predictions.csv"): [
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "y_pred",
        "correct",
    ],
    Path("binary-classification/probabilities.csv"): [
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "probability_class_0",
        "probability_class_1",
    ],
    Path("binary-classification/decision_scores.csv"): [
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "decision_score",
    ],
    Path("binary-classification/loss_history.csv"): [
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "iteration",
        "loss",
    ],
    Path("multiclass-classification/metrics.csv"): [
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "metric",
        "value",
    ],
    Path("multiclass-classification/predictions.csv"): [
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "y_pred",
        "correct",
    ],
    Path("multiclass-classification/probabilities.csv"): [
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "y_true",
        "probability_class_0",
        "probability_class_1",
        "probability_class_2",
    ],
    Path("multiclass-classification/loss_history.csv"): [
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "iteration",
        "loss",
    ],
    Path("unsupervised/metrics.csv"): [
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "metric",
        "value",
    ],
    Path("unsupervised/projections.csv"): [
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "method",
        "split",
        "component_1",
        "component_2",
        "label_reference",
    ],
    Path("unsupervised/clustering_assignments.csv"): [
        "run_id",
        "row_id",
        "workflow",
        "dataset",
        "method",
        "split",
        "cluster",
        "label_reference",
    ],
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required output file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Required output file is empty: {path}")


def require_columns(
    dataframe: pd.DataFrame,
    expected_columns: List[str],
    path: Path,
) -> None:
    missing_columns = [
        column for column in expected_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {missing_columns}. "
            f"Available columns: {list(dataframe.columns)}"
        )


def verify_csv(relative_path: Path, expected_columns: List[str]) -> pd.DataFrame:
    path = OUTPUT_ROOT / relative_path

    require_file(path)

    dataframe = pd.read_csv(path)
    require_columns(dataframe, expected_columns, path)

    if dataframe.empty:
        raise ValueError(f"Output CSV has header but no rows: {path}")

    print(f"[OK] {path} | rows={len(dataframe)} | columns={len(dataframe.columns)}")

    return dataframe


def verify_metric_files(metric_frames: List[pd.DataFrame]) -> None:
    combined = pd.concat(metric_frames, ignore_index=True)

    required_workflows = {
        "regression",
        "binary_classification",
        "multiclass_classification",
        "unsupervised",
    }

    observed_workflows = set(combined["workflow"].dropna().astype(str).unique())

    missing_workflows = required_workflows - observed_workflows

    if missing_workflows:
        raise ValueError(
            f"Missing workflows in metrics outputs: {sorted(missing_workflows)}"
        )

    if not pd.api.types.is_numeric_dtype(combined["value"]):
        raise ValueError("metrics.csv value column must be numeric")

    print("[OK] Combined metrics are readable and include all workflow families")

    summary = (
        combined
        .groupby(["workflow", "dataset", "model"], dropna=False)
        .size()
        .reset_index(name="metric_rows")
    )

    print("\nMetric row summary:")
    print(summary.to_string(index=False))


def main() -> None:
    print("\n[Phase 11] Verifying practical workflow outputs with Pandas\n")

    metric_frames: List[pd.DataFrame] = []

    for relative_path, expected_columns in EXPECTED_SCHEMAS.items():
        dataframe = verify_csv(relative_path, expected_columns)

        if relative_path.name == "metrics.csv":
            metric_frames.append(dataframe)

    verify_metric_files(metric_frames)

    print("\n[PASS] Practical outputs are readable from Python/Pandas")


if __name__ == "__main__":
    main()