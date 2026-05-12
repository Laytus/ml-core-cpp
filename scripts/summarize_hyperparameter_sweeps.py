#!/usr/bin/env python3
"""
Summarize Phase 11 hyperparameter sweep outputs.

Usage:
    python3 scripts/summarize_hyperparameter_sweeps.py

Purpose:
    - Read all practical workflow hyperparameter sweep CSV files.
    - Reconstruct parameter configurations per run.
    - Print best runs by relevant metrics.
    - Produce grounded summaries for docs/practical/sweeps/*.md.

This script does not train models.
It only analyzes already generated CSV outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import pandas as pd


OUTPUT_ROOT = Path("outputs/practical-exercises")


SWEEP_FILES: Dict[str, Path] = {
    "regression": OUTPUT_ROOT / "regression" / "hyperparameter_sweep.csv",
    "binary_classification": OUTPUT_ROOT / "binary-classification" / "hyperparameter_sweep.csv",
    "multiclass_classification": OUTPUT_ROOT / "multiclass-classification" / "hyperparameter_sweep.csv",
    "unsupervised": OUTPUT_ROOT / "unsupervised" / "hyperparameter_sweep.csv",
}


MetricDirection = Literal["min", "max"]


@dataclass(frozen=True)
class MetricTarget:
    metric: str
    direction: MetricDirection


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required sweep file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Required sweep file is empty: {path}")


def read_sweep(path: Path) -> pd.DataFrame:
    require_file(path)

    dataframe = pd.read_csv(path)

    required_columns = {
        "run_id",
        "workflow",
        "dataset",
        "model",
        "split",
        "param_name",
        "param_value",
        "metric",
        "value",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    if dataframe.empty:
        raise ValueError(f"Sweep file has no rows: {path}")

    dataframe["value"] = pd.to_numeric(dataframe["value"], errors="raise")

    return dataframe


def build_run_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long sweep rows into one row per run_id + metric.

    Input schema:
        run_id, workflow, dataset, model, split, param_name, param_value, metric, value

    Output schema:
        run_id, workflow, dataset, model, split, metric, value, params
    """
    group_columns = ["run_id", "workflow", "dataset", "model", "split", "metric"]

    rows: List[dict] = []

    for group_values, group in dataframe.groupby(group_columns, dropna=False):
        run_id, workflow, dataset, model, split, metric = group_values

        metric_values = group["value"].unique()

        if len(metric_values) != 1:
            raise ValueError(
                "Expected exactly one metric value per "
                f"run_id/model/metric group, got {len(metric_values)} for "
                f"run_id={run_id}, model={model}, metric={metric}"
            )

        params = (
            group[["param_name", "param_value"]]
            .drop_duplicates()
            .sort_values("param_name")
        )

        param_text = ", ".join(
            f"{row.param_name}={row.param_value}"
            for row in params.itertuples(index=False)
        )

        rows.append(
            {
                "run_id": run_id,
                "workflow": workflow,
                "dataset": dataset,
                "model": model,
                "split": split,
                "metric": metric,
                "value": float(metric_values[0]),
                "params": param_text,
            }
        )

    return pd.DataFrame(rows)


def best_rows(
    run_table: pd.DataFrame,
    metric: str,
    direction: MetricDirection,
    model: Optional[str] = None,
    top_n: int = 3,
) -> pd.DataFrame:
    subset = run_table[run_table["metric"] == metric].copy()

    if model is not None:
        subset = subset[subset["model"] == model].copy()

    if subset.empty:
        return subset

    ascending = direction == "min"

    return (
        subset
        .sort_values("value", ascending=ascending)
        .head(top_n)
        .reset_index(drop=True)
    )


def print_table(
    title: str,
    dataframe: pd.DataFrame,
    value_label: str,
) -> None:
    print(f"\n{title}")

    if dataframe.empty:
        print("  No rows found.")
        return

    for position, row in enumerate(dataframe.itertuples(index=False), start=1):
        print(
            "  "
            f"{position}. "
            f"run_id={row.run_id} | "
            f"model={row.model} | "
            f"{value_label}={row.value:.12g} | "
            f"params=({row.params})"
        )


def summarize_metric_targets(
    run_table: pd.DataFrame,
    section_title: str,
    targets: Iterable[MetricTarget],
    models: Optional[Iterable[str]] = None,
    top_n: int = 3,
) -> None:
    print(f"\n{'=' * 80}")
    print(section_title)
    print("=" * 80)

    if models is None:
        for target in targets:
            rows = best_rows(
                run_table,
                metric=target.metric,
                direction=target.direction,
                model=None,
                top_n=top_n,
            )
            print_table(
                title=f"Best overall by {target.metric} ({target.direction})",
                dataframe=rows,
                value_label=target.metric,
            )
        return

    for model in models:
        print(f"\n--- {model} ---")

        for target in targets:
            rows = best_rows(
                run_table,
                metric=target.metric,
                direction=target.direction,
                model=model,
                top_n=top_n,
            )
            print_table(
                title=f"Best by {target.metric} ({target.direction})",
                dataframe=rows,
                value_label=target.metric,
            )


def summarize_regression(run_table: pd.DataFrame) -> None:
    summarize_metric_targets(
        run_table=run_table,
        section_title="[Regression] GradientBoostingRegressor sweep",
        models=["GradientBoostingRegressor"],
        targets=[
            MetricTarget("mse", "min"),
            MetricTarget("rmse", "min"),
            MetricTarget("mae", "min"),
            MetricTarget("r2", "max"),
        ],
        top_n=3,
    )


def summarize_binary_classification(run_table: pd.DataFrame) -> None:
    summarize_metric_targets(
        run_table=run_table,
        section_title="[Binary classification] Hyperparameter sweeps",
        models=[
            "LogisticRegression",
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "TinyMLPBinaryClassifier",
        ],
        targets=[
            MetricTarget("f1", "max"),
            MetricTarget("recall", "max"),
            MetricTarget("precision", "max"),
            MetricTarget("accuracy", "max"),
        ],
        top_n=3,
    )


def summarize_multiclass_classification(run_table: pd.DataFrame) -> None:
    summarize_metric_targets(
        run_table=run_table,
        section_title="[Multiclass classification] KNNClassifier sweep",
        models=["KNNClassifier"],
        targets=[
            MetricTarget("macro_f1", "max"),
            MetricTarget("macro_recall", "max"),
            MetricTarget("macro_precision", "max"),
            MetricTarget("accuracy", "max"),
        ],
        top_n=3,
    )


def print_unsupervised_pca_summary(run_table: pd.DataFrame) -> None:
    pca_rows = run_table[
        (run_table["model"] == "PCA")
        & (run_table["metric"] == "cumulative_explained_variance_ratio")
    ].copy()

    print("\n--- PCA cumulative explained variance ---")

    if pca_rows.empty:
        print("  No PCA cumulative explained variance rows found.")
        return

    pca_rows = pca_rows.sort_values("value", ascending=False)

    for _, row in pca_rows.iterrows():
        print(
            "  "
            f"run_id={row['run_id']} | "
            f"cumulative_explained_variance_ratio={row['value']:.12g} | "
            f"params=({row['params']})"
        )


def print_unsupervised_kmeans_summary(run_table: pd.DataFrame) -> None:
    kmeans_rows = run_table[
        (run_table["model"] == "KMeans")
        & (run_table["metric"] == "inertia")
    ].copy()

    print("\n--- KMeans inertia by number of clusters ---")

    if kmeans_rows.empty:
        print("  No KMeans inertia rows found.")
        return

    kmeans_rows = kmeans_rows.sort_values("value", ascending=True)

    for _, row in kmeans_rows.iterrows():
        print(
            "  "
            f"run_id={row['run_id']} | "
            f"inertia={row['value']:.12g} | "
            f"params=({row['params']})"
        )


def summarize_unsupervised(run_table: pd.DataFrame) -> None:
    print(f"\n{'=' * 80}")
    print("[Unsupervised] PCA and KMeans sweeps")
    print("=" * 80)

    print_unsupervised_pca_summary(run_table)
    print_unsupervised_kmeans_summary(run_table)


def print_available_data_summary(all_frames: Dict[str, pd.DataFrame]) -> None:
    print("\n[Loaded sweep files]\n")

    for workflow, dataframe in all_frames.items():
        print(
            f"- {workflow}: "
            f"rows={len(dataframe)}, "
            f"models={sorted(dataframe['model'].dropna().unique())}"
        )


def main() -> None:
    print("\n[Phase 11] Hyperparameter sweep summary\n")

    raw_frames: Dict[str, pd.DataFrame] = {}
    run_tables: Dict[str, pd.DataFrame] = {}

    for workflow, path in SWEEP_FILES.items():
        raw = read_sweep(path)
        raw_frames[workflow] = raw
        run_tables[workflow] = build_run_table(raw)

    print_available_data_summary(raw_frames)

    summarize_regression(run_tables["regression"])
    summarize_binary_classification(run_tables["binary_classification"])
    summarize_multiclass_classification(run_tables["multiclass_classification"])
    summarize_unsupervised(run_tables["unsupervised"])

    print("\n[PASS] Hyperparameter sweep summaries generated")


if __name__ == "__main__":
    main()