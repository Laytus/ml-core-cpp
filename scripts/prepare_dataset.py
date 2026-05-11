#!/usr/bin/env python3
"""
Generic dataset preparation utility for Phase 11.

Usage:
    python3 scripts/prepare_dataset.py data/metadata/processing_specs/wine.json
    python3 scripts/prepare_dataset.py data/metadata/processing_specs/nasa_kc1_software_defects.json
    python3 scripts/prepare_dataset.py data/metadata/processing_specs/stock_ohlcv_engineered.json

Design rule:
    This script is spec-driven.
    It must not branch on dataset names such as "wine", "kc1", or "stock".

Allowed branches:
    - input format branches: csv, delimited, arff, csv_collection
    - reusable processor branches: ohlcv_feature_engineering
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


Row = Dict[str, str]
OutputRow = Dict[str, Any]
Spec = Dict[str, Any]


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def load_json(path: Path) -> Spec:
    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON spec must be an object: {path}")

    return data


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def require_keys(obj: Dict[str, Any], keys: Sequence[str], context: str) -> None:
    for key in keys:
        if key not in obj:
            raise ValueError(f"Missing required key '{key}' in {context}")


def normalize_token(value: str) -> str:
    return value.strip().strip("'").strip('"')


def is_missing(value: str, missing_values: Sequence[str]) -> bool:
    return normalize_token(value) in set(missing_values)


def parse_float(value: str, column_name: str) -> float:
    try:
        result = float(normalize_token(value))
    except ValueError as exc:
        raise ValueError(
            f"Could not parse numeric value in column '{column_name}': {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value in column '{column_name}': {value!r}")

    return result


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[OutputRow]) -> None:
    ensure_parent_dir(path)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def as_string_list(value: Any, context: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")

    result: List[str] = []

    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{context} must contain only strings")
        result.append(item)

    return result


# ---------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------


def read_delimited(input_spec: Dict[str, Any]) -> List[Row]:
    require_keys(input_spec, ["path", "delimiter", "has_header"], "delimited input spec")

    path = Path(str(input_spec["path"]))
    delimiter = str(input_spec["delimiter"])
    has_header = bool(input_spec["has_header"])

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        if has_header:
            reader = csv.DictReader(f, delimiter=delimiter)

            if reader.fieldnames is None:
                raise ValueError(f"Could not read header from {path}")

            rows: List[Row] = []

            for row in reader:
                cleaned_row: Row = {}

                for key, value in row.items():
                    if key is None:
                        raise ValueError(f"Invalid unnamed column in {path}")
                    cleaned_row[key] = "" if value is None else value.strip()

                rows.append(cleaned_row)

            return rows

        if "columns" not in input_spec:
            raise ValueError("Headerless delimited input requires 'columns' in the processing spec")

        columns = as_string_list(input_spec["columns"], "input.columns")
        reader = csv.reader(f, delimiter=delimiter)

        rows = []

        for line_number, values in enumerate(reader, start=1):
            if not values:
                continue

            if len(values) != len(columns):
                raise ValueError(
                    f"Wrong number of columns in {path} at line {line_number}: "
                    f"expected {len(columns)}, got {len(values)}"
                )

            rows.append({column: value.strip() for column, value in zip(columns, values)})

        return rows


def read_csv(input_spec: Dict[str, Any]) -> List[Row]:
    normalized_spec = dict(input_spec)
    normalized_spec.setdefault("delimiter", ",")
    return read_delimited(normalized_spec)


def parse_arff_attribute(line: str) -> str:
    stripped = line.strip()

    if not stripped.lower().startswith("@attribute"):
        raise ValueError(f"Invalid ARFF attribute line: {line!r}")

    rest = stripped[len("@attribute") :].strip()

    if not rest:
        raise ValueError(f"Invalid ARFF attribute line: {line!r}")

    if rest[0] in ("'", '"'):
        quote = rest[0]
        end = rest.find(quote, 1)

        if end == -1:
            raise ValueError(f"Unclosed quoted ARFF attribute name: {line!r}")

        return rest[1:end]

    parts = rest.split(maxsplit=1)

    if not parts:
        raise ValueError(f"Invalid ARFF attribute line: {line!r}")

    return parts[0]


def parse_arff_data_line(line: str) -> List[str]:
    """
    Basic dense ARFF data parser.

    Sparse ARFF rows such as "{0 1.2, 3 true}" are intentionally unsupported.
    """
    if line.lstrip().startswith("{"):
        raise ValueError("Sparse ARFF rows are not supported by this simple parser")

    reader = csv.reader([line], delimiter=",", quotechar="'", skipinitialspace=True)
    return [normalize_token(value) for value in next(reader)]


def read_arff(input_spec: Dict[str, Any]) -> List[Row]:
    require_keys(input_spec, ["path"], "ARFF input spec")

    path = Path(str(input_spec["path"]))

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    attributes: List[str] = []
    rows: List[Row] = []
    in_data = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("%"):
                continue

            lower = line.lower()

            if lower.startswith("@relation"):
                continue

            if lower.startswith("@attribute"):
                if in_data:
                    raise ValueError(f"Found @attribute after @data at line {line_number}")

                attributes.append(parse_arff_attribute(line))
                continue

            if lower.startswith("@data"):
                if not attributes:
                    raise ValueError("ARFF file reached @data before any @attribute definitions")

                in_data = True
                continue

            if not in_data:
                continue

            values = parse_arff_data_line(line)

            if len(values) != len(attributes):
                raise ValueError(
                    f"Wrong number of ARFF values at line {line_number}: "
                    f"expected {len(attributes)}, got {len(values)}"
                )

            rows.append({attribute: value for attribute, value in zip(attributes, values)})

    if not rows:
        raise ValueError(f"No data rows found in ARFF file: {path}")

    return rows


def read_csv_collection(input_spec: Dict[str, Any]) -> List[Row]:
    require_keys(input_spec, ["paths", "has_header"], "csv_collection input spec")

    raw_paths = input_spec["paths"]

    if not isinstance(raw_paths, list):
        raise ValueError("input.paths must be a list")

    paths = [Path(str(path)) for path in raw_paths]
    delimiter = str(input_spec.get("delimiter", ","))
    has_header = bool(input_spec["has_header"])

    all_rows: List[Row] = []

    for path in paths:
        file_spec: Dict[str, Any] = {
            "format": "csv",
            "path": str(path),
            "delimiter": delimiter,
            "has_header": has_header,
        }

        if "columns" in input_spec:
            file_spec["columns"] = input_spec["columns"]

        rows = read_csv(file_spec)
        source_name = path.stem

        for row in rows:
            row["_source_file"] = source_name

        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("CSV collection produced no rows")

    return all_rows


def read_input(spec: Spec) -> List[Row]:
    input_spec_raw = spec.get("input")

    if not isinstance(input_spec_raw, dict):
        raise ValueError("spec.input must be an object")

    input_spec: Dict[str, Any] = input_spec_raw
    input_format = str(input_spec.get("format"))

    if input_format == "csv":
        return read_csv(input_spec)

    if input_format == "delimited":
        return read_delimited(input_spec)

    if input_format == "arff":
        return read_arff(input_spec)

    if input_format == "csv_collection":
        return read_csv_collection(input_spec)

    raise ValueError(f"Unsupported input format: {input_format!r}")


# ---------------------------------------------------------------------
# Generic tabular processing
# ---------------------------------------------------------------------


def apply_column_renames(rows: List[Row], renames: Dict[str, Any]) -> List[Row]:
    if not renames:
        return rows

    clean_renames: Dict[str, str] = {}

    for old_name, new_name in renames.items():
        clean_renames[str(old_name)] = str(new_name)

    renamed_rows: List[Row] = []

    for row in rows:
        new_row: Row = {}

        for key, value in row.items():
            new_key = clean_renames.get(key, key)

            if new_key in new_row:
                raise ValueError(f"Column rename collision: {key!r} -> {new_key!r}")

            new_row[new_key] = value

        renamed_rows.append(new_row)

    return renamed_rows


def encode_target_value(value: str, target_spec: Dict[str, Any]) -> str:
    encoding_raw = target_spec.get("encoding")

    if encoding_raw is None:
        return value

    if not isinstance(encoding_raw, dict):
        raise ValueError("target.encoding must be an object when provided")

    encoding: Dict[str, Any] = {str(key): val for key, val in encoding_raw.items()}
    normalized = normalize_token(value)

    if normalized not in encoding:
        raise ValueError(
            f"Target value {value!r} is not covered by encoding map. "
            f"Allowed values: {sorted(encoding.keys())}"
        )

    return str(encoding[normalized])


def validate_required_columns(rows: Sequence[Row], required_columns: Sequence[str]) -> None:
    if not rows:
        raise ValueError("No rows available for validation")

    available = set(rows[0].keys())

    for column in required_columns:
        if column not in available:
            raise ValueError(
                f"Required column not found: {column!r}. "
                f"Available columns: {sorted(available)}"
            )


def get_features(spec: Spec) -> List[str]:
    if "features" not in spec:
        raise ValueError("Processing spec requires 'features'")

    return as_string_list(spec["features"], "features")


def get_target_spec(spec: Spec) -> Optional[Dict[str, Any]]:
    raw_target = spec.get("target")

    if raw_target is None:
        return None

    if not isinstance(raw_target, dict):
        raise ValueError("target must be an object when provided")

    if "column" not in raw_target:
        raise ValueError("target.column is required when target is provided")

    if not isinstance(raw_target["column"], str):
        raise ValueError("target.column must be a string")

    return raw_target


def get_validation_spec(spec: Spec) -> Dict[str, Any]:
    raw_validation = spec.get("validation", {})

    if not isinstance(raw_validation, dict):
        raise ValueError("validation must be an object when provided")

    return raw_validation


def get_missing_values(validation: Dict[str, Any]) -> List[str]:
    raw_missing_values = validation.get("missing_values", ["", "?", "NA", "NaN", "nan", "null"])

    if not isinstance(raw_missing_values, list):
        raise ValueError("validation.missing_values must be a list")

    return [str(value) for value in raw_missing_values]


def build_numeric_supervised_rows(spec: Spec, rows: List[Row]) -> List[OutputRow]:
    features = get_features(spec)
    target_spec = get_target_spec(spec)
    validation = get_validation_spec(spec)
    missing_values = get_missing_values(validation)

    required_columns = list(features)

    target_column: Optional[str] = None

    if target_spec is not None:
        target_column = str(target_spec["column"])
        required_columns.append(target_column)

    validate_required_columns(rows, required_columns)

    processed_rows: List[OutputRow] = []

    for row in rows:
        output_row: OutputRow = {}
        reject_row = False

        for feature in features:
            value = row[feature]

            if validation.get("reject_missing", True) and is_missing(value, missing_values):
                reject_row = True
                break

            output_row[feature] = parse_float(value, feature)

        if reject_row:
            continue

        if target_spec is not None:
            # This assignment is intentionally inside the target_spec block.
            # It gives Pylance a concrete str instead of Optional[str].
            concrete_target_column = str(target_spec["column"])
            raw_target = row[concrete_target_column]

            if validation.get("reject_missing", True) and is_missing(raw_target, missing_values):
                continue

            encoded_target = encode_target_value(raw_target, target_spec)
            output_row[concrete_target_column] = parse_float(encoded_target, concrete_target_column)

        processed_rows.append(output_row)

    if not processed_rows:
        raise ValueError("No rows remained after processing and validation")

    return processed_rows


# ---------------------------------------------------------------------
# OHLCV feature engineering
# ---------------------------------------------------------------------


def extract_ticker_from_source(source_file: str) -> str:
    """
    Converts names like:
        aapl_us_d -> AAPL
        msft_us_d -> MSFT
    """
    name = source_file.lower()
    name = re.sub(r"_us_d$", "", name)
    name = re.sub(r"_daily$", "", name)
    return name.upper()


def group_rows_by_source(rows: Sequence[Row]) -> Dict[str, List[Row]]:
    groups: Dict[str, List[Row]] = {}

    for row in rows:
        source = row.get("_source_file")

        if source is None:
            raise ValueError("OHLCV csv_collection rows require internal '_source_file' field")

        groups.setdefault(source, []).append(row)

    return groups


def compute_daily_return(current_close: float, previous_close: float) -> float:
    if previous_close <= 0.0:
        raise ValueError("Previous close must be positive to compute return")

    return (current_close - previous_close) / previous_close


def get_processor_spec(spec: Spec) -> Optional[Dict[str, Any]]:
    raw_processor = spec.get("processor")

    if raw_processor is None:
        return None

    if not isinstance(raw_processor, dict):
        raise ValueError("processor must be an object when provided")

    return raw_processor


def process_ohlcv_feature_engineering(
    spec: Spec,
    rows: List[Row],
) -> Tuple[List[OutputRow], List[OutputRow]]:
    processor = get_processor_spec(spec)

    if processor is None:
        raise ValueError("OHLCV processing requires a processor spec")

    validation = get_validation_spec(spec)

    date_col = str(processor["date_column"])
    open_col = str(processor["open_column"])
    high_col = str(processor["high_column"])
    low_col = str(processor["low_column"])
    close_col = str(processor["close_column"])
    volume_col = str(processor["volume_column"])

    lookback_return = int(processor.get("lookback_return", 5))
    lookback_volatility = int(processor.get("lookback_volatility", 5))
    target_horizon = int(processor.get("target_horizon", 1))

    if lookback_return <= 0:
        raise ValueError("lookback_return must be positive")

    if lookback_volatility <= 0:
        raise ValueError("lookback_volatility must be positive")

    if target_horizon <= 0:
        raise ValueError("target_horizon must be positive")

    required_columns = [
        date_col,
        open_col,
        high_col,
        low_col,
        close_col,
        volume_col,
        "_source_file",
    ]

    validate_required_columns(rows, required_columns)

    model_rows: List[OutputRow] = []
    reference_rows: List[OutputRow] = []

    groups = group_rows_by_source(rows)

    for source_file, group in groups.items():
        parsed: List[OutputRow] = []

        for row in group:
            parsed.append(
                {
                    "source_file": source_file,
                    "ticker": extract_ticker_from_source(source_file)
                    if bool(processor.get("ticker_from_filename", True))
                    else source_file,
                    "date": row[date_col],
                    "open": parse_float(row[open_col], open_col),
                    "high": parse_float(row[high_col], high_col),
                    "low": parse_float(row[low_col], low_col),
                    "close": parse_float(row[close_col], close_col),
                    "volume": parse_float(row[volume_col], volume_col),
                }
            )

        parsed.sort(key=lambda item: str(item["date"]))

        closes = [float(item["close"]) for item in parsed]
        volumes = [float(item["volume"]) for item in parsed]

        daily_returns: List[Optional[float]] = [None]

        for i in range(1, len(parsed)):
            if closes[i - 1] <= 0.0:
                daily_returns.append(None)
            else:
                daily_returns.append(compute_daily_return(closes[i], closes[i - 1]))

        min_index = max(lookback_return, lookback_volatility)
        max_index_exclusive = len(parsed) - target_horizon

        for i in range(min_index, max_index_exclusive):
            current = parsed[i]

            close_t = closes[i]
            close_t_minus_1 = closes[i - 1]
            close_t_minus_return = closes[i - lookback_return]
            volume_t = volumes[i]
            volume_t_minus_1 = volumes[i - 1]
            future_close = closes[i + target_horizon]

            if validation.get("reject_non_positive_close", True):
                if close_t <= 0.0 or close_t_minus_1 <= 0.0 or close_t_minus_return <= 0.0:
                    continue

            if validation.get("reject_non_positive_previous_volume", True):
                if volume_t_minus_1 <= 0.0:
                    continue

            volatility_window = daily_returns[i - lookback_volatility + 1 : i + 1]

            if any(value is None for value in volatility_window):
                continue

            volatility_values = [float(value) for value in volatility_window if value is not None]

            if len(volatility_values) != lookback_volatility:
                continue

            return_1d = compute_daily_return(close_t, close_t_minus_1)
            return_5d = compute_daily_return(close_t, close_t_minus_return)
            volatility_5d = statistics.pstdev(volatility_values)
            range_pct = (float(current["high"]) - float(current["low"])) / close_t
            volume_change_1d = (volume_t - volume_t_minus_1) / volume_t_minus_1
            target_next_return = (future_close - close_t) / close_t

            engineered: OutputRow = {
                "return_1d": return_1d,
                "return_5d": return_5d,
                "volatility_5d": volatility_5d,
                "range_pct": range_pct,
                "volume_change_1d": volume_change_1d,
                "target_next_return": target_next_return,
            }

            numeric_values = [float(value) for value in engineered.values()]

            if not all(math.isfinite(value) for value in numeric_values):
                continue

            model_rows.append(engineered)

            reference_rows.append(
                {
                    "ticker": str(current["ticker"]),
                    "date": str(current["date"]),
                    **engineered,
                }
            )

    if not model_rows:
        raise ValueError("OHLCV feature engineering produced no rows")

    return model_rows, reference_rows


def apply_processor_if_needed(
    spec: Spec,
    rows: List[Row],
) -> Tuple[Optional[List[OutputRow]], Optional[List[OutputRow]]]:
    processor = get_processor_spec(spec)

    if processor is None:
        return None, None

    processor_type = str(processor.get("type"))

    if processor_type == "ohlcv_feature_engineering":
        return process_ohlcv_feature_engineering(spec, rows)

    raise ValueError(f"Unsupported processor type: {processor_type!r}")


# ---------------------------------------------------------------------
# Main preparation flow
# ---------------------------------------------------------------------


def get_output_spec(spec: Spec) -> Dict[str, Any]:
    raw_output = spec.get("output")

    if not isinstance(raw_output, dict):
        raise ValueError("output must be an object")

    if "path" not in raw_output:
        raise ValueError("output.path is required")

    return raw_output


def get_output_columns(spec: Spec) -> List[str]:
    features = get_features(spec)
    target_spec = get_target_spec(spec)

    columns = list(features)

    if target_spec is not None:
        columns.append(str(target_spec["column"]))

    return columns


def prepare_dataset(spec_path: Path) -> None:
    spec = load_json(spec_path)

    require_keys(spec, ["dataset_name", "input", "output", "features"], "processing spec")

    rows = read_input(spec)

    raw_renames = spec.get("column_renames", {})

    if not isinstance(raw_renames, dict):
        raise ValueError("column_renames must be an object when provided")

    rows = apply_column_renames(rows, raw_renames)

    processor_rows, reference_rows = apply_processor_if_needed(spec, rows)

    if processor_rows is not None:
        processed_rows = processor_rows
    else:
        processed_rows = build_numeric_supervised_rows(spec, rows)

    output_spec = get_output_spec(spec)
    output_path = Path(str(output_spec["path"]))
    output_columns = get_output_columns(spec)

    write_csv(output_path, output_columns, processed_rows)

    print(f"Prepared dataset: {spec['dataset_name']}")
    print(f"Rows written: {len(processed_rows)}")
    print(f"Output: {output_path}")

    reference_path_raw = output_spec.get("reference_path")

    if reference_path_raw is not None and reference_rows is not None:
        reference_path = Path(str(reference_path_raw))

        if not reference_rows:
            raise ValueError("reference_path was provided but no reference rows were produced")

        reference_columns = list(reference_rows[0].keys())
        write_csv(reference_path, reference_columns, reference_rows)

        print(f"Reference rows written: {len(reference_rows)}")
        print(f"Reference output: {reference_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a dataset from a JSON processing spec."
    )
    parser.add_argument(
        "spec_path",
        type=Path,
        help="Path to dataset processing spec JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(args.spec_path)


if __name__ == "__main__":
    main()