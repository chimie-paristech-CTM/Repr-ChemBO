"""Merge the contexts of completed runs into a single CSV file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.context_manager import ContextManager


class Merger:
    """Collect every run's info into one tabular summary."""

    OUTPUT_FILENAME = "merged_contexts.csv"
    ANALYZE_FILENAME = "__analyzed.csv"

    def __init__(self) -> None:
        self.output_root = Path(ContextManager.OUTPUT_ROOT)
        self.failure_root = Path(ContextManager.FAILED_RUNS_ROOT)

    def merge(self) -> Path:
        """Write all run contexts to ``output/merged_contexts.csv``.
        
        NOTE all historical runs will be included!

        Nested JSON objects are flattened using dot-separated column names.
        Runs and columns are sorted to keep the generated file deterministic.
        Missing fields in an individual context are written as empty cells.
        """
        contexts = [self._load_context(path) for path in self._context_paths()]

        if contexts:
            dataframe = pd.json_normalize(contexts, sep=".")
            dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)
        else:
            dataframe = pd.DataFrame()

        self.output_root.mkdir(parents=True, exist_ok=True)
        output_path = self.output_root / self.OUTPUT_FILENAME
        dataframe.to_csv(output_path, index=False, encoding="utf-8")
        return output_path

    def analyze(self, parameters: str | list[str]) -> pd.DataFrame:
        """Calculate mean and standard deviation scores by parameter values.

        Args:
            parameters: A column name or a list of column names from the merged
                CSV, for example ``["fp.name", "dataset.name", "hyperprior"]``.

        Returns:
            One row per parameter combination, containing ``run_count``, the
            mean and standard deviation of AUC, plus mean and standard deviation
            top-coverage columns for every percentage in
            ``scoring.top_perc_cov``.

        Raises:
            FileNotFoundError: If :meth:`merge` has not produced a CSV yet.
            ValueError: If no grouping parameter is supplied, score columns
                are missing, or a top-coverage value is malformed.
            KeyError: If a requested parameter is not present in the CSV.
        """
        group_columns = [parameters] if isinstance(parameters, str) else list(parameters)
        if not group_columns:
            raise ValueError("At least one analysis parameter is required")
        if len(group_columns) != len(set(group_columns)):
            raise ValueError("Analysis parameters must not contain duplicates")

        merged_path = self.output_root / self.OUTPUT_FILENAME
        if not merged_path.is_file():
            raise FileNotFoundError(
                f"Merged context CSV not found: {merged_path}. Run merge() first."
            )

        dataframe = pd.read_csv(merged_path)
        missing_parameters = [
            column for column in group_columns if column not in dataframe.columns
        ]
        if missing_parameters:
            raise KeyError(f"Unknown analysis parameters: {missing_parameters}")
        if "AUC" not in dataframe.columns or "top_coverage" not in dataframe.columns:
            raise ValueError("Merged contexts must contain AUC and top_coverage")

        dataframe = dataframe.copy()
        dataframe["AUC"] = pd.to_numeric(dataframe["AUC"], errors="coerce")
        coverage_columns = self._expand_top_coverage(dataframe)
        score_columns = ["AUC", *coverage_columns]
        score_aggregations = {
            f"{column}_{statistic}": (column, statistic)
            for column in score_columns
            for statistic in ("mean", "std")
        }

        # The core logic is groupby: aggregate over all other factors
        analysis = (
            dataframe.groupby(group_columns, dropna=False, sort=True)
            .agg(
                run_count=("AUC", "size"),
                **score_aggregations,
            )
            .reset_index()
        )
        
        analysis.to_csv(self.output_root / self.ANALYZE_FILENAME)
        
        return analysis

    def _expand_top_coverage(self, dataframe: pd.DataFrame) -> list[str]:
        """Expand list-valued coverage scores into percentage-specific columns."""
        percentage_column = "scoring.top_perc_cov"
        has_percentages = percentage_column in dataframe.columns
        expanded_rows: list[dict[str, float]] = []

        for index, row in dataframe.iterrows():
            coverages = self._parse_number_list(row["top_coverage"], "top_coverage", index)
            if has_percentages:
                percentages = self._parse_number_list(
                    row[percentage_column], percentage_column, index
                )
                if len(coverages) != len(percentages):
                    raise ValueError(
                        f"Row {index}: top_coverage and {percentage_column} "
                        "must have the same length"
                    )
                labels = [self._format_percentage(value) for value in percentages]
            else:
                labels = [str(position) for position in range(len(coverages))]

            expanded_rows.append(
                {
                    f"top_coverage_{label}": coverage
                    for label, coverage in zip(labels, coverages)
                }
            )

        expanded = pd.DataFrame(expanded_rows, index=dataframe.index, dtype=float)
        for column in expanded.columns:
            dataframe[column] = expanded[column]
        return list(expanded.columns)

    @staticmethod
    def _parse_number_list(value: Any, column: str, row: Any) -> list[float]:
        """Parse a JSON list stored in a CSV cell and validate its values."""
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            if not isinstance(parsed, list):
                raise TypeError
            return [float(item) for item in parsed]
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Row {row}: {column} must be a list of numbers, got {value!r}"
            ) from error

    @staticmethod
    def _format_percentage(value: float) -> str:
        """Format a proportion such as 0.05 as a compact ``5%`` label."""
        return f"{value * 100:g}%"

    def _context_paths(self) -> list[Path]:
        """Return context files from immediate run-id directories."""
        if not self.output_root.is_dir():
            return []

        return [
            context_path
            for run_dir in sorted(self.output_root.iterdir(), key=lambda p: p.name)
            if run_dir.is_dir()
            and (context_path := run_dir / "context.json").is_file()
        ]

    @staticmethod
    def _load_context(path: Path) -> dict[str, Any]:
        """Load one context and ensure it can represent one CSV row."""
        with path.open("r", encoding="utf-8") as context_file:
            context = json.load(context_file)

        if not isinstance(context, dict):
            raise ValueError(f"Context must be a JSON object: {path}")
        return context
