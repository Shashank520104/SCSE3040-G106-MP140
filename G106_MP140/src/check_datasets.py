"""Compare train.csv against later.csv and holdout_features.csv.

Flags schema differences, categories seen in later/holdout but never in
train, and numeric columns whose distribution shifted a lot. Meant to run
before filling findings.yaml, as a sanity check on the seven data problems.

Usage:
    python src/check_datasets.py
    python src/check_datasets.py --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "resale_thousand_rs"
CATEGORICAL_COLUMNS = ["brand_tier", "city_zone"]
# Relative shift in mean/std above this fraction is reported as a "major shift".
SHIFT_THRESHOLD = 0.20


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    return pd.read_csv(path)


def compare_schema(train: pd.DataFrame, other: pd.DataFrame, other_name: str) -> None:
    print(f"\n--- Schema: train.csv vs {other_name} ---")

    train_cols, other_cols = set(train.columns), set(other.columns)
    only_in_train = train_cols - other_cols
    only_in_other = other_cols - train_cols

    if only_in_train:
        print(f"  Columns only in train.csv (missing from {other_name}): {sorted(only_in_train)}")
    if only_in_other:
        print(f"  Columns only in {other_name} (missing from train.csv): {sorted(only_in_other)}")
    if not only_in_train and not only_in_other:
        print("  Same columns in both files.")

    shared = sorted(train_cols & other_cols)
    dtype_mismatches = [
        col for col in shared if train[col].dtype != other[col].dtype
    ]
    if dtype_mismatches:
        print("  Dtype mismatches on shared columns:")
        for col in dtype_mismatches:
            print(f"    {col}: train={train[col].dtype}  {other_name}={other[col].dtype}")
    else:
        print("  No dtype mismatches on shared columns.")


def compare_categories(train: pd.DataFrame, other: pd.DataFrame, other_name: str) -> None:
    print(f"\n--- New categories: train.csv vs {other_name} ---")
    found_any = False
    for col in CATEGORICAL_COLUMNS:
        if col not in train.columns or col not in other.columns:
            continue
        train_values = set(train[col].dropna().unique())
        other_values = set(other[col].dropna().unique())
        new_values = other_values - train_values
        if new_values:
            found_any = True
            print(f"  {col}: new values in {other_name} not seen in train.csv -> {sorted(new_values)}")
    if not found_any:
        print("  No new category values found.")


def compare_distributions(train: pd.DataFrame, other: pd.DataFrame, other_name: str) -> None:
    print(f"\n--- Distribution shifts: train.csv vs {other_name} ---")
    numeric_cols = [
        col for col in train.select_dtypes(include=np.number).columns
        if col in other.columns and col != TARGET_COLUMN
    ]

    found_any = False
    for col in numeric_cols:
        train_mean, train_std = train[col].mean(), train[col].std()
        other_mean, other_std = other[col].mean(), other[col].std()

        mean_shift = abs(other_mean - train_mean) / train_mean if train_mean else float("inf")
        std_shift = abs(other_std - train_std) / train_std if train_std else float("inf")

        if mean_shift > SHIFT_THRESHOLD or std_shift > SHIFT_THRESHOLD:
            found_any = True
            print(
                f"  {col}: train mean={train_mean:.2f} std={train_std:.2f}  "
                f"{other_name} mean={other_mean:.2f} std={other_std:.2f}  "
                f"(mean shift {mean_shift:.0%}, std shift {std_shift:.0%})"
            )
    if not found_any:
        print("  No major shifts (>20%) in shared numeric columns.")


def run_comparison(train: pd.DataFrame, other: pd.DataFrame | None, other_name: str) -> None:
    if other is None:
        return
    compare_schema(train, other, other_name)
    compare_categories(train, other, other_name)
    compare_distributions(train, other, other_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=Path(__file__).resolve().parent.parent / "data",
        type=Path,
        help="Folder containing train.csv, later.csv, holdout_features.csv",
    )
    args = parser.parse_args()

    train = load_csv(args.data_dir / "train.csv")
    later = load_csv(args.data_dir / "later.csv")
    holdout = load_csv(args.data_dir / "holdout_features.csv")

    if train is None:
        print("\ntrain.csv is required to run any comparison. Nothing to do.")
        return

    run_comparison(train, later, "later.csv")
    run_comparison(train, holdout, "holdout_features.csv")

    print(
        "\nUse these results to fill findings.yaml: new categories feed "
        "'spelling_variants', columns missing from holdout_features.csv are "
        "candidates for 'leaking_column', and large distribution shifts are "
        "a clue for 'unit_mixup' or 'sentinel_value'."
    )


if __name__ == "__main__":
    main()
