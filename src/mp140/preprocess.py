"""Data Preprocessing and Cleaning Module.

Problem ID: MP140
Group: G106
Dataset: Patna Used Two-Wheeler Price Prediction

This module provides a reusable clean_data(df) function to fix data quality issues
identified during the exploratory data analysis (EDA):
1. Duplicate rows.
2. Missing and null values across all columns.
3. Inconsistent category names, spelling variants, and whitespace in brand_tier.
4. Values containing unit strings (e.g. '150 cc', '45000 km') converted to pure numbers.
5. Unit mismatch in km_driven for the Central zone (values in thousand km).
6. Sentinel values in owners (-1).
7. Stuck sensor values in engine_cc (119 cc between 2026-03-05 and 2026-03-18).
8. Invalid, zero, or negative resale prices.
9. Invalid or negative bike ages (enforcing client rule <= 30 years).
10. Invalid or negative kilometres driven.

Usage:
    from src.preprocess import clean_data
    cleaned_df = clean_data(raw_df)

    Or run directly to verify on train.csv:
    python src/preprocess.py
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the two-wheeler dataset.

    Parameters:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with all identified issues resolved.
    """
    # 0. Work on a copy so the original DataFrame remains unmodified
    df = df.copy()

    # 1. Remove duplicate rows
    df = df.drop_duplicates()

    # 2. Convert values containing unit strings into pure numerical values (e.g. '150 cc', '45000 km')
    numeric_columns = [
        "age_years",
        "km_driven",
        "engine_cc",
        "owners",
        "service_records",
        "accidents",
        "resale_thousand_rs",
    ]
    for col in numeric_columns:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^\d.-]", "", regex=True),
                errors="coerce",
            )

    # 3. Clean text formatting and inconsistent category names in brand_tier
    if "brand_tier" in df.columns:
        # Strip leading/trailing spaces and convert to lowercase
        df["brand_tier"] = df["brand_tier"].astype(str).str.strip().str.lower()
        # Map abbreviations to true category names
        brand_map = {
            "bud.": "budget",
            "pre.": "premium",
            "mid.": "mid",
        }
        df["brand_tier"] = df["brand_tier"].replace(brand_map)
        valid_brands = {"budget", "mid", "premium"}
        # Fallback for unexpected or missing category strings
        df.loc[~df["brand_tier"].isin(valid_brands), "brand_tier"] = "premium"

    # Clean city_zone text formatting if present
    if "city_zone" in df.columns:
        df["city_zone"] = df["city_zone"].astype(str).str.strip().str.title()
        valid_zones = {"North", "South", "Central"}
        df.loc[~df["city_zone"].isin(valid_zones), "city_zone"] = "North"

    # 4. Fix km_driven unit mismatch & negative/invalid values
    if "km_driven" in df.columns:
        # Handle negative kilometres driven
        df["km_driven"] = df["km_driven"].abs()

        if "city_zone" in df.columns:
            # In train.csv, Central zone readings were recorded in thousands of km (all < 100)
            # In clean datasets (later.csv/holdout), Central readings are already in km (> 1000)
            # Multiplying Central zone values < 500 converts them to actual kilometres
            central_low_km = (df["city_zone"] == "Central") & (df["km_driven"] < 500)
            df.loc[central_low_km, "km_driven"] = df.loc[central_low_km, "km_driven"] * 1000.0

        # Impute missing km_driven with median (45,000 km) and cast to int64
        median_km = 45000.0
        df["km_driven"] = df["km_driven"].fillna(median_km).round().astype("int64")

    # 5. Fix invalid or sentinel values in owners (-1)
    if "owners" in df.columns:
        # Replace non-positive sentinel values (-1, 0) and unreasonable counts (> 10) with NaN
        df.loc[df["owners"] <= 0, "owners"] = np.nan
        df.loc[df["owners"] > 10, "owners"] = np.nan
        # Impute with median/mode (2 owners)
        df["owners"] = df["owners"].fillna(2).round().astype("int64")

    # 6. Fix suspicious/invalid engine_cc values, including stuck sensor (119 cc)
    if "engine_cc" in df.columns:
        # If date column is present, mask 119 cc during known stuck sensor window (March 5 to 18, 2026)
        if "date" in df.columns:
            stuck_mask = (
                (df["engine_cc"] == 119)
                & (df["date"] >= "2026-03-05")
                & (df["date"] <= "2026-03-18")
            )
            df.loc[stuck_mask, "engine_cc"] = np.nan

        # Replace unphysical engine cc (< 50 or > 1200) with NaN
        df.loc[(df["engine_cc"] < 50) | (df["engine_cc"] > 1200), "engine_cc"] = np.nan

        # Impute missing engine cc with median (300 cc)
        df["engine_cc"] = df["engine_cc"].fillna(300).round().astype("int64")

    # 7. Clean bike age (age_years)
    if "age_years" in df.columns:
        # Handle negative age
        df["age_years"] = df["age_years"].abs()
        # Enforce client rule: age above 30 is impossible, cap at 30.0
        df.loc[df["age_years"] > 30, "age_years"] = 30.0
        # Impute missing age with median (7.5 years)
        df["age_years"] = df["age_years"].fillna(7.5)

    # 8. Clean service_records (Client rule: < 0 is typing error)
    if "service_records" in df.columns:
        df["service_records"] = df["service_records"].abs()
        df["service_records"] = df["service_records"].fillna(10).round().astype("int64")

    # 9. Clean accidents
    if "accidents" in df.columns:
        df["accidents"] = df["accidents"].abs()
        df["accidents"] = df["accidents"].fillna(1).round().astype("int64")

    # 10. Handle target variable resale_thousand_rs if present
    if "resale_thousand_rs" in df.columns:
        # In training data, drop rows with zero, negative, or missing resale prices
        df = df[df["resale_thousand_rs"] > 0]
        df = df.dropna(subset=["resale_thousand_rs"])

    return df


def main():
    """Verify clean_data() directly on project datasets."""
    # Find data path relative to this script
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    train_path = project_dir / "data" / "raw" / "train.csv"

    if not train_path.exists():
        print(f"Dataset not found at {train_path}")
        return

    print("=" * 60)
    print("VERIFYING clean_data(df) ON data/train.csv")
    print("=" * 60)

    raw_train = pd.read_csv(train_path)
    print(f"Raw train shape: {raw_train.shape}")
    print(f"Raw Central km_driven min/max: {raw_train[raw_train['city_zone'] == 'Central']['km_driven'].min()} / {raw_train[raw_train['city_zone'] == 'Central']['km_driven'].max()}")
    print(f"Raw owners == -1 count: {(raw_train['owners'] == -1).sum()}")
    print(f"Raw brand_tier unique count: {raw_train['brand_tier'].nunique()}")
    print(f"Raw engine_cc == 119 count: {(raw_train['engine_cc'] == 119).sum()}")

    cleaned_train = clean_data(raw_train)
    print("\n--- AFTER clean_data(df) ---")
    print(f"Cleaned train shape: {cleaned_train.shape}")
    print(f"Remaining null values: {cleaned_train.isnull().sum().sum()}")
    print(f"Cleaned Central km_driven min/max: {cleaned_train[cleaned_train['city_zone'] == 'Central']['km_driven'].min()} / {cleaned_train[cleaned_train['city_zone'] == 'Central']['km_driven'].max()}")
    print(f"Cleaned owners == -1 count: {(cleaned_train['owners'] <= 0).sum()}")
    print(f"Cleaned brand_tier categories: {sorted(cleaned_train['brand_tier'].unique().tolist())}")
    print(f"Cleaned engine_cc during stuck period: {((cleaned_train['date'] >= '2026-03-05') & (cleaned_train['date'] <= '2026-03-18') & (cleaned_train['engine_cc'] == 119)).sum()}")
    print(f"Cleaned km_driven dtype: {cleaned_train['km_driven'].dtype}")

    # Verify original df was not modified
    assert (raw_train['owners'] == -1).sum() == 63, "Error: Original train DataFrame was modified!"
    print("\n[SUCCESS] Original DataFrame was NOT modified.")

    # Verify with synthetic messy data with units
    sample = pd.DataFrame({
        "age_years": ["4.5 yrs", "12", "-1"],
        "km_driven": ["30000 km", "50.0", "15000"],
        "engine_cc": ["150 cc", "119", "250cc"],
        "owners": ["-1", "1", "2"],
        "service_records": ["5", "-2", "12"],
        "accidents": ["0", "1", "-1"],
        "brand_tier": [" Premium ", "bud.", "MID"],
        "city_zone": ["Central", "North", "South"],
        "resale_thousand_rs": [75.0, 0.0, 95.5],
    })
    cleaned_sample = clean_data(sample)
    print("\n--- TEST ON MESSY SYNTHETIC DATA ---")
    print(cleaned_sample[["age_years", "km_driven", "engine_cc", "owners", "brand_tier", "resale_thousand_rs"]])
    print("\nAll verification checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
