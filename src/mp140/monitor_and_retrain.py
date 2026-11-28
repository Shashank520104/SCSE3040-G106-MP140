from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from src.mp140.preprocess import clean_data
from src.mp140.train import FEATURE_COLS, NUMERIC_COLS, build_pipeline, compute_sha256


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_percents = np.where(expected_counts == 0, 1e-6, expected_counts / len(expected))
    actual_percents = np.where(actual_counts == 0, 1e-6, actual_counts / len(actual))

    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi_value)


def audit_seven_issues(raw_train: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    target_col = "resale_thousand_rs"

    # 1. Leaking column: in train, missing in holdout (excluding target)
    extra_cols = [c for c in raw_train.columns if c not in holdout.columns and c != target_col]
    leaking_present = len(extra_cols) > 0
    leaking_col = extra_cols[0] if leaking_present else None

    # 2. Unit mixup: Central zone km_driven recorded in thousands (factor = 1000)
    central_mean = pd.to_numeric(raw_train.loc[raw_train["city_zone"] == "Central", "km_driven"], errors="coerce").mean()
    north_mean = pd.to_numeric(raw_train.loc[raw_train["city_zone"] == "North", "km_driven"], errors="coerce").mean()
    unit_mixup_present = bool(north_mean / max(central_mean, 1e-6) > 500)

    # 3. Sentinel value: owners == -1
    owners_num = pd.to_numeric(raw_train["owners"], errors="coerce")
    sentinel_count = int((owners_num == -1).sum())

    # 4. Duplicate rows
    dup_count = int(raw_train.duplicated().sum())

    # 5. Stuck sensor: engine_cc == 119
    stuck_rows = raw_train[pd.to_numeric(raw_train["engine_cc"], errors="coerce") == 119].sort_values("date")
    stuck_present = len(stuck_rows) > 5
    start_date = str(stuck_rows["date"].min()) if stuck_present else None
    end_date = str(stuck_rows["date"].max()) if stuck_present else None

    # 6. Target typos: resale_thousand_rs 10x too big (> 400)
    target_num = pd.to_numeric(raw_train[target_col], errors="coerce")
    typo_count = int((target_num > 400).sum())

    # 7. Spelling variants: brand_tier
    spelling_present = raw_train["brand_tier"].nunique() > 3

    return {
        "leaking_column": {
            "present": leaking_present,
            "column": leaking_col,
        },
        "unit_mixup": {
            "present": unit_mixup_present,
            "column": "km_driven" if unit_mixup_present else None,
            "group_value": "Central" if unit_mixup_present else None,
            "factor": 1000 if unit_mixup_present else None,
        },
        "sentinel_value": {
            "present": sentinel_count > 0,
            "column": "owners" if sentinel_count > 0 else None,
            "value": -1 if sentinel_count > 0 else None,
            "count": sentinel_count if sentinel_count > 0 else None,
        },
        "duplicate_rows": {
            "present": dup_count > 0,
            "count": dup_count if dup_count > 0 else None,
        },
        "stuck_sensor": {
            "present": stuck_present,
            "column": "engine_cc" if stuck_present else None,
            "start_date": start_date,
            "end_date": end_date,
        },
        "target_typos": {
            "present": typo_count > 0,
            "count": typo_count if typo_count > 0 else None,
        },
        "spelling_variants": {
            "present": spelling_present,
            "column": "brand_tier" if spelling_present else None,
            "true_levels": 3 if spelling_present else None,
        },
    }


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_train_path = Path(params["data"]["raw_train_path"])
    raw_later_path = Path(params["data"]["raw_later_path"])
    holdout_path = Path(params["data"]["holdout_path"])
    target_col = params["model"]["target_col"]
    random_state = params["model"]["random_state"]

    raw_train = pd.read_csv(raw_train_path)
    later_df = pd.read_csv(raw_later_path)
    holdout_df = pd.read_csv(holdout_path)

    data_issues = audit_seven_issues(raw_train, holdout_df)

    clean_train = clean_data(raw_train)
    clean_later = clean_data(later_df)

    # 1. PSI for every numeric column
    psi_scores = {}
    for col in NUMERIC_COLS:
        psi_val = calculate_psi(clean_train[col].to_numpy(), clean_later[col].to_numpy(), bins=10)
        psi_scores[col] = round(psi_val, 4)

    most_drifted_feature = max(psi_scores, key=psi_scores.get)

    # 2. Load v1 champion model
    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]
    registered_model_name = params["mlflow"]["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    v1_info = client.get_model_version_by_alias(registered_model_name, "champion")
    v1_run_id = v1_info.run_id
    v1_model = joblib.load(params["artifacts"]["champion_model_path"])

    # v1 MAE on later.csv per city_zone
    mae_by_group = {}
    for zone in ["North", "South", "Central"]:
        sub = clean_later[clean_later["city_zone"] == zone]
        preds_zone = v1_model.predict(sub[FEATURE_COLS])
        mae_by_group[zone] = round(float(mean_absolute_error(sub[target_col], preds_zone)), 4)

    worst_served_group = max(mae_by_group, key=mae_by_group.get)

    # 3. Sort later.csv by date, split 70% / 30%
    later_sorted = clean_later.sort_values("date").reset_index(drop=True)
    split_idx = int(len(later_sorted) * 0.70)
    later_train_70 = later_sorted.iloc[:split_idx]
    later_check_30 = later_sorted.iloc[split_idx:]

    combined_train = pd.concat([clean_train, later_train_70], ignore_index=True)

    # Retrain v2 using best v1 config (ridge_alpha_1)
    best_v1_cfg = params["model"]["runs"][0]
    v2_pipeline = build_pipeline(best_v1_cfg, random_state)
    v2_pipeline.fit(combined_train[FEATURE_COLS], combined_train[target_col])

    mae_v1_on_check = round(float(mean_absolute_error(later_check_30[target_col], v1_model.predict(later_check_30[FEATURE_COLS]))), 4)
    mae_v2_on_check = round(float(mean_absolute_error(later_check_30[target_col], v2_pipeline.predict(later_check_30[FEATURE_COLS]))), 4)

    improvement_ratio = (mae_v1_on_check - mae_v2_on_check) / mae_v1_on_check
    decision = "promote" if improvement_ratio >= 0.15 else "keep"

    data_sha256 = compute_sha256(raw_train_path)
    with mlflow.start_run(run_name="v2_retrained_ridge") as run_v2:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_version_stage", "v2")
        mlflow.log_param("promotion_decision", decision)
        mlflow.log_metric("mae", mae_v2_on_check)
        mlflow.log_metric("mae_v1_on_check", mae_v1_on_check)
        mlflow.sklearn.log_model(
            sk_model=v2_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )
        v2_run_id = run_v2.info.run_id

    mv2 = mlflow.register_model(model_uri=f"runs:/{v2_run_id}/model", name=registered_model_name)

    if decision == "promote":
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="champion",
            version=mv2.version,
        )
        champion_model = v2_pipeline
        champion_version = int(mv2.version)
        joblib.dump(v2_pipeline, params["artifacts"]["champion_model_path"])
    else:
        champion_model = v1_model
        champion_version = int(v1_info.version)

    # 4. Holdout predictions using @champion
    clean_holdout = clean_data(holdout_df)
    holdout_preds = champion_model.predict(clean_holdout[FEATURE_COLS])

    pred_out = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": np.round(holdout_preds, 4),
    })
    pred_path = Path(params["artifacts"]["predictions_path"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_out.to_csv(pred_path, index=False)

    # 5. Populate findings.yaml
    findings = {
        "group_id": "G106",
        "problem_id": "MP140",
        "data_issues": data_issues,
        "tracking": {
            "experiment_name": experiment_name,
            "best_run_id": v1_run_id,
            "registered_model_name": registered_model_name,
            "champion_version": champion_version,
        },
        "monitoring": {
            "psi": psi_scores,
            "most_drifted_feature": most_drifted_feature,
            "mae_by_group": mae_by_group,
            "worst_served_group": worst_served_group,
            "mae_v1_on_check": mae_v1_on_check,
            "mae_v2_on_check": mae_v2_on_check,
            "decision": decision,
        },
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/findings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)
    with open("findings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)

    print("\n" + "=" * 60)
    print("STAGE 4 RESULTS SUMMARY")
    print("=" * 60)
    print(yaml.safe_dump(findings, sort_keys=False))
    print(f"Saved {len(pred_out)} holdout predictions to {pred_path}")


if __name__ == "__main__":
    main()