import hashlib
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.mp140.preprocess import clean_data

FEATURE_COLS = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
    "brand_tier",
    "city_zone",
]

NUMERIC_COLS = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
]

CATEGORICAL_COLS = [
    "brand_tier",
    "city_zone",
]


def compute_sha256(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(run_cfg: dict, random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    model_type = run_cfg["type"]
    if model_type == "ridge":
        regressor = Ridge(alpha=run_cfg.get("alpha", 1.0), random_state=random_state)
    elif model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=run_cfg.get("n_estimators", 100),
            max_depth=run_cfg.get("max_depth", 10),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        regressor = GradientBoostingRegressor(
            n_estimators=run_cfg.get("n_estimators", 150),
            learning_rate=run_cfg.get("learning_rate", 0.1),
            max_depth=run_cfg.get("max_depth", 4),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_train_path = Path(params["data"]["raw_train_path"])
    target_col = params["model"]["target_col"]
    test_size = params["model"]["test_size"]
    random_state = params["model"]["random_state"]

    data_sha256 = compute_sha256(raw_train_path)
    raw_df = pd.read_csv(raw_train_path)
    clean_df = clean_data(raw_df)

    X = clean_df[FEATURE_COLS]
    y = clean_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]
    registered_model_name = params["mlflow"]["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    best_mae = float("inf")
    best_run_id = None
    best_pipeline = None
    best_run_name = None

    for run_cfg in params["model"]["runs"]:
        run_name = run_cfg["name"]
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_version_stage", "v1")
            for k, v in run_cfg.items():
                mlflow.log_param(k, v)

            pipeline = build_pipeline(run_cfg, random_state)
            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_val)
            mae = float(mean_absolute_error(y_val, preds))
            r2 = float(r2_score(y_val, preds))

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(f"Run '{run_name}' | Run ID: {run.info.run_id} | MAE: {mae:.4f} | R2: {r2:.4f}")

            if mae < best_mae:
                best_mae = mae
                best_run_id = run.info.run_id
                best_pipeline = pipeline
                best_run_name = run_name

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=mv.version,
    )

    champion_path = Path(params["artifacts"]["champion_model_path"])
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, champion_path)

    print("\n" + "=" * 60)
    print(f"Best v1 Model: {best_run_name} (MAE: {best_mae:.4f})")
    print(f"Best Run ID: {best_run_id}")
    print(f"Registered as '{registered_model_name}' version {mv.version} with @champion alias")
    print(f"Saved champion artifact to {champion_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
