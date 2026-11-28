# SCSE3040 MLOps Mini Project — Group G106 (MP140)

- **Group ID:** G106
- **Problem ID:** MP140 — Used Two-Wheeler Price Prediction (Patna)
- **Course:** SCSE3040 MLOps

## Team Members
| Name | Enrolment Number | Role |
|---|---|---|
| Bendalam Tambu Ashish | S24CSEU0899 | Data & Container Lead |
| Shashank Kumar | S24CSEU0940 | Group Lead / Model & Monitoring Lead |
| Vanshika Gupta | S24CSEU1135 | Testing, CI & Kubernetes Lead |

## Pipeline Overview
- **Data Versioning:** DVC (`data/raw/train.csv.dvc`, `data/raw/later.csv.dvc`)
- **Experiment Tracking & Registry:** MLflow (`sqlite:///mlflow.db`, registered model `mp140-price-model` with `@champion` alias)
- **Prediction Service:** FastAPI (`app/main.py`) with Pydantic validation (`age_years <= 30`, `service_records >= 0`) and JSON stdout logging
- **Container & Orchestration:** `Dockerfile` (`mp140-api:1.0`) and Kubernetes manifests (`k8s/deployment.yaml`, `k8s/service.yaml` with 4 replicas)
- **CI & Monitoring:** GitHub Actions (`.github/workflows/ci.yml`), 10-bin PSI drift detection, v2 retraining evaluation (`src/mp140/monitor_and_retrain.py`), and `reports/findings.yaml`