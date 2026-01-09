from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Project is mounted at /opt/project
PROJECT_ROOT = Path("/opt/project")
sys.path.append(str(PROJECT_ROOT))

from src.monitoring.drift import run_drift_detection


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="credit_card_fraud_monitoring",
    default_args=default_args,
    description="Batch drift detection for fraud model",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "monitoring"],
) as dag:

    run_drift = PythonOperator(
        task_id="run_drift_detection",
        python_callable=run_drift_detection,
    )

    run_drift
