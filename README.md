Credit Card Fraud Detection — Production-Grade MLOps Pipeline

This repository demonstrates how a machine learning model is built, versioned, evaluated, and monitored as a production system, not just a notebook experiment.

The project focuses on end-to-end MLOps practices using DVC, MLflow, modular pipelines, and drift-ready architecture, while keeping the ML model intentionally simple so the emphasis remains on system design and reliability.

🚀 Project Goals

Build a reproducible ML training pipeline

Track data, parameters, and artifacts using DVC

Log experiments and metrics with MLflow

Serve models via clean inference abstractions

Prepare the system for production monitoring and drift detection

Keep the architecture cloud-ready (Azure ML compatible)

🧠 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where:

Accuracy alone is misleading

Recall and precision trade-offs matter

Threshold tuning is critical

Data drift is common in production

This project treats fraud detection as a system problem, not just a modeling task.

🏗️ High-Level Architecture
Raw Data (DVC)
   ↓
Preprocessing Pipeline
   ↓
Train / Validate / Test Split
   ↓
Model Training + Cross-Validation
   ↓
Threshold Optimization
   ↓
Evaluation & Reporting
   ↓
Model + Artifacts (DVC + MLflow)
   ↓
Inference / Monitoring (Drift-Ready)

📂 Repository Structure
credit_card_fraud_detection/
│
├── api/                     # FastAPI inference service
│
├── pipelines/               # Orchestration only (DVC / Azure ML ready)
│   ├── train_pipeline.py
│   ├── evaluate_pipeline.py
│   └── drift_pipeline.py
│
├── src/
│   ├── data/                # Data loading, cleaning, splitting
│   ├── features/            # Feature engineering (imbalance handling)
│   ├── models/              # Model factory, training, registry
│   ├── evaluation/          # Metrics, thresholding, reports
│   ├── inference/           # Prediction abstractions
│   ├── monitoring/          # Drift detection contract
│   └── utils/               # Config, logging, IO
│
├── data/
│   ├── raw/                 # Raw datasets (DVC tracked)
│   ├── processed/           # Train/val/test splits (DVC outputs)
│   ├── reference/           # Baseline data for drift checks
│   └── incoming/            # New production data
│
├── artifacts/               # Models, scalers, metrics (DVC outputs)
├── reports/                 # Human-readable evaluation & drift reports
│
├── dvc.yaml                 # Pipeline definition
├── dvc.lock                 # Reproducibility lockfile
├── params.yaml              # Tunable ML parameters
├── requirements.txt
└── README.md

🔁 Pipeline Stages (DVC)
1️⃣ Preprocessing

Load raw CSV

Clean data (missing values, duplicates, outliers)

Train / validation / test split

Fit scaler on train only

Persist splits and scaler

dvc repro preprocess

2️⃣ Training

Load processed data

Handle class imbalance (SMOTE on train only)

Train model

Cross-validate

Optimize decision threshold

Evaluate on test set

Save model and metrics

dvc repro train

3️⃣ Evaluation

Load trained model and scaler

Run threshold analysis

Generate evaluation reports

Persist metrics for tracking

dvc repro evaluate

4️⃣ Drift Detection (Contract-Based)

Checks for presence of reference & incoming data

Emits drift signals without breaking pipelines

Designed for post-deployment monitoring tools

Compatible with Evidently / Azure ML Monitoring

dvc repro drift_check


Drift detection is intentionally decoupled from training to keep pipelines deterministic and production-safe.

📊 Experiment Tracking (MLflow)

MLflow is used to:

Track metrics and parameters

Compare experiment runs

Prepare for remote tracking backends (Azure ML)

Start UI locally:

mlflow ui

🧪 Model Performance (Example)

Imbalanced dataset (~0.2% fraud)

Accuracy alone is misleading

Threshold tuning improves recall

Evaluation focuses on:

Precision

Recall

F1-score

ROC-AUC

PR-AUC

🔐 Design Principles

Separation of concerns

Pipelines orchestrate

Modules implement logic

No data leakage

Reproducibility first

Monitoring ≠ Training

Cloud-agnostic by default

☁️ Cloud & Deployment Readiness

This project is intentionally structured to support:

Azure ML Jobs

Azure Blob Storage (DVC remote)

AKS / Container deployment

Production monitoring tools

Next phase: Azure ML integration for training orchestration and registry.

🧩 Why This Project Matters

This repository demonstrates:

Real MLOps engineering (not tutorials)

Correct handling of imbalanced data

Clean pipeline orchestration

Drift-ready system design

Interview-grade architecture decisions

🔜 Next Steps

Integrate Azure ML training jobs

Configure MLflow remote backend

Containerize inference service

Deploy to AKS

Add live monitoring dashboards
