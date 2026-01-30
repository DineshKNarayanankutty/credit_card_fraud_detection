# Credit Card Fraud Detection — Production-Grade MLOps System

This repository demonstrates how a **machine learning model is built, tracked, registered, and served as a production system**, not just a notebook experiment.

The project focuses on **end-to-end MLOps practices** using **DVC, MLflow, Azure ML, and FastAPI**, with a strong emphasis on **reproducibility, model governance, and cloud deployment**.
The ML model itself is intentionally simple so the focus remains on **system design, reliability, and production readiness**.

---

## 🚀 Project Objectives

* Build a **fully reproducible ML pipeline** (data → model → evaluation)
* Track **data, parameters, and artifacts** using DVC
* Log experiments, metrics, and artifacts with **MLflow**
* Train and **register models in Azure ML**
* Serve predictions through a **containerized FastAPI service**
* Deploy inference to **Azure Web App for Containers**
* Follow **production MLOps design principles**

---

## 🧠 Problem Statement

Credit card fraud detection is a **highly imbalanced classification problem** (~0.2% fraud), where:

* Accuracy alone is misleading
* Recall–precision trade-offs are critical
* Threshold tuning directly impacts business outcomes
* Data behavior can change over time in production

This project treats fraud detection as a **system engineering problem**, not just a modeling exercise.

---

## 🏗️ High-Level Architecture

```
Raw Data (DVC)
   ↓
Preprocessing Pipeline
   ↓
Train / Validation / Test Split
   ↓
Model Training + Cross-Validation
   ↓
Threshold Optimization
   ↓
Evaluation & Reporting
   ↓
Model + Artifacts (DVC + MLflow)
   ↓
Azure ML Model Registry
   ↓
FastAPI Inference Service
   ↓
Azure Web App (Containerized Deployment)
```

---

## 📂 Repository Structure

```
credit_card_fraud_detection/
│
├── api/                     # FastAPI inference service
│
├── pipelines/               # Pipeline orchestration (DVC / Azure ML ready)
│   ├── train_pipeline.py
│   └── evaluate_pipeline.py
│
├── src/
│   ├── data/                # Data loading, cleaning, splitting
│   ├── features/            # Imbalance handling (SMOTE)
│   ├── models/              # Model factory, training, registry
│   ├── evaluation/          # Metrics, thresholding, reports
│   ├── inference/           # Prediction abstractions
│   └── utils/               # Config, logging, IO
│
├── data/
│   ├── raw/                 # Raw dataset (DVC tracked)
│   └── processed/           # Train/val/test splits (DVC outputs)
│
├── artifacts/               # Model, scaler, metrics (DVC outputs)
├── reports/                 # Evaluation reports
│
├── dvc.yaml                 # DVC pipeline definition
├── dvc.lock                 # Reproducibility lockfile
├── params.yaml              # Tunable ML parameters
├── requirements.txt
└── README.md
```

---

## 🔁 Pipeline Stages (DVC)

### 1️⃣ Preprocessing

* Load raw dataset
* Clean data (missing values, duplicates)
* Perform **leakage-safe train/validation/test split**
* Fit scaler **only on training data**
* Persist processed data and scaler

```bash
dvc repro preprocess
```

---

### 2️⃣ Training

* Load processed datasets
* Handle class imbalance using **SMOTE (train only)**
* Train model and perform cross-validation
* Optimize decision threshold using validation data
* Evaluate on test set
* Log metrics and artifacts to **MLflow**
* Save artifacts via **DVC**
* Register model in **Azure ML Model Registry**

```bash
dvc repro train
```

---

### 3️⃣ Evaluation

* Load trained model and scaler
* Run threshold-based evaluation
* Generate human-readable evaluation reports
* Persist evaluation metrics

```bash
dvc repro evaluate
```

---

## 📊 Experiment Tracking (MLflow)

MLflow is used to:

* Log model parameters and hyperparameters
* Track metrics (precision, recall, F1, ROC-AUC, PR-AUC)
* Store artifacts (model, scaler)
* Enable comparison across runs
* Support both **local and Azure ML-backed tracking**

Launch MLflow UI locally:

```bash
mlflow ui
```

---

## ☁️ Azure ML Integration

* Training pipeline is **Azure ML compatible**
* Models are **registered in Azure ML Model Registry**
* Artifacts follow Azure ML output conventions
* Enables versioned, auditable model promotion

---

## 🚀 Inference & Deployment

* Built a **FastAPI-based inference service**
* Supports single and batch predictions
* Loads model and scaler dynamically
* Containerized using Docker
* Deployed on **Azure Web App for Containers**
* Ready for horizontal scaling and CI/CD integration

---

## 🔐 Design Principles

* Strict **separation of concerns**
* Pipelines orchestrate, modules implement logic
* No data leakage
* Deterministic, reproducible runs
* Cloud-first but cloud-agnostic structure
* Production-readiness over experimentation

---

## 🧩 Why This Project Matters

This repository demonstrates:

* Real-world MLOps engineering practices
* Proper handling of highly imbalanced data
* Reproducible ML pipelines with DVC
* Experiment tracking and model governance with MLflow & Azure ML
* End-to-end deployment from training to live inference

---

## 🔜 Potential Extensions

* Azure ML Jobs for fully managed training
* CI/CD pipeline for model promotion
* Centralized MLflow tracking backend
* Live monitoring and drift dashboards
* AKS-based inference deployment
