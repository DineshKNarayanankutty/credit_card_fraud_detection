import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def download_model_from_aml():
    """
    Download model artifacts from Azure ML Model Registry
    using Managed Identity.
    """

    model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
    scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")

    # Skip download if already present (container restart safe)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Model artifacts already present, skipping download.")
        return

    print("Downloading model from Azure ML Model Registry...")

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZURE_ML_WORKSPACE"),
    )

    model = ml_client.models.get(
        name=os.getenv("AML_MODEL_NAME"),
        label=os.getenv("AML_MODEL_LABEL", "latest")
    )

    model.download(
        path=ARTIFACT_DIR,
        exist_ok=True
    )

    print("Model artifacts downloaded successfully.")
