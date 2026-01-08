from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import os

def download_model_from_aml():
    model_name = os.environ["AML_MODEL_NAME"]
    model_label = os.environ.get("AML_MODEL_LABEL", "latest")

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    target_dir = "/app/artifacts"
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading model '{model_name}:{model_label}' to {target_dir}")

    model = ml_client.models.get(
        name=model_name,
        label=model_label,
    )

    ml_client.models.download(
        name=model.name,
        version=model.version,
        download_path=target_dir,
    )

    # 🔍 IMPORTANT PART — find actual files
    model_path = None
    scaler_path = None

    for root, _, files in os.walk(target_dir):
        if "model.pkl" in files:
            model_path = os.path.join(root, "model.pkl")
        if "scaler.pkl" in files:
            scaler_path = os.path.join(root, "scaler.pkl")

    if not model_path:
        raise FileNotFoundError("model.pkl not found after AML download")

    if not scaler_path:
        raise FileNotFoundError("scaler.pkl not found after AML download")

    # ✅ Expose paths to the API
    os.environ["MODEL_PATH"] = model_path
    os.environ["SCALER_PATH"] = scaler_path

    print(f"Resolved MODEL_PATH={model_path}")
    print(f"Resolved SCALER_PATH={scaler_path}")
    print("Model download completed")
