from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import os
import shutil

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

    # IMPORTANT: clean directory manually (SDK does NOT do this)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading model '{model_name}:{model_label}' to {target_dir}")

    # Step 1: resolve label → version
    model = ml_client.models.get(
        name=model_name,
        label=model_label,
    )

    # Step 2: download by NAME + VERSION ONLY
    ml_client.models.download(
        name=model.name,
        version=model.version,
        download_path=target_dir,
    )

    print("Model download completed")
