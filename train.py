from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd

# Create ML client (works in local dev; in Azure job use managed identity)
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE"],
)

# Get registered dataset (Asset) by name
data_asset = ml_client.data.get(name="my_registered_dataset", version="1")

# Load dataset file path
data_path = data_asset.path

# Example: if it’s a CSV
df = pd.read_csv(data_path)
print(df.head())
