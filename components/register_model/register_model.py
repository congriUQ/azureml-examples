import argparse
import json
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import ManagedIdentityCredential


parser = argparse.ArgumentParser("score")
parser.add_argument("--model", type=str)
parser.add_argument("--eval_report", type=str, help="Path of output evaluation result")
args = parser.parse_args()


print(f"env:\n\n{json.dumps(dict(os.environ), indent=4)}")
cred = ManagedIdentityCredential()
token = cred.get_token("https://management.azure.com/.default")

ml_client = MLClient(
    credential=cred,
    subscription_id=os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
    resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
    workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
)

with open(Path(args.eval_report) / "eval_report.json") as f:
    eval_report = json.load(f)


model = Model(
    path="./model.pkl",
    name="logistic_regression",
    description="A sample logistic regression model for the Diabetes dataset",
    tags={"type": "logistic_regression"},
    type="custom_model",
    properties=eval_report,
)

registered_model = ml_client.models.create_or_update(model)
