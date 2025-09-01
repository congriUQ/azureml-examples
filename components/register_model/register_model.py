import argparse
import json
import os
from pathlib import Path

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import ManagedIdentityCredential


parser = argparse.ArgumentParser("register")
parser.add_argument("--model", type=str, help="Path to trained model directory")
parser.add_argument("--eval_report", type=str, help="Path of output evaluation result directory")
parser.add_argument("--hyperparameters", type=str, help="Path of output hyperparameters directory")
parser.add_argument("--accuracy_threshold", type=float, help="Minimum accuracy required")
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
mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load evaluation report
with open(Path(args.eval_report) / "eval_report.json") as f:
    eval_report = json.load(f)

accuracy = eval_report["accuracy"]
mlflow.log_metric("accuracy", accuracy)
print(f"Gate check: accuracy={accuracy}, threshold={args.accuracy_threshold}")

# Decide approval
approve = accuracy >= args.accuracy_threshold
print(f"Gate decision: {'approve' if approve else 'reject'}")
if not approve:
    print("Model not approved, will not be registered.")
    exit(0)

# Load hyperparameters
with open(Path(args.hyperparameters) / "hyperparams.json") as f:
    hyperparams = json.load(f)

for param in hyperparams:
    mlflow.log_param(f"{param}", hyperparams[param])

# Combine metadata for model registration
properties = {**eval_report, **hyperparams}

model = Model(
    path=(Path(args.model) / "model.pkl"),
    name="logistic_regression",
    description="A sample logistic regression model for the Diabetes dataset",
    tags={"type": "logistic_regression"},
    type="custom_model",
    properties=properties,
)

registered_model = ml_client.models.create_or_update(model)
print(f"Model registered: {registered_model.name} v{registered_model.version}")

