import os
import mltable
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential
from azure.ai.ml.entities import Model
from pickle import dump
import json
import mlflow
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration
from azure.ai.ml.entities import Environment


parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str)
args = parser.parse_args()

print(f"args\n\n{args}")

print(f"env:\n\n{json.dumps(dict(os.environ), indent=4)}")
cred = ManagedIdentityCredential()
token = cred.get_token("https://management.azure.com/.default")

ml_client = MLClient(
    credential=cred,
    subscription_id=os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
    resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
    workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
)

env = Environment(
    name="sklearn_juicebase",
    version="15",
    #conda_file="conda-dependencies.yml",
    build="Dockerfile"
)
ml_client.environments.create_or_update(env)

mlflow.start_run()
mlflow.autolog()

# Load MLTable dataset
diabetes_dataset = mltable.load(args.training_data).to_pandas_dataframe()
print(diabetes_dataset.head())

x_train, x_test, y_train, y_test = train_test_split(
    diabetes_dataset.drop("Outcome", axis=1),
    diabetes_dataset["Outcome"],
    stratify=diabetes_dataset["Outcome"],
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train logistic regression
clf = LogisticRegression()
clf.fit(x_train, y_train)

# Save Model
mlflow.sklearn.save_model(
    sk_model=clf,
    path="model.pkl",
)
# with open("model.pkl", "wb") as f:
#     dump(clf, f, protocol=5)

# Evaluate
y_pred = clf.predict(x_test)
eval = classification_report(y_test, y_pred, output_dict=True)
# Log scalar metrics only
for metric, value in eval.items():
    if isinstance(value, dict):
        for sub_metric, sub_value in value.items():
            if isinstance(sub_value, (int, float)):
                mlflow.log_metric(f"{metric}_{sub_metric}", sub_value)
    elif isinstance(value, (int, float)):
        mlflow.log_metric(metric, value)

display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
display.figure_.savefig("confusion_matrix.png")
#mlflow.log_artifact("confusion_matrix.png")

mlflow.end_run()

model = Model(
    path="./model.pkl",
    name="logistic_regression",
    description="A sample logistic regression model for the Diabetes dataset",
    tags={"type": "logistic_regression"},
    type="custom_model",
    properties=eval,
)

registered_model = ml_client.models.create_or_update(model)

endpoint = ManagedOnlineEndpoint(
    name="diabetes-endpoint", description="Managed endpoint for Diabetes model"
)

print(f"endpoint: {endpoint}")

ml_client.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="blue",  # deployment name
    endpoint_name="diabetes-endpoint",
    model=model,
    environment="azureml:sklearn_juicebase@15",
    code_configuration=CodeConfiguration(
        code="./",  # folder with score.py
        scoring_script="score.py",
    ),
    # compute="azureml:DS11v2lp",
    # settings={"default_compute": "DS11v2lp"},
    instance_type="Standard_D2as_v4",
    instance_count=1,
)

envs = ml_client.environments.list()
for e in envs:
    print(e.name, e.version, e.id)

print("Deployment env:", deployment.environment)

print(f"deployment: {deployment}")

ml_client.begin_create_or_update(deployment).result()

# Set this deployment as default
ml_client.online_endpoints.begin_update(
    endpoint_name="diabetes-endpoint", traffic={"blue": 100}
)
