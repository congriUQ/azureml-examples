import os
import mltable
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential
from azure.ai.ml.entities import Model
from pickle import dump
import json
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration


parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str)
args = parser.parse_args()

print(f"args\n\n{args}")


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

with open("model.pkl", "wb") as f:
    dump(clf, f, protocol=5)

# Evaluate
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))

print(f"env:\n\n{json.dumps(dict(os.environ), indent=4)}")
cred = ManagedIdentityCredential()
token = cred.get_token("https://management.azure.com/.default")

ml_client = MLClient(
    credential=cred,
    subscription_id=os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
    resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
    workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
)

model = Model(
    path="./model.pkl",
    name="logistic_regression",
    description="A sample logistic regression model for the Diabetes dataset",
    tags={"type": "logistic_regression"},
    type="custom_model",
    properties=classification_report(y_test, y_pred, output_dict=True),
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
    environment="azureml:diabetes:5",
    code_configuration=CodeConfiguration(
        code="./",  # folder with score.py
        scoring_script="score.py",
    ),
    instance_type="Standard_D2as_v4",
    instance_count=1,
)


print(f"deployment: {deployment}")

ml_client.begin_create_or_update(deployment).result()

# Set this deployment as default
ml_client.online_endpoints.begin_update(
    endpoint_name="diabetes-endpoint", traffic={"blue": 100}
)
