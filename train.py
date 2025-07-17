import mltable
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import ModelType
from pickle import dump


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

from pickle import dump
with open("model.pkl", "wb") as f:
    dump(clf, f, protocol=5)

# Evaluate
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))

ml_client = MLClient.from_config(DefaultAzureCredential())

model = Model(
    path="./model.pkl",
    name="logistic_regression",
    description="A sample logistic regression model for the Diabetes dataset",
    tags={"type": "logistic_regression"},
    type=ModelType.MLFLOW,
    properties=classification_report(y_test, y_pred, output_dict=True)
)

registered_model = ml_client.models.create_or_update(model)
