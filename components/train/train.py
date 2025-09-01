import argparse
import os
import json
from pathlib import Path
from pickle import dump

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from azureml.core import Run


parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--max_epochs", type=int, help="Max # of epochs for the training")
parser.add_argument("--learning_rate", type=float, help="Learning rate")
parser.add_argument("--learning_rate_schedule", type=str, help="Learning rate schedule")
parser.add_argument("--model_output", type=str, help="Path of output model")
parser.add_argument("--parameter_output", type=str, help="Path of output parameters")

args = parser.parse_args()

print("hello training world...")

lines = [
    f"Training data path: {args.training_data}",
    f"Max epochs: {args.max_epochs}",
    f"Learning rate: {args.learning_rate}",
    f"Learning rate schedule: {args.learning_rate_schedule}",
    f"Model output path: {args.model_output}",
    f"Parameter output path: {args.parameter_output}",
]

for line in lines:
    print(line)

print("mounted_path files: ")
arr = os.listdir(args.training_data)
print(arr)

# Load training data
x_train = np.load(Path(args.training_data) / "x_train.npy")
y_train = np.load(Path(args.training_data) / "y_train.npy")

# Train model
clf = LogisticRegression()
clf.fit(x_train, y_train)

# Save model
model_output_path = Path(args.model_output) / "model.pkl"
model_output_path.parent.mkdir(parents=True, exist_ok=True)
with open(model_output_path, "wb") as model_file:
    dump(clf, model_file, protocol=5)

# Collect hyperparameters
hyperparams = {
    "max_epochs": args.max_epochs,
    "learning_rate": args.learning_rate,
    "learning_rate_schedule": args.learning_rate_schedule,
}
for param in hyperparams:
    mlflow.log_param(f"{param}", hyperparams[param])

# Save hyperparameters to JSON
param_output_path = Path(args.parameter_output) / "hyperparams.json"
param_output_path.parent.mkdir(parents=True, exist_ok=True)
with open(param_output_path, "w") as f:
    json.dump(hyperparams, f)

run = Run.get_context()
print("Experiment:", run.experiment.name)
print("Run ID:", run.id)

# save run info to output so registration step can use it
run_output_path = Path(args.parameter_output) / "run_info.json"
with open(run_output_path, "w") as f:
    f.write(json.dumps({"experiment_name": run.experiment.name, "azureml.run_id": run.id}))
