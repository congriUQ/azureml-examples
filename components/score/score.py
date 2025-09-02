import argparse
from pathlib import Path
import pickle
import numpy as np
import mlflow.sklearn


parser = argparse.ArgumentParser("score")
parser.add_argument("--model_input", type=str, help="Path of input model")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--score_output", type=str, help="Path of scoring output")

args = parser.parse_args()

print("hello scoring world...")

lines = [
    f"Model path: {args.model_input}",
    f"Test data path: {args.test_data}",
    f"Scoring output path: {args.score_output}",
]

for line in lines:
    print(line)

# model_path = Path(args.model_input) / "model.pkl"
# with open(model_path, "rb") as model_file:
#     model = pickle.load(model_file)
model = mlflow.sklearn.load_model(args.model_input)
print("Model: ", model)

# score
x_test = np.load(Path(args.test_data) / "x_test.npy")
y_pred = model.predict(x_test)
np.save(Path(args.score_output) / "y_pred.npy", y_pred)
