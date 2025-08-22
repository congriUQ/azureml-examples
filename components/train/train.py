import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from pickle import dump


parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--max_epochs", type=int, help="Max # of epochs for the training")
parser.add_argument("--learning_rate", type=float, help="Learning rate")
parser.add_argument("--learning_rate_schedule", type=str, help="Learning rate schedule")
parser.add_argument("--model_output", type=str, help="Path of output model")

args = parser.parse_args()

print("hello training world...")

lines = [
    f"Training data path: {args.training_data}",
    f"Max epochs: {args.max_epochs}",
    f"Learning rate: {args.learning_rate}",
    f"Learning rate: {args.learning_rate_schedule}",
    f"Model output path: {args.model_output}",
]

for line in lines:
    print(line)

print("mounted_path files: ")
arr = os.listdir(args.training_data)
print(arr)

x_train = np.load(Path(args.training_data) / "x_train.npy")
y_train = np.load(Path(args.training_data) / "y_train.npy")

clf = LogisticRegression()
clf.fit(x_train, y_train)

with open(Path(args.model_output) / "model.pkl", "wb") as model_file:
    dump(clf, model_file, protocol=5)


# Do the train and save the trained model as a file into the output folder.
# Here only output a dummy data for demo.
curtime = datetime.now().strftime("%b-%d-%Y %H:%M:%S")
model = f"This is a dummy model with id: {str(uuid4())} generated at: {curtime}\n"
