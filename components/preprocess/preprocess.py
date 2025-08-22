import os

import mltable
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str)
parser.add_argument("--training_data", type=str)
parser.add_argument("--test_data", type=str)
args = parser.parse_args()

print(f"args\n\n{args}")


# Load MLTable dataset
diabetes_dataset = mltable.load(args.raw_data).to_pandas_dataframe()
print(diabetes_dataset.head())

x_train, x_test, y_train, y_test = train_test_split(
    diabetes_dataset.drop("Outcome", axis=1),
    diabetes_dataset["Outcome"],
    stratify=diabetes_dataset["Outcome"],
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(f"parent parent folder:\n {os.listdir(Path(args.training_data).parents[2])}")
print(f"parent folder:\n {os.listdir(Path(args.training_data).parents[1])}")
np.save(Path(args.training_data) / "x_train.npy", x_train)
np.save(Path(args.test_data) / "x_test.npy", x_test)
