import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report



parser = argparse.ArgumentParser("score")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--scoring_result", type=str, help="Path of scoring result")
parser.add_argument("--eval_output", type=str, help="Path of output evaluation result")

args = parser.parse_args()

print("hello evaluation world...")

lines = [
    f"Scoring result path: {args.scoring_result}",
    f"Evaluation output path: {args.eval_output}",
]

for line in lines:
    print(line)

# Evaluate the incoming scoring result and output evaluation result.
# Here only output a dummy file for demo.
y_test = np.load(Path(args.test_data) / "y_test.npy")
y_pred = np.load(Path(args.scoring_result) / "y_pred.npy")

clf_report = classification_report(y_test, y_pred)
print(classification_report(y_test, y_pred))
