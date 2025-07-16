import mltable
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str)
args = parser.parse_args()

print(f"args\n\n{args}")


# Load MLTable dataset
tbl = mltable.load(args.training_data)
df = tbl.to_pandas_dataframe()
print(df.head())
