import pandas as pd
from pathlib import Path

RAW_DATA = Path("data/raw/creditcard.csv")
REF_DIR = Path("data/reference")
INC_DIR = Path("data/incoming")

REF_DIR.mkdir(exist_ok=True)
INC_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RAW_DATA)

# Drop target column for input drift (important)
if "Class" in df.columns:
    df = df.drop(columns=["Class"])

# Simple split to simulate time
reference_df = df.sample(frac=0.7, random_state=42)
incoming_df = df.drop(reference_df.index)

reference_df.to_csv(REF_DIR / "reference.csv", index=False)
incoming_df.to_csv(INC_DIR / "incoming.csv", index=False)

print("Reference & incoming monitoring data created")
