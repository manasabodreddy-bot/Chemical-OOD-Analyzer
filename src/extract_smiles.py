import pandas as pd
import os

# UPDATE this path if needed
INPUT_PATH = "../data/zinc15_250K_2D.csv"
OUTPUT_PATH = "../data/zinc_smiles.csv"

df = pd.read_csv(INPUT_PATH)

# ZINC files usually store SMILES in a column called 'smiles'
if "smiles" not in df.columns:
    raise ValueError(f"SMILES column not found. Columns: {df.columns}")

smiles = df["smiles"].dropna().drop_duplicates()

out_df = pd.DataFrame({"smiles": smiles})

os.makedirs("../data", exist_ok=True)
out_df.to_csv(OUTPUT_PATH, index=False)

print(f"Extracted {len(out_df)} unique SMILES to {OUTPUT_PATH}")
