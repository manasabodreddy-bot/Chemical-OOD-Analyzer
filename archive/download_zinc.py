import deepchem as dc
import pandas as pd
import os

# Load a small ZINC subset via DeepChem
tasks, datasets, transformers = dc.molnet.load_zinc15(
    reload=False,
    data_dir="../data"
)

dataset = datasets[0]   # use train split
smiles = dataset.ids[:5000]  # keep it small and fast

df = pd.DataFrame({"smiles": smiles})

os.makedirs("../data", exist_ok=True)
df.to_csv("../data/zinc_raw.csv", index=False)

print("Saved ZINC subset:", len(df))

