from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

ID_PATH = "../data/splits/id_smiles.csv"
OOD_PATH = "../data/splits/ood_smiles.csv"
OUTPUT_DIR = "../results"

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

print("Loading data...")
id_df = pd.read_csv(ID_PATH)
ood_df = pd.read_csv(OOD_PATH)

print("Extracting scaffolds...")
id_scaffolds = [get_scaffold(s) for s in id_df["smiles"]]
ood_scaffolds = [get_scaffold(s) for s in ood_df["smiles"]]

id_scaffolds = [s for s in id_scaffolds if s]
ood_scaffolds = [s for s in ood_scaffolds if s]

id_counts = Counter(id_scaffolds)
ood_counts = Counter(ood_scaffolds)

print("Unique ID scaffolds:", len(id_counts))
print("Unique OOD scaffolds:", len(ood_counts))

shared = set(id_counts.keys()) & set(ood_counts.keys())
print("Shared scaffolds between ID and OOD:", len(shared))

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.figure()
plt.hist(id_counts.values(), bins=50, alpha=0.7)
plt.xlabel("Molecules per scaffold (ID)")
plt.ylabel("Count")
plt.title("ID scaffold size distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "id_scaffold_sizes.png"))

plt.figure()
plt.hist(ood_counts.values(), bins=50, alpha=0.7)
plt.xlabel("Molecules per scaffold (OOD)")
plt.ylabel("Count")
plt.title("OOD scaffold size distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ood_scaffold_sizes.png"))

print("Scaffold statistics completed.")
