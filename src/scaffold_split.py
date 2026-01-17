from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import os
from collections import defaultdict
import random

INPUT_PATH = "../data/zinc_smiles.csv"
OUTPUT_DIR = "../data/splits"

RANDOM_SEED = 42
ID_FRACTION = 0.8   # 80% scaffolds for training

random.seed(RANDOM_SEED)

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

print("Loading SMILES...")
df = pd.read_csv(INPUT_PATH)

print("Generating scaffolds...")
scaffold_to_smiles = defaultdict(list)

for smi in df["smiles"]:
    scaffold = get_scaffold(smi)
    if scaffold:
        scaffold_to_smiles[scaffold].append(smi)

scaffolds = list(scaffold_to_smiles.keys())
random.shuffle(scaffolds)

split_idx = int(len(scaffolds) * ID_FRACTION)
id_scaffolds = set(scaffolds[:split_idx])
ood_scaffolds = set(scaffolds[split_idx:])

id_smiles = []
ood_smiles = []

for scaf, smi_list in scaffold_to_smiles.items():
    if scaf in id_scaffolds:
        id_smiles.extend(smi_list)
    else:
        ood_smiles.extend(smi_list)

os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.DataFrame({"smiles": id_smiles}).to_csv(
    os.path.join(OUTPUT_DIR, "id_smiles.csv"), index=False
)
pd.DataFrame({"smiles": ood_smiles}).to_csv(
    os.path.join(OUTPUT_DIR, "ood_smiles.csv"), index=False
)

print(f"ID molecules: {len(id_smiles)}")
print(f"OOD molecules: {len(ood_smiles)}")
print("Scaffold split completed.")
