from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

ID_PATH = "../data/splits/id_smiles.csv"
OOD_PATH = "../data/splits/ood_smiles.csv"
OUTPUT_DIR = "../results"

N_BITS = 2048
RADIUS = 2
MAX_ID = 3000     # limit for speed
MAX_OOD = 3000

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)

print("Loading data...")
id_df = pd.read_csv(ID_PATH).sample(n=MAX_ID, random_state=42)
ood_df = pd.read_csv(OOD_PATH).sample(n=MAX_OOD, random_state=42)

print("Featurizing ID molecules...")
id_fps = [smiles_to_fp(smi) for smi in id_df["smiles"]]
id_fps = [fp for fp in id_fps if fp is not None]

def ood_score(fp, id_fps, exclude_self=False):
    sims = DataStructs.BulkTanimotoSimilarity(fp, id_fps)
    if exclude_self:
        sims = [s for s in sims if s < 0.999]  # remove self-match
    return 1.0 - max(sims)

print("Scoring ID molecules...")
id_scores = []
for smi in id_df["smiles"]:
    fp = smiles_to_fp(smi)
    if fp:
        id_scores.append(ood_score(fp, id_fps, exclude_self=True))

print("Scoring OOD molecules...")
ood_scores = []
for smi in ood_df["smiles"]:
    fp = smiles_to_fp(smi)
    if fp:
        ood_scores.append(ood_score(fp, id_fps))

y_true = np.array([0]*len(id_scores) + [1]*len(ood_scores))
y_scores = np.array(id_scores + ood_scores)

auc = roc_auc_score(y_true, y_scores)

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.hist(id_scores, bins=50, alpha=0.6, label="ID")
plt.hist(ood_scores, bins=50, alpha=0.6, label="OOD")
plt.legend()
plt.xlabel("OOD score (1 - max Tanimoto similarity)")
plt.ylabel("Count")
plt.title(f"Tanimoto-based OOD (AUC = {auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tanimoto_ood_hist.png"))
print("\nSanity check: mean max similarity")
print("Mean max similarity (ID):", 1 - np.mean(id_scores))
print("Mean max similarity (OOD):", 1 - np.mean(ood_scores))

print(f"ROC-AUC (Tanimoto): {auc:.3f}")
print("Tanimoto OOD evaluation completed.")
