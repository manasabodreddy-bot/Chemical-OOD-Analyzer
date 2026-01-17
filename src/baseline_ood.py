from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ID_PATH = "../data/splits/id_smiles.csv"
OOD_PATH = "../data/splits/ood_smiles.csv"
OUTPUT_DIR = "../results"

N_BITS = 2048
RADIUS = 2
MAX_ID_SAMPLES = 20000  # keep runtime reasonable

def featurize(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, RADIUS, nBits=N_BITS
            )
            fps.append(np.array(fp))
    return np.array(fps)

print("Loading data...")
id_df = pd.read_csv(ID_PATH)
ood_df = pd.read_csv(OOD_PATH)

id_smiles = id_df["smiles"].sample(
    n=min(MAX_ID_SAMPLES, len(id_df)), random_state=42
)
ood_smiles = ood_df["smiles"].sample(
    n=min(MAX_ID_SAMPLES, len(ood_df)), random_state=42
)

print("Featurizing ID molecules...")
X_id = featurize(id_smiles)

print("Featurizing OOD molecules...")
X_ood = featurize(ood_smiles)

print("Computing centroid...")
centroid = X_id.mean(axis=0)

def distance(x):
    return np.linalg.norm(x - centroid)

id_scores = np.array([distance(x) for x in X_id])
ood_scores = np.array([distance(x) for x in X_ood])

y_true = np.concatenate([
    np.zeros(len(id_scores)),
    np.ones(len(ood_scores))
])
y_scores = np.concatenate([id_scores, ood_scores])

auc = roc_auc_score(y_true, y_scores)

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.hist(id_scores, bins=50, alpha=0.6, label="ID")
plt.hist(ood_scores, bins=50, alpha=0.6, label="OOD")
plt.legend()
plt.title(f"OOD Score Distribution (AUC = {auc:.3f})")
plt.xlabel("Distance to ID centroid")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ood_score_hist.png"))

print(f"ROC-AUC: {auc:.3f}")
print("Baseline OOD evaluation completed.")
