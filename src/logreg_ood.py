from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

ID_PATH = "../data/splits/id_smiles.csv"
OOD_PATH = "../data/splits/ood_smiles.csv"
OUTPUT_DIR = "../results"

N_BITS = 2048
RADIUS = 2
MAX_SAMPLES = 20000   # total (ID + OOD)

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

n_per_class = min(len(id_df), len(ood_df), MAX_SAMPLES // 2)

id_df = id_df.sample(n=n_per_class, random_state=42)
ood_df = ood_df.sample(n=n_per_class, random_state=42)

print("Featurizing ID molecules...")
X_id = featurize(id_df["smiles"])
y_id = np.zeros(len(X_id))

print("Featurizing OOD molecules...")
X_ood = featurize(ood_df["smiles"])
y_ood = np.ones(len(X_ood))

X = np.vstack([X_id, X_ood])
y = np.concatenate([y_id, y_ood])

print("Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training Logistic Regression...")
clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

print("Evaluating...")
y_scores = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_scores)

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.hist(y_scores[y_test == 0], bins=50, alpha=0.6, label="ID")
plt.hist(y_scores[y_test == 1], bins=50, alpha=0.6, label="OOD")
plt.legend()
plt.xlabel("Predicted OOD probability")
plt.ylabel("Count")
plt.title(f"Logistic Regression OOD (AUC = {auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logreg_ood_hist.png"))

print(f"ROC-AUC (LogReg): {auc:.3f}")
print("Logistic Regression OOD evaluation completed.")
