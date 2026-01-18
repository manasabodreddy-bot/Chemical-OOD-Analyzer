Chemical OOD Detection under Scaffold Shift
Overview

This repository contains an exploratory study on out-of-distribution (OOD) detection for small molecules using the ZINC dataset.
The focus is on a particularly challenging setting: scaffold-based distribution shift, where in-distribution (ID) and OOD molecules share no chemical scaffolds.

The project was carried out to understand how far simple and commonly used molecular representations can go under such constraints, and where they clearly start to fail.

Motivation

In many molecular machine learning studies, models are evaluated using random train–test splits.
While convenient, these splits often allow significant structural overlap between training and test molecules.

In practical drug discovery settings, however, models are frequently asked to make predictions on entirely new chemical scaffolds.
Performance under these conditions is often much worse than under random splits.

The aim of this project is therefore not to maximize performance, but to evaluate and understand baseline behavior under a strict scaffold shift.

Dataset and split

Dataset: ZINC (accessed via DeepChem)

Input format: SMILES

Scaffold definition: Bemis–Murcko scaffolds (RDKit)

Molecules were grouped by scaffold and split such that no scaffold appears in both ID and OOD sets.

Scaffold statistics

Unique ID scaffolds: 111,013

Unique OOD scaffolds: 27,754

Shared scaffolds: 0

This confirms that the split is strictly scaffold-disjoint and does not suffer from structural leakage.

Methods
Data preparation

Extraction of valid SMILES from ZINC

Removal of invalid molecules

RDKit-based canonicalization

OOD setup

ID vs OOD treated as a binary classification problem

OOD scores interpreted as “degree of novelty” relative to ID data

Baselines evaluated

1. Random baseline
A random score assigned to each molecule, used only as a sanity check.

2. Similarity-based baseline

Morgan fingerprints (radius = 2)

Maximum Tanimoto similarity to the ID set

Care taken to avoid self-matching, which initially led to artificially inflated scores

3. Logistic regression baseline

Input: Morgan fingerprints (2048 bits)

Task: ID vs OOD classification

Balanced sampling and stratified train–test split

Metric: ROC-AUC

Results
Method	ROC-AUC
Random	~0.50
Tanimoto similarity (corrected)	~0.54
Logistic regression	0.577

Overall performance remains close to chance, even for chemically informed representations.

This is not unexpected given the absence of shared scaffolds and the limited expressivity of the models used.

Observations

Initial similarity-based results showed near-perfect performance, which was traced back to evaluation artifacts caused by self-similarity.

After correcting for this, performance dropped substantially, highlighting how fragile naive OOD evaluations can be.

Logistic regression offers a small improvement over similarity heuristics, but still struggles to generalize across unseen scaffolds.

These results suggest that scaffold-based OOD detection is inherently difficult, especially when relying on simple fingerprint-based methods.

Limitations

No property or activity labels were considered

Only simple baselines were evaluated

Deep learning models were intentionally excluded to keep the analysis interpretable and focused

Possible extensions

Graph-based molecular representations

Task-aware OOD detection

Density-based or uncertainty-aware approaches

Evaluation on standard MoleculeNet benchmarks

Reproducibility
Environment

Python 3.9+

RDKit

DeepChem

scikit-learn

PyTorch (CPU)

Install dependencies:

pip install -r requirements.txt

Script order
python src/extract_smiles.py
python src/scaffold_split.py
python src/baseline_ood.py
python src/tanimoto_ood.py
python src/logreg_ood.py
python src/scaffold_stats.py

Summary

This project shows that even simple, widely used molecular representations struggle to distinguish in-distribution molecules from structurally novel ones when evaluated under strict scaffold shift.
Rather than presenting strong performance, the results highlight limitations and failure modes that are often obscured by less rigorous evaluation settings.

Author

Manasa Reddy
(Prepared as part of a PhD application portfolio)