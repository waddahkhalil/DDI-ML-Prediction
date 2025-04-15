import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load Dataset
df = pd.read_csv("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/reduced_ddi_dataset.csv")

# Step 2: Function to Convert SMILES to Molecular Fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to a Morgan Fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    else:
        return [0] * n_bits  # Return zero vector if invalid SMILES

# Step 3: Compute Fingerprints for Both Drugs
df["Drug_Fingerprint"] = df["Drug SMILES"].apply(lambda x: smiles_to_fingerprint(str(x)))
df["Interacting_Drug_Fingerprint"] = df["Interacting Drug SMILES"].apply(lambda x: smiles_to_fingerprint(str(x)))

# Step 4: Convert Fingerprints to NumPy Arrays
X1 = np.array(df["Drug_Fingerprint"].tolist())
X2 = np.array(df["Interacting_Drug_Fingerprint"].tolist())

# Step 5: Concatenate Both Drugs' Features (Final Input Features)
X = np.hstack((X1, X2))

# Step 6: Standardize Features for Better ML Performance
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Prepare Labels for ML Training
y = df["Label"].values  # 1 = Positive DDI, 0 = Negative DDI

# Step 8: Save Preprocessed Data
np.save("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/X_features.npy", X_scaled)
np.save("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/y_labels.npy", y)

print(f"Preprocessing complete! Feature shape: {X_scaled.shape}")
