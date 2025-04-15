import random
import pandas as pd

# Step 1: Load Positive DDI Dataset
positive_ddi_file = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/drug_interactions_with_smiles.csv"
positive_ddi = pd.read_csv(positive_ddi_file)

# Step 2: Extract Unique Drugs and Their SMILES
drug_smiles_dict = {}  # Dictionary to store Drug ID -> SMILES mapping

for _, row in positive_ddi.iterrows():
    drug_smiles_dict[row["Drug ID"]] = row["Drug SMILES"]
    drug_smiles_dict[row["Interacting Drug ID"]] = row["Interacting Drug SMILES"]

# Step 3: Create a Set of Positive DDI Pairs (for fast lookup)
positive_pairs = set(zip(positive_ddi['Drug ID'], positive_ddi['Interacting Drug ID']))

# Step 4: Generate Negative Samples (Non-Interacting Pairs)
all_drugs = list(drug_smiles_dict.keys())  # List of unique Drug IDs
negative_samples = []
num_negatives = len(positive_ddi)  # Generate same number as positive DDIs

while len(negative_samples) < num_negatives:
    drug1, drug2 = random.sample(all_drugs, 2)  # Pick two random drugs
    if (drug1, drug2) not in positive_pairs and (drug2, drug1) not in positive_pairs:
        # Get SMILES for both drugs
        smiles1 = drug_smiles_dict.get(drug1, "N/A")
        smiles2 = drug_smiles_dict.get(drug2, "N/A")
        negative_samples.append((drug1, drug2, smiles1, smiles2, 0))  # Label = 0 for negative

# Step 5: Convert to DataFrame & Save
negative_ddi = pd.DataFrame(negative_samples, columns=["Drug ID", "Interacting Drug ID", "Drug SMILES", "Interacting Drug SMILES", "Label"])
negative_ddi_file = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/negative_drug_interactions_with_smiles.csv"
negative_ddi.to_csv(negative_ddi_file, index=False)

print(f"Successfully generated {len(negative_ddi)} negative DDIs and saved to {negative_ddi_file}")

# Step 6: Merge with Positive DDI to Create a Balanced Dataset
positive_ddi["Label"] = 1  # Mark positive samples

# Ensure same column format for merging
positive_ddi = positive_ddi[["Drug ID", "Interacting Drug ID", "Drug SMILES", "Interacting Drug SMILES", "Label"]]

# Merge Positive and Negative Samples
balanced_dataset = pd.concat([positive_ddi, negative_ddi])

# Shuffle the dataset to mix positive & negative samples
balanced_dataset = balanced_dataset.sample(frac=1).reset_index(drop=True)

# Save the final dataset
balanced_dataset_file = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/balanced_ddi_dataset_with_smiles.csv"
balanced_dataset.to_csv(balanced_dataset_file, index=False)

print(f"Balanced dataset saved as {balanced_dataset_file} with {len(balanced_dataset)} samples")
