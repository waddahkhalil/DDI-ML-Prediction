import pandas as pd

# Load dataset
dataset_file = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/balanced_ddi_dataset_with_smiles.csv"
df = pd.read_csv(dataset_file)

# Sample 50,000 rows (stratified sampling to keep class balance)
df_sampled = df.groupby("Label").apply(lambda x: x.sample(n=25000, random_state=42)).reset_index(drop=True)

# Save the reduced dataset
reduced_file = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/reduced_ddi_dataset.csv"
df_sampled.to_csv(reduced_file, index=False)

print(f"Reduced dataset saved as {reduced_file} with {len(df_sampled)} samples")
