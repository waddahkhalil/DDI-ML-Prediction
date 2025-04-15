import numpy as np
import pandas as pd

df = pd.read_csv("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/balanced_ddi_dataset_with_smiles.csv")


# Define the number of samples needed per class
subset_size = 5000  # 5000 samples for each class (0 and 1)

# Separate the dataset based on labels
df_positive = df[df['Label'] == 1].sample(n=subset_size, random_state=42)
df_negative = df[df['Label'] == 0].sample(n=subset_size, random_state=42)


# Concatenate to form a balanced subset
df_subset = pd.concat([df_positive, df_negative]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save the subset
subset_path = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/ddi_subset_10k.csv"
df_subset.to_csv(subset_path, index=False)

# Display subset info
df_subset.info(), df_subset.head()












