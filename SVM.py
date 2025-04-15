import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt 


# Step 1: Load Features & Labels
X = np.load("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/X_features.npy")  # 4096-dimensional molecular fingerprint vectors
y = np.load("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/y_labels.npy")  # 1 = Positive DDI, 0 = Negative DDI

# Step 2: Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Standardize Features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Data Loaded & Preprocessed: Training Samples: {X_train.shape}, Testing Samples: {X_test.shape}")

# SVM Train model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# SVM Predict
y_pred = svm_model.predict(X_test)

# SVM Evaluate
print("SVM Results")
print(classification_report(y_test, y_pred))

auc_roc = roc_auc_score(y_test, y_pred) # AUC-ROC score 
print(f"AUC-ROC: {auc_roc}")

# plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred) #fpr = false positive rate, tpr = true positive rate
plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random') #Random chance line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()