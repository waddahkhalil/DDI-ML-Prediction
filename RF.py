import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
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


# RF Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# RF Predict
y_pred = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# RF Evaluate
print("Random Forest Results")
print(classification_report(y_test, y_pred))

auc_roc = roc_auc_score(y_test, y_pred) # AUC-ROC score 
print(f"AUC-ROC: {auc_roc}")

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot(cmap=plt.cm.Blues)
plt.title('Random Forest Confusion Matrix')
plt.show()

# plot the ROC curve
plt.figure(figsize=(10,8))

# Random Forest
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_rf)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Machine Learning Models')
plt.legend()
plt.grid()
plt.show()