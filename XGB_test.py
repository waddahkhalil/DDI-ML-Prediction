import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt 



# Step 1: Load Features & Labels
X = np.load("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/X_features.npy")  # 4096-dimensional molecular fingerprint vectors
y = np.load("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/y_labels.npy")  # 1 = Positive DDI, 0 = Negative DDI

# Step 2: Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Train XGBoost Model
xgb_model = XGBClassifier(
    objective='binary:logistic',  # Binary classification task
    eval_metric='logloss',  # Log loss function
    use_label_encoder=False,
    n_estimators=100,  # Number of trees (increase for better performance)
    learning_rate=0.1,  # Step size shrinkage
    max_depth=6,  # Maximum depth of trees
    subsample=0.8,  # Subsample ratio to prevent overfitting
    colsample_bytree=0.8,  # Fraction of features used per tree
    random_state=42
)

xgb_model.fit(X_train, y_train)


# Predict on Test Data
y_pred = xgb_model.predict(X_test)

# Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", round(accuracy, 4))
print("Classification Report:\n", classification_report(y_test, y_pred))

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