import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 


# Step 1: Load Features & Labels
X = np.load("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/X_features_subset.npy")  # 4096-dimensional molecular fingerprint vectors
y = np.load("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/y_labels_subset.npy")  # 1 = Positive DDI, 0 = Negative DDI

# Step 2: Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Standardize Features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Data Loaded & Preprocessed: Training Samples: {X_train.shape}, Testing Samples: {X_test.shape}")

### Random Forest Hyperparameter Tuning ###
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_search = RandomizedSearchCV(rf_model, param_distributions=rf_params, n_iter=10, cv=3, 
                               verbose=1, random_state=42, n_jobs=-1, scoring='roc_auc')
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_

print("\nBest RF Parameters:", rf_search.best_params_)
