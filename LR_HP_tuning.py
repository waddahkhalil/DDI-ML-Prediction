import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
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


### Logistic Regression Hyperparameter Tuning ###
log_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}

log_model = LogisticRegression(random_state=42, max_iter=500)
log_search = RandomizedSearchCV(log_model, param_distributions=log_params, n_iter=5, cv=3, 
                                verbose=1, random_state=42, n_jobs=-1, scoring='roc_auc')
log_search.fit(X_train, y_train)
log_best = log_search.best_estimator_

print("\nBest Logistic Regression Parameters:", log_search.best_params_)