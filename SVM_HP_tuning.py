import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt 


# Step 1: Load Features & Labels
X = np.load("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/X_features_subset.npy")  
y = np.load("D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/DT/y_labels_subset.npy")  

# Step 2: Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Standardize Features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Data Loaded & Preprocessed: Training Samples: {X_train.shape}, Testing Samples: {X_test.shape}")

### SVM Hyperparameter Tuning ###
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm_model = SVC(probability=True, random_state=42)
svm_search = RandomizedSearchCV(svm_model, param_distributions=svm_params, n_iter=5, cv=3, 
                                verbose=1, random_state=42, n_jobs=-1, scoring='roc_auc')
svm_search.fit(X_train, y_train)
svm_best = svm_search.best_estimator_

print("\nBest SVM Parameters:", svm_search.best_params_)