# DDI-ML-Prediction
# DDI-Prediction-ML-Project

This project applies various machine learning models to predict Drug-Drug Interactions (DDIs) using features extracted from SMILES representations of drugs provided in the DrugBank database.

## 📁 Project Overview

The project involves the following steps:
1. **Data extraction from DrugBank XML**
2. **Negative sample generation**
3. **Feature engineering using Morgan fingerprints**
4. **Dataset balancing and subsampling**
5. **Model training and hyperparameter tuning**
6. **Model evaluation and performance reporting**

## 📜 Script Descriptions

### `new_script_for_ddi_extract_v4_optimized_specific_drugs.py`
Parses the DrugBank XML file to extract approved small-molecule drugs and their interactions, including SMILES and metadata. Saves the positive interaction dataset as CSV.

### `balanced_ddi_dataset.py`
Generates random non-interacting (negative) drug pairs, combines them with positive interactions, and creates a balanced DDI dataset for ML training.

### `generating_subset_10k.py`
Creates a smaller subset (e.g., 10,000 samples) from the full dataset for quicker training and hyperparameter tuning on limited-resource machines.

### `SMILES_Preprocessing.py`
Converts SMILES strings into 2048-bit Morgan molecular fingerprints using RDKit. Saves the processed features (`X`) and labels (`y`) as `.npy` files.

### `ML_models_build.py`
Builds and evaluates machine learning models (Logistic Regression, Random Forest, SVM, KNN, Naive Bayes, XGBoost) using the processed data. Outputs classification performance metrics and ROC curves.

### `RF_HP_tuning.py`
Performs hyperparameter tuning on the Random Forest classifier using RandomizedSearchCV with cross-validation to optimize predictive performance.

---

## 📦 Requirements

- Python 3.11+
- `rdkit`
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`

Install with:

```bash
pip install -r requirements.txt
