import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import sys
import json

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATASET_PATH = 'pect_ndt_full_dataset.npz'
RESULT_DIR = 'result'

try:
    data = np.load(DATASET_PATH)
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_valid']
    y_valid = data['y_valid']
    X_scan = data['X_scan']
    X_in_corr = data['X_in_corr']
    Xc = data['Xc']
    Xg = data['Xg']
    m = data['m']
    st = data['st']
    print(f"Dataset '{DATASET_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset file '{DATASET_PATH}' not found. Please ensure the file is in the correct directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    sys.exit(1)

y_train = y_train.astype(int)
y_valid = y_valid.astype(int)

# Reshape X_train and X_valid to 2D arrays (n_samples, n_features)
# The RandomForestClassifier expects input data to be 2-dimensional.
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)

print("Skipping data normalisation as requested.")

# Random Forest Hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH = 10

model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_SEED, n_jobs=-1)

print("Starting model training (Random Forest)...")
model.fit(X_train, y_train)
print("Model training completed.")

print("Evaluating model performance on validation set...")
y_pred_proba = model.predict_proba(X_valid)[:, 1] # Probability of the positive class
y_pred = model.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
roc_auc = roc_auc_score(y_valid, y_pred_proba)

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print("------------------------------")

os.makedirs(RESULT_DIR, exist_ok=True)
print(f"Results directory '{RESULT_DIR}' ensured.")

results_data = {
    "model_architecture_description": "Random Forest Classifier for binary classification.",
    "hyperparameters": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "random_seed": RANDOM_SEED
    },
    "evaluation_metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc_score": roc_auc
    }
}

json_output_path = os.path.join(RESULT_DIR, 'results.json')
with open(json_output_path, 'w') as f:
    json.dump(results_data, f, indent=4)
print(f"Evaluation metrics and hyperparameters saved to '{json_output_path}'.")

print("\nDeep learning pipeline completed successfully!")
