import numpy as np
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Dataset Information:
# The dataset is located at `pect_ndt_full_dataset.npz`.
# You *must* use the following exact code snippet to load the data:
dataset_path = 'pect_ndt_full_dataset.npz'
if not os.path.exists(dataset_path):
    print(f"Error: Dataset file not found at {dataset_path}")
    sys.exit(1)

data = np.load(dataset_path)
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

# 1. Data Preparation and Splitting
X_combined = np.concatenate((X_train, X_valid), axis=0)
y_combined = np.concatenate((y_train, y_valid), axis=0)

# Stratified split: 80% train, 20% (validation + test)
X_train_combined, X_val_test, y_train_combined, y_val_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# Split the 20% into 10% validation and 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test
)

# Reshape data for StandardScaler (flatten the last two dimensions)
X_train_combined = X_train_combined.reshape(X_train_combined.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Apply StandardScaler to features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 2. Model Configuration
model = Sequential([
    InputLayer(input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Training Process
epochs = 50
batch_size = 32

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train_combined,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping],
    verbose=0 # Suppress verbose output during training
)

# 4. Evaluation
y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision (macro): {precision:.4f}")
print(f"Test Recall (macro): {recall:.4f}")
print(f"Test F1-score (macro): {f1:.4f}")
print(f"Test ROC AUC: {roc_auc:.4f}")

# 5. Save Results
results_dir = 'result'
os.makedirs(results_dir, exist_ok=True)

results_data = {
    "evaluation_metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    },
    "hyperparameters": {
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "adam",
        "loss_function": "binary_crossentropy",
        "model_layers": [
            {"type": "InputLayer", "units": X_train_scaled.shape[1]},
            {"type": "Dense", "units": 64, "activation": "relu"},
            {"type": "Dense", "units": 32, "activation": "relu"},
            {"type": "Dense", "units": 1, "activation": "sigmoid"}
        ],
        "early_stopping_patience": 5,
        "data_split_random_state": 42,
        "tensorflow_seed": 42,
        "numpy_seed": 42
    }
}

results_file_path = os.path.join(results_dir, 'results.json')
with open(results_file_path, 'w') as f:
    json.dump(results_data, f, indent=4)

print(f"\nResults saved to {results_file_path}")
