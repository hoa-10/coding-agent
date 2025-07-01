import pandas as pd
import numpy as np
import os
import json
import random
import argparse
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # New import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Configuration ---
DATA_FILE = 'sensor_data.csv'
# RESULTS_DIR and RESULTS_FILE will be set by command-line arguments

# Sensor columns to use
SENSOR_COLUMNS = ['sensor_1', 'sensor_2', 'sensor_3']

# Anomaly Generation Parameters (Base values)
ANOMALY_PERCENTAGE_RANGE = (0.05, 0.10) # 5% to 10% of data points
PERTURBATION_FACTOR_STD_BASE = (5, 10) # Multiplier for standard deviation for anomaly perturbation

# Windowing Parameters (Base values)
WINDOW_SIZE_BASE = 60
OVERLAP_BASE = 30

# Model Parameters (Base values)
RANDOM_FOREST_HYPERPARAMETERS_BASE = {
    'n_estimators': 100,
    'random_state': 42
}

LOGISTIC_REGRESSION_HYPERPARAMETERS_BASE = {
    'random_state': 42,
    'solver': 'liblinear' # A good default solver for small datasets
}

# Data Split Ratios
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1 # This will be 1 - TRAIN_RATIO - VALIDATION_RATIO

# Define experiment configurations
# Each entry corresponds to a run (index 0 for run 1, index 1 for run 2, etc.)
EXPERIMENT_CONFIGS = [
    # Run 1: Investigate Window Size (Smaller Window)
    {
        'WINDOW_SIZE': 30,
        'OVERLAP': 15, # Adjusted for new window size
        'MODEL_TYPE': 'RandomForest',
        'MODEL_HYPERPARAMETERS': RANDOM_FOREST_HYPERPARAMETERS_BASE,
        'PERTURBATION_FACTOR_STD': PERTURBATION_FACTOR_STD_BASE,
    },
    # Run 2: Investigate Overlap (Increased Overlap with Baseline Window Size)
    # Goal: Assess the impact of increased window overlap on performance.
    # Parameters: WINDOW_SIZE=60 (baseline), OVERLAP=45 (increased from 30), others baseline.
    {
        'WINDOW_SIZE': WINDOW_SIZE_BASE,
        'OVERLAP': 45,
        'MODEL_TYPE': 'RandomForest',
        'MODEL_HYPERPARAMETERS': RANDOM_FOREST_HYPERPARAMETERS_BASE,
        'PERTURBATION_FACTOR_STD': PERTURBATION_FACTOR_STD_BASE,
    },
    # Run 3: Investigate n_estimators (Increased Trees for Random Forest)
    # Goal: Determine if increasing the number of trees in the Random Forest improves detection.
    # Parameters: n_estimators=200 (increased from 100), others baseline.
    {
        'WINDOW_SIZE': WINDOW_SIZE_BASE,
        'OVERLAP': OVERLAP_BASE,
        'MODEL_TYPE': 'RandomForest',
        'MODEL_HYPERPARAMETERS': {'n_estimators': 200, 'random_state': 42},
        'PERTURBATION_FACTOR_STD': PERTURBATION_FACTOR_STD_BASE,
    },
    # Run 4: Investigate Anomaly Perturbation Factor (Larger Anomaly Magnitudes)
    # Goal: Examine how larger anomaly magnitudes affect the classifier's ability to detect them.
    # Parameters: PERTURBATION_FACTOR_STD=(10, 15) (increased from (5, 10)), others baseline.
    {
        'WINDOW_SIZE': WINDOW_SIZE_BASE,
        'OVERLAP': OVERLAP_BASE,
        'MODEL_TYPE': 'RandomForest',
        'MODEL_HYPERPARAMETERS': RANDOM_FOREST_HYPERPARAMETERS_BASE,
        'PERTURBATION_FACTOR_STD': (10, 15),
    },
    # Run 5: Investigate Logistic Regression Classifier
    # Goal: Evaluate a simpler classifier (Logistic Regression) to see if it can also achieve high performance.
    # Parameters: Model=LogisticRegression, others baseline.
    {
        'WINDOW_SIZE': WINDOW_SIZE_BASE,
        'OVERLAP': OVERLAP_BASE,
        'MODEL_TYPE': 'LogisticRegression',
        'MODEL_HYPERPARAMETERS': LOGISTIC_REGRESSION_HYPERPARAMETERS_BASE,
        'PERTURBATION_FACTOR_STD': PERTURBATION_FACTOR_STD_BASE,
    }
]

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run sensor anomaly detection experiment.")
parser.add_argument('--out_dir', type=str, required=True,
                    help="Output directory for results (e.g., 'run_1').")
args = parser.parse_args()

# Extract run number from out_dir
match = re.match(r'run_(\d+)', args.out_dir)
if not match:
    raise ValueError(f"Invalid --out_dir format: {args.out_dir}. Expected 'run_X'.")
run_number = int(match.group(1))

# Set results directory and file
RESULTS_DIR = args.out_dir
RESULTS_FILE = os.path.join(RESULTS_DIR, 'final_info.json')

# Apply current run's configuration
if run_number <= 0 or run_number > len(EXPERIMENT_CONFIGS):
    raise ValueError(f"Run number {run_number} is out of bounds. Max runs: {len(EXPERIMENT_CONFIGS)}")

current_config = EXPERIMENT_CONFIGS[run_number - 1] # Adjust for 0-based indexing

# Overwrite global parameters with current run's configuration
WINDOW_SIZE = current_config['WINDOW_SIZE']
OVERLAP = current_config['OVERLAP']
MODEL_TYPE = current_config['MODEL_TYPE'] # New parameter
MODEL_HYPERPARAMETERS = current_config['MODEL_HYPERPARAMETERS'] # New parameter
PERTURBATION_FACTOR_STD = current_config['PERTURBATION_FACTOR_STD']

print(f"Running experiment for Run {run_number} with configuration:")
print(f"  WINDOW_SIZE: {WINDOW_SIZE}")
print(f"  OVERLAP: {OVERLAP}")
print(f"  MODEL_TYPE: {MODEL_TYPE}") # New print
print(f"  MODEL_HYPERPARAMETERS: {MODEL_HYPERPARAMETERS}") # New print
print(f"  PERTURBATION_FACTOR_STD: {PERTURBATION_FACTOR_STD}")

# --- Dummy Data Generation (for script usability) ---
if not os.path.exists(DATA_FILE):
    print(f"'{DATA_FILE}' not found. Generating dummy data...")
    num_data_points = 2000
    np.random.seed(42) # for reproducibility of dummy data
    
    # Simulate some sensor data with trends and noise
    time = np.arange(num_data_points)
    
    sensor_1 = 100 + 0.1 * time + np.random.normal(0, 5, num_data_points) + 10 * np.sin(time / 50)
    sensor_2 = 50 + 0.05 * time + np.random.normal(0, 3, num_data_points) + 5 * np.cos(time / 30)
    sensor_3 = 200 - 0.08 * time + np.random.normal(0, 7, num_data_points) + 15 * np.sin(time / 70)
    
    dummy_df = pd.DataFrame({
        'sensor_1': sensor_1,
        'sensor_2': sensor_2,
        'sensor_3': sensor_3
    })
    dummy_df.to_csv(DATA_FILE, index=False)
    print(f"Dummy data saved to '{DATA_FILE}'.")
else:
    print(f"'{DATA_FILE}' found. Using existing data.")

# --- Main Pipeline ---

# 1. Data Loading
df = pd.read_csv(DATA_FILE)
print(f"Loaded data with {len(df)} rows.")

# 2. Preprocessing Steps & Feature Engineering

# 2.1 Synthetic Anomaly Generation
df['is_anomaly'] = 0
num_anomalies_to_generate = int(df.shape[0] * random.uniform(*ANOMALY_PERCENTAGE_RANGE))

# Select unique random indices for anomalies
anomaly_indices = np.random.choice(df.index, size=num_anomalies_to_generate, replace=False)
df.loc[anomaly_indices, 'is_anomaly'] = 1

# Perturb sensor values for anomalous points
for col in SENSOR_COLUMNS:
    std_dev = df[col].std()
    for idx in anomaly_indices:
        perturbation_magnitude = random.uniform(*PERTURBATION_FACTOR_STD) * std_dev
        if random.random() < 0.5: # 50% chance to add or subtract
            df.loc[idx, col] += perturbation_magnitude
        else:
            df.loc[idx, col] -= perturbation_magnitude
print(f"Synthetically introduced {num_anomalies_to_generate} anomalies.")
print(f"Anomaly ratio: {df['is_anomaly'].sum() / len(df):.2%}")

# 2.2 Window Creation
window_features = []
window_labels = []
step_size = WINDOW_SIZE - OVERLAP

for i in range(0, len(df) - WINDOW_SIZE + 1, step_size):
    window_df = df.iloc[i : i + WINDOW_SIZE]
    
    # Flatten sensor readings for the window
    features = window_df[SENSOR_COLUMNS].values.flatten()
    window_features.append(features)
    
    # Determine window label: 1 if any point in window is anomalous, else 0
    label = int(window_df['is_anomaly'].any())
    window_labels.append(label)

X = np.array(window_features)
y = np.array(window_labels)

print(f"Created {len(X)} windows.")
print(f"Window features shape: {X.shape}")
print(f"Window labels shape: {y.shape}")
print(f"Anomalous windows: {np.sum(y)} ({np.sum(y) / len(y):.2%})")

# 2.3 Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Window features scaled.")

# 3. Data Splitting (Chronological)
num_samples = len(X_scaled)
train_end_idx = int(num_samples * TRAIN_RATIO)
validation_end_idx = int(num_samples * (TRAIN_RATIO + VALIDATION_RATIO))

X_train, y_train = X_scaled[:train_end_idx], y[:train_end_idx]
X_val, y_val = X_scaled[train_end_idx:validation_end_idx], y[train_end_idx:validation_end_idx]
X_test, y_test = X_scaled[validation_end_idx:], y[validation_end_idx:]

print(f"Data split: Train={len(X_train)} Validation={len(X_val)} Test={len(X_test)} samples.")

# 4. Model Configuration & Training Process
model = None
if MODEL_TYPE == 'RandomForest':
    model = RandomForestClassifier(**MODEL_HYPERPARAMETERS)
    print(f"Training RandomForestClassifier with hyperparameters: {MODEL_HYPERPARAMETERS}")
elif MODEL_TYPE == 'LogisticRegression':
    model = LogisticRegression(**MODEL_HYPERPARAMETERS)
    print(f"Training LogisticRegression with hyperparameters: {MODEL_HYPERPARAMETERS}")
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

model.fit(X_train, y_train)
print("Model training complete.")

# 5. Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0) # zero_division=0 to handle cases where there are no positive predictions/actuals
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n--- Evaluation on Test Set ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 6. Save Results
os.makedirs(RESULTS_DIR, exist_ok=True)

results = {
    'evaluation_metrics': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    },
    'model_type': MODEL_TYPE, # New field
    'model_hyperparameters': MODEL_HYPERPARAMETERS,
    'pipeline_parameters': {
        'window_size': WINDOW_SIZE,
        'overlap': OVERLAP,
        'anomaly_perturbation_factor_std': PERTURBATION_FACTOR_STD,
        'data_split_ratios': {
            'train': TRAIN_RATIO,
            'validation': VALIDATION_RATIO,
            'test': TEST_RATIO
        }
    }
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to '{RESULTS_FILE}'.")
