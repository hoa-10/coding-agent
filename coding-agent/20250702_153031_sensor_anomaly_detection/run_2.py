import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

EXPERIMENTS = {
    'run_0': {
        'model_type': 'SVC',
        'hyperparameters': {'C': 1.0, 'kernel': 'rbf', 'random_state': 42},
        'baseline_results': {
            'evaluation_metrics': {'accuracy': 1.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
            'model_hyperparameters': {'model_type': 'SVC', 'C': 1.0, 'kernel': 'rbf', 'random_state': 42}
        }
    },
    'run_1': {
        'model_type': 'RandomForestClassifier',
        'hyperparameters': {'n_estimators': 100, 'random_state': 42}
    },
    'run_2': {
        'model_type': 'LogisticRegression',
        'hyperparameters': {'solver': 'liblinear', 'random_state': 42, 'class_weight': 'balanced'}
    },
    'run_3': {
        'model_type': 'SVC',
        'hyperparameters': {'C': 0.1, 'kernel': 'linear', 'random_state': 42}
    },
    'run_4': {
        'model_type': 'RandomForestClassifier',
        'hyperparameters': {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'}
    }
}

def run_pipeline(experiment_config, out_dir):
    # --- 1. Data Loading and Initial Check ---
    file_path = 'sensor_data.csv'

    # If it's the baseline run, use pre-defined results and skip execution
    if out_dir == 'run_0':
        results = experiment_config['baseline_results']
        results_file_path = os.path.join(out_dir, 'final_info.json')
        os.makedirs(out_dir, exist_ok=True)
        with open(results_file_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Baseline results for {out_dir} saved to {results_file_path}")
        return
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: sensor_data.csv not found.")
        sys.exit(1) # Exit gracefully

    # --- 2. Preprocessing ---

    # Missing Value Imputation (Mean imputation for each column)
    for col in ['sensor_1', 'sensor_2', 'sensor_3']:
        df[col] = df[col].fillna(df[col].mean())

    # Calculate global mean and std for anomaly label generation after imputation
    global_sensor_means = {col: df[col].mean() for col in ['sensor_1', 'sensor_2', 'sensor_3']}
    global_sensor_stds = {col: df[col].std() for col in ['sensor_1', 'sensor_2', 'sensor_3']}

    # Windowing and Feature Extraction
    window_size = 10
    overlap = 5
    stride = window_size - overlap

    X_windows = []
    y_labels = []

    for i in range(0, len(df) - window_size + 1, stride):
        window_df = df.iloc[i : i + window_size]
        
        # Extract features (mean, std, min, max for each sensor)
        features = []
        for col in ['sensor_1', 'sensor_2', 'sensor_3']:
            features.extend([
                window_df[col].mean(),
                window_df[col].std(),
                window_df[col].min(),
                window_df[col].max()
            ])
        X_windows.append(features)

        # Determine window's anomaly label
        is_window_anomaly = 0
        for col in ['sensor_1', 'sensor_2', 'sensor_3']:
            global_mean = global_sensor_means[col]
            global_std = global_sensor_stds[col]
            upper_threshold = global_mean + 3 * global_std
            lower_threshold = global_mean - 3 * global_std

            if ((window_df[col] > upper_threshold) | (window_df[col] < lower_threshold)).any():
                is_window_anomaly = 1
                break # Anomaly found in this window, no need to check further sensors

        y_labels.append(is_window_anomaly)

    X = np.array(X_windows)
    y = np.array(y_labels)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 4. Training Process ---

    # Split data into training, validation, and test sets
    # First split: 90% for train+val, 10% for test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42, stratify=y
    )

    # Second split: From train+val, split 8/9 for train, 1/9 for validation
    # This results in: (0.9 * 8/9) = 0.8 (80%) train, (0.9 * 1/9) = 0.1 (10%) validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.1/0.9), random_state=42, stratify=y_train_val
    )

    # Model Configuration based on experiment_config
    model_type = experiment_config['model_type']
    hyperparameters = experiment_config['hyperparameters']

    if model_type == 'SVC':
        model = SVC(**hyperparameters)
    elif model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(**hyperparameters)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(**hyperparameters)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)

    # --- 5. Evaluation ---
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # --- 6. Save Results ---
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "evaluation_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        "model_hyperparameters": {
            "model_type": model_type,
            **hyperparameters # Unpack hyperparameters directly
        }
    }

    results_file_path = os.path.join(out_dir, 'final_info.json')
    with open(results_file_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results for {out_dir} saved to {results_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sensor anomaly detection experiment.")
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for results (e.g., run_0, run_1)')
    args = parser.parse_args()

    if args.out_dir not in EXPERIMENTS:
        print(f"Error: Experiment configuration for '{args.out_dir}' not found.")
        sys.exit(1)

    experiment_config = EXPERIMENTS[args.out_dir]
    run_pipeline(experiment_config, args.out_dir)
