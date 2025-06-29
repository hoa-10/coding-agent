import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def generate_sensor_data_csv(filename="sensor_data.csv"):
    num_rows = 1440
    data = pd.DataFrame({
        'sensor_1': np.sin(np.linspace(0, 100, num_rows)) * 10 + np.random.randn(num_rows) * 0.5,
        'sensor_2': np.cos(np.linspace(0, 80, num_rows)) * 8 + np.random.randn(num_rows) * 0.7,
        'sensor_3': np.random.randn(num_rows) * 2 + 5
    })
    for _ in range(10):
        idx = np.random.randint(50, num_rows - 50)
        sensor_idx = np.random.randint(0, 3)
        magnitude = np.random.uniform(10, 20)
        data.iloc[idx : idx + 5, sensor_idx] += np.random.choice([-1, 1]) * magnitude
    data.to_csv(filename, index=False)

def generate_synthetic_labels(data_df):
    iso_forest = IsolationForest(random_state=42, contamination='auto')
    iso_forest.fit(data_df)
    predictions = iso_forest.predict(data_df)
    labels = np.array([1 if p == -1 else 0 for p in predictions])
    return labels

def create_windows(data, labels, window_size, stride):
    X_windows = []
    y_windows = []
    
    for i in range(0, len(data) - window_size + 1, stride):
        window_data = data[i : i + window_size]
        X_windows.append(window_data.flatten())
        y_windows.append(labels[i + window_size - 1])

    return np.array(X_windows), np.array(y_windows)

def temporal_split(X, y, train_ratio, val_ratio, test_ratio):
    total_size = len(X)
    train_end = int(total_size * train_ratio)
    val_end = int(total_size * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(y_true, y_pred, set_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    print(f"\n--- {set_name.capitalize()} Set Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Anomaly): {precision:.4f}")
    print(f"Recall (Anomaly): {recall:.4f}")
    print(f"F1-score (Anomaly): {f1:.4f}")

    return {
        f"{set_name}_accuracy": accuracy,
        f"{set_name}_precision": precision,
        f"{set_name}_recall": recall,
        f"{set_name}_f1": f1,
    }

if __name__ == "__main__":
    sensor_data_file = "sensor_data.csv"
    if not os.path.exists(sensor_data_file):
        generate_sensor_data_csv(sensor_data_file)
        print(f"Generated dummy data: {sensor_data_file}")
    else:
        print(f"Using existing data: {sensor_data_file}")

    df = pd.read_csv(sensor_data_file)
    print(f"Data loaded. Raw shape: {df.shape}")

    y_raw_labels = generate_synthetic_labels(df)
    print(f"Synthetic labels generated. Anomaly count in raw data: {np.sum(y_raw_labels)}")

    WINDOW_SIZE = 10
    STRIDE = 1
    X_windowed, y_windowed = create_windows(df.values, y_raw_labels, WINDOW_SIZE, STRIDE)
    print(f"Windowed data shape: X={X_windowed.shape}, y={y_windowed.shape}")

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
        X_windowed, y_windowed, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    print(f"Train split shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation split shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test split shapes: X={X_test.shape}, y={y_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled using StandardScaler.")

    MODEL_PARAMS = {
        "model_name": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42
    }
    model = RandomForestClassifier(
        n_estimators=MODEL_PARAMS["n_estimators"],
        random_state=MODEL_PARAMS["random_state"]
    )
    print(f"Training {MODEL_PARAMS['model_name']} model...")
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    results = {}

    y_val_pred = model.predict(X_val_scaled)
    val_metrics = evaluate_model(y_val, y_val_pred, "val")
    results.update(val_metrics)

    y_test_pred = model.predict(X_test_scaled)
    test_metrics = evaluate_model(y_test, y_test_pred, "test")
    results.update(test_metrics)

    results["pipeline_parameters"] = {
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO
    }
    results["model_parameters"] = MODEL_PARAMS

    RESULTS_DIR = "result"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")

    print("\nPipeline finished successfully.")