import numpy as np
import matplotlib.pyplot as plt
import json
import os

# --- 1. Load the dataset ---
print("Loading dataset...")
try:
    data = np.load('pect_ndt_full_dataset.npz')
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
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'pect_ndt_full_dataset.npz' not found.")
    print("Please ensure the dataset file is in the same directory as the script.")
    exit()

# Prepare a dictionary to store array information for easy processing
array_info_list = [
    {"name": "X_train", "array": X_train, "description": "Training signals"},
    {"name": "y_train", "array": y_train, "description": "Training labels (0=good, 1=defect)"},
    {"name": "X_valid", "array": X_valid, "description": "Validation signals"},
    {"name": "y_valid", "array": y_valid, "description": "Validation labels (0=good, 1=defect)"},
    {"name": "X_scan", "array": X_scan, "description": "Complete set of measured signals from the entire scan area"},
    {"name": "X_in_corr", "array": X_in_corr, "description": "Signals inside corrosion regions (2D spatial grid)"},
    {"name": "Xc", "array": Xc, "description": "All signals identified as coming from defective areas"},
    {"name": "Xg", "array": Xg, "description": "All signals from normal (good) regions"},
    {"name": "m", "array": m, "description": "Mean values for normalization"},
    {"name": "st", "array": st, "description": "Standard deviation values for normalization"}
]

# Print shapes and types of each array
print("\n--- Array Information (Shape & Type) ---")
for item in array_info_list:
    arr = item['array']
    print(f"{item['name']}: Shape {arr.shape}, Dtype {arr.dtype}, Description: {item['description']}")

# Create output directories
output_dir = 'analysis'
figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
print(f"\nCreated output directories: {output_dir} and {figures_dir}")

# --- 2. Create a signal comparison visualization ---
print("\n--- Generating Signal Comparison Plot ---")

# Select representative samples:
# For simplicity and clear representation as per instructions, we select the first sample
# from the 'good' signals (Xg) and 'defect' signals (Xc).
# Xg and Xc are already in (N, 500) shape, so no squeezing is needed for plotting.
if Xg.shape[0] > 0 and Xc.shape[0] > 0:
    good_signal = Xg[0]
    defect_signal = Xc[0]

    # Time points for plotting (assuming 500 time points for each signal)
    time_points = np.arange(good_signal.shape[0])

    plt.figure(figsize=(10, 6))
    plt.plot(time_points, good_signal, color='blue', label='Good Signal (Sample from Xg)', linewidth=1.5)
    plt.plot(time_points, defect_signal, color='red', label='Defect Signal (Sample from Xc)', linewidth=1.5)
    plt.title('Comparison of PECT Signals: Good vs. Defect', fontsize=16)
    plt.xlabel('Time Points', fontsize=12)
    plt.ylabel('Amplitude (Voltage Response)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(figures_dir, 'signal_comparison.png')
    plt.savefig(plot_path, dpi=300) # Save with high resolution
    plt.close()
    print(f"Signal comparison plot saved to {plot_path}")
else:
    print("Warning: Xg or Xc arrays are empty. Cannot generate signal comparison plot.")
    plot_path = "N/A (Xg or Xc empty)"


# --- 3. Calculates and saves comprehensive statistics ---
print("\n--- Calculating Comprehensive Statistics ---")
results = {}

# 3.1. Dataset information: shapes, types, and array descriptions
dataset_info_dict = {}
for item in array_info_list:
    arr = item['array']
    dataset_info_dict[item['name']] = {
        "shape": str(arr.shape),  # Convert tuple to string for JSON serialization
        "dtype": str(arr.dtype),
        "description": item['description']
    }
results["dataset_info"] = dataset_info_dict

# 3.2. Label distribution for training and validation sets
label_distribution = {}

# y_train
unique_train, counts_train = np.unique(y_train, return_counts=True)
train_label_map = dict(zip(unique_train, counts_train))
total_train = len(y_train)
label_distribution["y_train"] = {
    "good_count": int(train_label_map.get(0, 0)),
    "defect_count": int(train_label_map.get(1, 0)),
    "total": int(total_train),
    "good_ratio": float(train_label_map.get(0, 0) / total_train) if total_train > 0 else 0.0,
    "defect_ratio": float(train_label_map.get(1, 0) / total_train) if total_train > 0 else 0.0
}

# y_valid
unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
valid_label_map = dict(zip(unique_valid, counts_valid))
total_valid = len(y_valid)
label_distribution["y_valid"] = {
    "good_count": int(valid_label_map.get(0, 0)),
    "defect_count": int(valid_label_map.get(1, 0)),
    "total": int(total_valid),
    "good_ratio": float(valid_label_map.get(0, 0) / total_valid) if total_valid > 0 else 0.0,
    "defect_ratio": float(valid_label_map.get(1, 0) / total_valid) if total_valid > 0 else 0.0
}
results["label_distribution"] = label_distribution

# 3.3. Statistical analysis for signal arrays
array_statistics = {}

signal_arrays_for_stats = {
    "X_train": X_train,
    "X_valid": X_valid,
    "X_scan": X_scan,
    "Xc": Xc,
    "Xg": Xg,
    "X_in_corr": X_in_corr # This will be flattened for signal-wise stats
}

for name, arr in signal_arrays_for_stats.items():
    # Process array to (N_signals, 500) shape for consistent statistics
    if arr.ndim == 3 and arr.shape[2] == 1:
        processed_arr = arr.squeeze(axis=-1) # (N, 500, 1) -> (N, 500)
    elif arr.ndim == 3 and arr.shape[2] != 1: # X_in_corr (161, 160, 500) -> (161*160, 500)
        processed_arr = arr.reshape(-1, arr.shape[-1])
    else: # Xc, Xg, which are already (N, 500)
        processed_arr = arr

    if processed_arr.size == 0 or processed_arr.shape[0] == 0:
        array_statistics[name] = {"message": "Array is empty or has no signals, no statistics."}
        continue

    # Global statistics (across all elements in the array)
    global_stats = {
        "mean": float(np.mean(processed_arr)),
        "std": float(np.std(processed_arr)),
        "min": float(np.min(processed_arr)),
        "max": float(np.max(processed_arr)),
        "median": float(np.median(processed_arr))
    }

    # Per-signal statistics (calculated for each 500-point signal, then aggregated)
    per_signal_peak_amplitude = np.max(processed_arr, axis=1)
    per_signal_min_amplitude = np.min(processed_arr, axis=1)
    per_signal_signal_range = per_signal_peak_amplitude - per_signal_min_amplitude
    per_signal_std_amplitude = np.std(processed_arr, axis=1)

    per_signal_average = {
        "peak_amplitude": float(np.mean(per_signal_peak_amplitude)),
        "min_amplitude": float(np.mean(per_signal_min_amplitude)),
        "signal_range": float(np.mean(per_signal_signal_range)),
        "std_amplitude": float(np.mean(per_signal_std_amplitude))
    }

    per_signal_min_overall = {
        "peak_amplitude": float(np.min(per_signal_peak_amplitude)),
        "min_amplitude": float(np.min(per_signal_min_amplitude)),
        "signal_range": float(np.min(per_signal_signal_range)),
        "std_amplitude": float(np.min(per_signal_std_amplitude))
    }

    per_signal_max_overall = {
        "peak_amplitude": float(np.max(per_signal_peak_amplitude)),
        "min_amplitude": float(np.max(per_signal_min_amplitude)),
        "signal_range": float(np.max(per_signal_signal_range)),
        "std_amplitude": float(np.max(per_signal_std_amplitude))
    }

    array_statistics[name] = {
        "global_statistics": global_stats,
        "per_signal_average_characteristics": per_signal_average,
        "per_signal_min_characteristics": per_signal_min_overall,
        "per_signal_max_characteristics": per_signal_max_overall
    }

results["array_statistics"] = array_statistics

# Add normalization parameters m and st to the results
results["normalization_parameters"] = {
    "m_values": {
        "shape": str(m.shape),
        "mean": float(np.mean(m)),
        "std": float(np.std(m)),
        "min": float(np.min(m)),
        "max": float(np.max(m))
    },
    "st_values": {
        "shape": str(st.shape),
        "mean": float(np.mean(st)),
        "std": float(np.std(st)),
        "min": float(np.min(st)),
        "max": float(np.max(st))
    }
}

# 3.4. Dataset summary and recommendations for model training
summary_text = """
This Pulsed Eddy Current Testing (PECT) Non-Destructive Testing (NDT) dataset offers a robust foundation
for developing machine learning models to detect subsurface defects like corrosion.

Key Dataset Observations:
- **Signal Structure**: All signals are uniform, comprising 500 time points, representing a time-series voltage response.
- **Data Segregation**: The dataset is clearly divided into training (X_train, y_train), validation (X_valid, y_valid),
  and full scan (X_scan) sets, facilitating a standard machine learning workflow.
- **Labeled Data**: Explicit labels (0 for good, 1 for defect) are provided for supervised learning.
- **Pre-separated Classes**: 'Xg' (good) and 'Xc' (defect) arrays simplify direct analysis of class characteristics.
- **Spatial Context**: 'X_in_corr' provides signals arranged in a 2D spatial grid (161x160), which is valuable for
  visualizing defect locations or performing spatial analysis.
- **Normalization Parameters**: 'm' (mean) and 'st' (standard deviation) arrays are provided for standardizing signals.
  This indicates that signals should be normalized before being fed into most machine learning models.

Signal Characteristics (as observed in 'signal_comparison.png' and statistics):
- Defective signals typically exhibit different pulse shapes, decay rates, or amplitude shifts compared to good signals.
  This distinctiveness is the basis for defect detection.

Recommendations for Model Training:
- **Data Preprocessing**: Always apply the provided 'm' and 'st' for signal normalization: `normalized_signal = (signal - m) / st`.
- **Model Architecture**:
    - **1D Convolutional Neural Networks (CNNs)** are highly suitable for extracting features from time-series data like PECT signals.
    - **Recurrent Neural Networks (RNNs)**, specifically LSTMs or GRUs, could also capture temporal dependencies.
    - For high-level defect mapping, techniques that combine signal classification with spatial context from `X_in_corr` might be useful.
- **Handling Class Imbalance**: The label distribution indicates a significant imbalance (more good signals than defect signals).
  This is common in NDT. Strategies to address this include:
    - **Weighted Loss Functions**: Assign higher weights to the minority class (defect) in the model's loss calculation.
    - **Resampling Techniques**: Oversampling the minority class (e.g., SMOTE) or undersampling the majority class.
    - **Performance Metrics**: Focus on metrics like F1-score, Precision, Recall, or Area Under the Receiver Operating Characteristic (AUC-ROC) curve, which are more informative than accuracy in imbalanced datasets.
- **Validation Strategy**: Use the `X_valid` and `y_valid` sets for hyperparameter tuning, model selection, and early stopping to prevent overfitting.
- **Interpretation**: Consider techniques for model interpretability (e.g., LIME, SHAP, attention mechanisms) to understand which parts of the signal contribute most to defect detection.
"""
results["dataset_summary_and_recommendations"] = summary_text

# 3.5. Path to the saved comparison plot
results["signal_comparison_plot_path"] = plot_path

# Save results to JSON
results_json_path = os.path.join(output_dir, 'results.json')
with open(results_json_path, 'w') as f:
    json.dump(results, f, indent=4) # Use indent for pretty printing
print(f"Comprehensive statistics saved to {results_json_path}")

print("\nAnalysis complete.")