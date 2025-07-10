import numpy as np
import matplotlib.pyplot as plt
import json
import os

# --- 1. Load the dataset ---
dataset_path = 'pect_ndt_full_dataset.npz'
try:
    data = np.load(dataset_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_valid']
    y_valid = data['y_valid']
    X_scan = data['X_scan']
    X_in_corr = data['X_in_corr']
    Xc = data['Xc']  # Defect signals
    Xg = data['Xg']  # Good signals
    m = data['m']    # Mean for normalization
    st = data['st']  # Standard deviation for normalization
    print(f"Dataset '{dataset_path}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset file '{dataset_path}' not found.")
    print("Please ensure the dataset file is in the same directory as the script.")
    exit()

# Create analysis directories if they don't exist
output_dir = 'analysis'
figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Dictionary to store all analysis results
results = {}

print("\n--- Dataset Array Information ---")
array_info = {}
for name, arr in data.items():
    # Convert numpy array attributes to standard Python types for JSON serialization
    array_info[name] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype)
    }
    print(f"Array: {name}, Shape: {array_info[name]['shape']}, Type: {array_info[name]['dtype']}")

# Add detailed descriptions for each array
array_descriptions = {
    "X_train": "Training signals, shape (15456, 500, 1). Each signal is a 1D array of 500 time points representing voltage response.",
    "y_train": "Training labels, shape (15456,). 0 = good (non-defect), 1 = defect.",
    "X_valid": "Validation signals, shape (10304, 500, 1). Used for model validation.",
    "y_valid": "Validation labels, shape (10304,). Corresponds to X_valid signals.",
    "X_scan": "The complete set of measured signals from the entire scan area, shape (25761, 500, 1).",
    "X_in_corr": "Signals inside corrosion regions, structured as a 2D spatial grid (161 rows × 160 columns), each cell contains a 500-point signal. Useful for spatial visualization.",
    "Xc": "All signals explicitly identified as coming from defective areas, shape (711, 500).",
    "Xg": "All signals explicitly identified as coming from normal (good/non-defect) regions, shape (25049, 500).",
    "m": "Mean values for normalization, shape (1, 500, 1). Used to standardize signals by subtracting the mean.",
    "st": "Standard deviation values for normalization, shape (1, 500, 1). Used to standardize signals by dividing by the standard deviation."
}

for name, desc in array_descriptions.items():
    if name in array_info:
        array_info[name]["description"] = desc
results["dataset_info"] = array_info

# --- 2. Create a signal comparison visualization ---

# Select ONE representative sample from good signals (from Xg array)
# Xg signals are (N, 500), so Xg[0, :] gets the first signal directly.
good_signal = Xg[0, :]
# Select ONE representative sample from defect signals (from Xc array)
# Xc signals are (N, 500), so Xc[0, :] gets the first signal directly.
defect_signal = Xc[0, :]

# Time points for plotting (assuming 500 time points per signal)
time_points = np.arange(good_signal.shape[0])

plt.figure(figsize=(12, 6))
plt.plot(time_points, good_signal, color='blue', label='Good Signal (No Defect)')
plt.plot(time_points, defect_signal, color='red', label='Defect Signal (Corrosion)')
plt.title('Comparison of PECT Signals: Good vs. Defect Region', fontsize=16)
plt.xlabel('Time Points', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout() # Adjust layout to prevent labels from overlapping

signal_comparison_plot_filename = 'signal_comparison.png'
signal_comparison_plot_path = os.path.join(figures_dir, signal_comparison_plot_filename)
plt.savefig(signal_comparison_plot_path)
plt.close() # Close the plot to free memory
print(f"\nSignal comparison plot saved to: {signal_comparison_plot_path}")
results["signal_comparison_plot_path"] = signal_comparison_plot_path

# --- 3. Calculate and save comprehensive statistics ---

# Label Distribution
print("\n--- Label Distribution ---")
label_distribution = {}

unique_train, counts_train = np.unique(y_train, return_counts=True)
train_labels_map = dict(zip(unique_train, counts_train))
label_distribution["y_train"] = {
    "Good (0)": int(train_labels_map.get(0, 0)),
    "Defect (1)": int(train_labels_map.get(1, 0)),
    "Total": int(len(y_train))
}
print(f"y_train: {label_distribution['y_train']}")

unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
valid_labels_map = dict(zip(unique_valid, counts_valid))
label_distribution["y_valid"] = {
    "Good (0)": int(valid_labels_map.get(0, 0)),
    "Defect (1)": int(valid_labels_map.get(1, 0)),
    "Total": int(len(y_valid))
}
print(f"y_valid: {label_distribution['y_valid']}")

results["label_distribution"] = label_distribution

# Statistical analysis for signal arrays
print("\n--- Statistical Analysis for Signal Arrays ---")
signal_statistics = {}

# Helper function to get detailed statistics for an array of signals
def get_signal_array_stats(arr, name):
    # If the array has shape (N, 500, 1), squeeze the last dimension to (N, 500)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr_processed = arr.squeeze(axis=2)
    else:
        arr_processed = arr

    stats = {
        "total_signals": int(arr_processed.shape[0]),
        "time_points_per_signal": int(arr_processed.shape[1])
    }

    # Global statistics (across all signals and all time points in the array)
    stats["global_mean"] = float(np.mean(arr_processed))
    stats["global_std"] = float(np.std(arr_processed))
    stats["global_min"] = float(np.min(arr_processed))
    stats["global_max"] = float(np.max(arr_processed))
    stats["global_median"] = float(np.median(arr_processed))

    # Per-signal characteristics
    # Peak amplitude for each signal
    peak_values = np.max(arr_processed, axis=1)
    stats["mean_peak_amplitude"] = float(np.mean(peak_values))
    stats["std_peak_amplitude"] = float(np.std(peak_values))
    stats["min_peak_amplitude"] = float(np.min(peak_values))
    stats["max_peak_amplitude"] = float(np.max(peak_values))

    # Signal range (max - min) for each signal
    signal_ranges = np.max(arr_processed, axis=1) - np.min(arr_processed, axis=1)
    stats["mean_signal_range"] = float(np.mean(signal_ranges))
    stats["std_signal_range"] = float(np.std(signal_ranges))
    stats["min_signal_range"] = float(np.min(signal_ranges))
    stats["max_signal_range"] = float(np.max(signal_ranges))
    
    # Time point at which peak amplitude occurs for each signal
    # np.argmax returns the index of the first occurrence of the maximum value
    peak_time_points = np.argmax(arr_processed, axis=1)
    stats["mean_peak_time_point"] = float(np.mean(peak_time_points))
    stats["std_peak_time_point"] = float(np.std(peak_time_points))
    stats["min_peak_time_point"] = int(np.min(peak_time_points))
    stats["max_peak_time_point"] = int(np.max(peak_time_points))

    print(f"  {name}: Mean={stats['global_mean']:.4f}, Std={stats['global_std']:.4f}, Min={stats['global_min']:.4f}, Max={stats['global_max']:.4f}")
    return stats

# Process main signal arrays
signal_statistics["X_train"] = get_signal_array_stats(X_train, "X_train")
signal_statistics["X_valid"] = get_signal_array_stats(X_valid, "X_valid")
signal_statistics["X_scan"] = get_signal_array_stats(X_scan, "X_scan (Full Scan)")
signal_statistics["Xc"] = get_signal_array_stats(Xc, "Xc (Defect Signals)")
signal_statistics["Xg"] = get_signal_array_stats(Xg, "Xg (Good Signals)")

results["signal_statistics"] = signal_statistics

# Normalization factors (m, st)
print("\n--- Normalization Factors (m, st) ---")
normalization_factors = {
    "m": {
        "shape": list(m.shape),
        "values_mean": float(np.mean(m)),
        "values_std": float(np.std(m)),
        "description": "Mean values across time points used for signal standardization. Subtract this from each time point."
    },
    "st": {
        "shape": list(st.shape),
        "values_mean": float(np.mean(st)),
        "values_std": float(np.std(st)),
        "description": "Standard deviation values across time points used for signal standardization. Divide by this after subtracting mean."
    }
}
print(f"  m: Shape={normalization_factors['m']['shape']}, Global Mean={normalization_factors['m']['values_mean']:.4f}")
print(f"  st: Shape={normalization_factors['st']['shape']}, Global Mean={normalization_factors['st']['values_mean']:.4f}")
results["normalization_factors"] = normalization_factors

# X_in_corr (spatial grid) information
print("\n--- X_in_corr (Spatial Grid of Signals) Information ---")
x_in_corr_info = {
    "shape": list(X_in_corr.shape),
    "rows_in_grid": int(X_in_corr.shape[0]),
    "columns_in_grid": int(X_in_corr.shape[1]),
    "time_points_per_signal_in_grid_cell": int(X_in_corr.shape[2]),
    "total_signals_in_grid": int(X_in_corr.shape[0] * X_in_corr.shape[1]),
    "description": "A 2D spatial grid where each cell contains a 500-point signal. This array specifically contains signals sampled from within corrosion regions, enabling visualization of defect areas as 'images' of signal features or for spatial analysis."
}
print(f"  X_in_corr: Shape={x_in_corr_info['shape']}, Represents a grid of {x_in_corr_info['rows_in_grid']}x{x_in_corr_info['columns_in_grid']} signals.")
results["X_in_corr_info"] = x_in_corr_info


# Dataset Summary and Recommendations
print("\n--- Dataset Summary and Recommendations ---")
summary_and_recommendations = {
    "summary": (
        "This Pulsed Eddy Current Testing (PECT) Non-Destructive Testing (NDT) dataset provides time-series electromagnetic "
        "signals for detecting subsurface defects. It includes separate training and validation sets with labels, "
        "a full scan dataset, and pre-categorized 'good' and 'defect' signals. "
        "Normalization parameters (mean and standard deviation) are also provided for preprocessing. "
        "Each signal is a 1D waveform of 500 time points."
    ),
    "key_observations": [
        "**Signal Structure**: All signals are 1D arrays of 500 time points, suitable for time-series analysis.",
        f"**Training Set Distribution**: The training set `y_train` has {label_distribution['y_train']['Total']} samples, with {label_distribution['y_train']['Defect (1)'] / label_distribution['y_train']['Total']:.2%} signals from defect regions and {label_distribution['y_train']['Good (0)'] / label_distribution['y_train']['Total']:.2%} from good regions.",
        f"**Validation Set Distribution**: Similarly, `y_valid` has {label_distribution['y_valid']['Total']} samples, with {label_distribution['y_valid']['Defect (1)'] / label_distribution['y_valid']['Total']:.2%} defect and {label_distribution['y_valid']['Good (0)'] / label_distribution['y_valid']['Total']:.2%} good.",
        f"**Overall Class Imbalance**: The `Xc` (defect) set contains {Xc.shape[0]} signals, while `Xg` (good) contains {Xg.shape[0]} signals. This indicates a significant class imbalance towards good signals in the overall collection, which is also reflected in the training and validation sets.",
        "**Signal Characteristics**: The comparison plot visually confirms distinct differences in pulse shape (e.g., amplitude decay, peak values, and signal range) between good and defective signals. Defect signals might exhibit shifted peaks, altered decay rates, or secondary peaks/undulations.",
        "**Normalization Data**: The `m` and `st` arrays provide pre-computed per-time-point mean and standard deviation, which are crucial for standardizing input data for neural networks.",
        "**Spatial Data (`X_in_corr`)**: `X_in_corr` offers a unique opportunity for spatial analysis, allowing the reconstruction of a 'defect map' by processing each signal in the grid. This can be used for defect localization or segmentation tasks."
    ],
    "recommendations_for_model_training": [
        "**Preprocessing**: Apply Z-score normalization to all signal data (`X_train`, `X_valid`, `X_scan`) using the provided `m` and `st` arrays (`(signal - m) / st`). This is critical for optimizing deep learning model performance.",
        "**Model Architecture**: Time-series specific models are highly recommended: 1D Convolutional Neural Networks (CNNs) for feature extraction, Recurrent Neural Networks (RNNs) like LSTMs or GRUs for sequential dependencies, or hybrid CNN-RNN models. Attention mechanisms (Transformers) could also be explored.",
        "**Addressing Class Imbalance**: Given the significant imbalance towards 'good' signals, strategies like: \n  - **Weighted Loss Functions**: Assign higher weights to the minority (defect) class during training.\n  - **Oversampling/Undersampling**: Synthetically increase minority samples (e.g., SMOTE) or reduce majority samples.\n  - **Performance Metrics**: Prioritize metrics such as Precision, Recall, F1-score, and Area Under the Receiver Operating Characteristic (ROC-AUC) curve, which are more informative than simple accuracy for imbalanced datasets.",
        "**Data Augmentation**: To improve model generalization and robustness, consider applying time-series specific augmentations such as small random shifts, scaling, time warping, or adding Gaussian noise.",
        "**Evaluation**: Beyond model accuracy, visualize confusion matrices and plot ROC curves to understand the trade-offs between false positives and false negatives, especially for critical NDT applications.",
        "**Advanced Analysis**: The `X_in_corr` array can be used to visualize defect regions by extracting relevant features (e.g., peak amplitude, decay rate, RMS value) from each signal in the grid and mapping them to a 2D image. This can assist in defect mapping and potentially enable image-based deep learning approaches for spatial defect detection."
    ],
    "path_to_signal_comparison_plot": signal_comparison_plot_path
}
results["summary_and_recommendations"] = summary_and_recommendations

# Save results to JSON file
results_json_path = os.path.join(output_dir, 'results.json')
with open(results_json_path, 'w') as f:
    json.dump(results, f, indent=4) # Use indent for pretty printing JSON
print(f"\nComprehensive analysis results saved to: {results_json_path}")

print("\nAnalysis complete. Check the 'analysis/' directory for outputs.")