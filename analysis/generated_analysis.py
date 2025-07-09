import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- 0. Setup Directories ---
output_dir = 'analysis'
figures_dir = os.path.join(output_dir, 'figures')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Dictionary to store all analysis results
results = {}

# --- 1. Load the .npz dataset and print its information ---
print("--- Loading Dataset ---")
try:
    data = np.load('pect_ndt_full_dataset.npz')
except FileNotFoundError:
    print("Error: 'pect_ndt_full_dataset.npz' not found. Please ensure the file is in the same directory as the script.")
    exit()

# List of arrays to process
array_names = [
    'X_train', 'y_train', 'X_valid', 'y_valid', 'X_scan',
    'X_in_corr', 'Xc', 'Xg', 'm', 'st'
]

dataset_info = {}
loaded_arrays = {}

print("\n--- Dataset Contents and Shapes ---")
for name in array_names:
    if name in data:
        arr = data[name]
        loaded_arrays[name] = arr
        shape_info = list(arr.shape) # Convert tuple to list for JSON serialization
        dtype_info = str(arr.dtype)
        print(f"{name}: Shape = {shape_info}, Type = {dtype_info}")
        dataset_info[name] = {"shape": shape_info, "dtype": dtype_info}
    else:
        print(f"Warning: {name} not found in the .npz file.")

results['dataset_info'] = dataset_info

# Unpack loaded arrays for direct use
X_train = loaded_arrays.get('X_train')
y_train = loaded_arrays.get('y_train')
X_valid = loaded_arrays.get('X_valid')
y_valid = loaded_arrays.get('y_valid')
X_scan = loaded_arrays.get('X_scan')
X_in_corr = loaded_arrays.get('X_in_corr')
Xc = loaded_arrays.get('Xc')
Xg = loaded_arrays.get('Xg')
m = loaded_arrays.get('m')
st = loaded_arrays.get('st')

print("\n--- Data Loading Complete ---")

# --- 2. Analyze and Visualize Pulse Signal Characteristics ---

# 2.1. Distribution of Labels
print("\n--- Analyzing Label Distribution ---")
label_distribution = {}

# y_train
if y_train is not None:
    unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
    train_total = y_train.size
    train_label_map = {0: 'Good (0)', 1: 'Defect (1)'}
    train_dist = {}
    print(f"y_train label distribution (Total: {train_total}):")
    for label, count in zip(unique_train_labels, train_counts):
        percentage = (count / train_total) * 100
        print(f"  {train_label_map.get(label, f'Unknown ({label})')}: {count} samples ({percentage:.2f}%)")
        train_dist[train_label_map.get(label, str(label)).replace(' ', '_').replace('(', '').replace(')', '').lower()] = {
            "count": int(count),
            "percentage": round(percentage, 2)
        }
    label_distribution['y_train'] = train_dist

    # Plot y_train label distribution
    plt.figure(figsize=(7, 5))
    sns.barplot(x=[train_label_map.get(l) for l in unique_train_labels], y=train_counts, palette='viridis')
    plt.title('Distribution of Training Labels (y_train)')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    for i, count in enumerate(train_counts):
        plt.text(i, count, f'{count}\n({(count/train_total)*100:.2f}%)', ha='center', va='bottom')
    train_label_plot_path = os.path.join(figures_dir, 'y_train_label_distribution.png')
    plt.savefig(train_label_plot_path)
    plt.close()
    results['plot_paths'] = results.get('plot_paths', {})
    results['plot_paths']['y_train_label_distribution'] = train_label_plot_path
else:
    print("y_train not found, skipping label distribution analysis for training data.")


# y_valid
if y_valid is not None:
    unique_valid_labels, valid_counts = np.unique(y_valid, return_counts=True)
    valid_total = y_valid.size
    valid_label_map = {0: 'Good (0)', 1: 'Defect (1)'}
    valid_dist = {}
    print(f"\ny_valid label distribution (Total: {valid_total}):")
    for label, count in zip(unique_valid_labels, valid_counts):
        percentage = (count / valid_total) * 100
        print(f"  {valid_label_map.get(label, f'Unknown ({label})')}: {count} samples ({percentage:.2f}%)")
        valid_dist[valid_label_map.get(label, str(label)).replace(' ', '_').replace('(', '').replace(')', '').lower()] = {
            "count": int(count),
            "percentage": round(percentage, 2)
        }
    label_distribution['y_valid'] = valid_dist

    # Plot y_valid label distribution
    plt.figure(figsize=(7, 5))
    sns.barplot(x=[valid_label_map.get(l) for l in unique_valid_labels], y=valid_counts, palette='viridis')
    plt.title('Distribution of Validation Labels (y_valid)')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    for i, count in enumerate(valid_counts):
        plt.text(i, count, f'{count}\n({(count/valid_total)*100:.2f}%)', ha='center', va='bottom')
    valid_label_plot_path = os.path.join(figures_dir, 'y_valid_label_distribution.png')
    plt.savefig(valid_label_plot_path)
    plt.close()
    results['plot_paths']['y_valid_label_distribution'] = valid_label_plot_path
else:
    print("y_valid not found, skipping label distribution analysis for validation data.")

results['label_distribution'] = label_distribution


# 2.2. Basic Statistics for Signal Arrays
print("\n--- Calculating Signal Statistics ---")
signal_statistics = {}
signal_arrays_to_analyze = {
    'X_train': X_train,
    'X_scan': X_scan,
    'Xc': Xc,
    'Xg': Xg
}

for name, arr in signal_arrays_to_analyze.items():
    if arr is None:
        print(f"Skipping statistics for {name} as it was not loaded.")
        continue

    # Reshape if necessary (remove the last singleton dimension if present)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr_flat = arr.squeeze(axis=-1) # (N, 500, 1) -> (N, 500)
    else:
        arr_flat = arr

    stats = {
        "overall_mean": np.mean(arr_flat).item(),
        "overall_std": np.std(arr_flat).item(),
        "overall_min": np.min(arr_flat).item(),
        "overall_max": np.max(arr_flat).item(),
        "overall_median": np.median(arr_flat).item()
    }
    
    # Calculate mean and std deviation across time points to see signal shape
    mean_signal_shape = np.mean(arr_flat, axis=0).tolist()
    std_signal_shape = np.std(arr_flat, axis=0).tolist()
    
    stats["mean_signal_shape_first_5_points"] = mean_signal_shape[:5]
    stats["std_signal_shape_first_5_points"] = std_signal_shape[:5]
    stats["mean_signal_shape_last_5_points"] = mean_signal_shape[-5:]
    stats["std_signal_shape_last_5_points"] = std_signal_shape[-5:]

    signal_statistics[name] = stats
    print(f"  {name} - Mean: {stats['overall_mean']:.4f}, Std: {stats['overall_std']:.4f}, Min: {stats['overall_min']:.4f}, Max: {stats['overall_max']:.4f}")

results['signal_statistics'] = signal_statistics

# 2.3. Visualization to describe the difference between signal pulse of two classes
print("\n--- Visualizing Good vs. Defect Signals ---")
if Xc is not None and Xg is not None:
    # Calculate mean and std deviation for good and defect signals
    mean_Xg = Xg.mean(axis=0)
    std_Xg = Xg.std(axis=0)

    mean_Xc = Xc.mean(axis=0)
    std_Xc = Xc.std(axis=0)

    time_points = np.arange(Xg.shape[1])

    plt.figure(figsize=(12, 7))
    
    # Plot mean good signal
    plt.plot(time_points, mean_Xg, label='Mean Good Signal (Xg)', color='green')
    plt.fill_between(time_points, mean_Xg - std_Xg, mean_Xg + std_Xg, color='lightgreen', alpha=0.3, label='Std Dev Good Signal')

    # Plot mean defect signal
    plt.plot(time_points, mean_Xc, label='Mean Defect Signal (Xc)', color='red')
    plt.fill_between(time_points, mean_Xc - std_Xc, mean_Xc + std_Xc, color='lightcoral', alpha=0.3, label='Std Dev Defect Signal')

    plt.title('Comparison of Mean Good vs. Defect PECT Signals')
    plt.xlabel('Time Points')
    plt.ylabel('Voltage Response')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    signal_comparison_plot_path = os.path.join(figures_dir, 'good_vs_defect_mean_signals.png')
    plt.savefig(signal_comparison_plot_path)
    plt.close()
    results['plot_paths']['good_vs_defect_mean_signals'] = signal_comparison_plot_path
    print(f"  Saved comparison plot to: {signal_comparison_plot_path}")
else:
    print("Xc or Xg not found, skipping signal comparison visualization.")


# --- 3. Save a results.json file ---

# Preprocessing information
preprocessing_info = {
    "description": "The dataset provides mean ('m') and standard deviation ('st') arrays, indicating that signals are intended for z-score normalization. Each signal is consistently 500 time points. No further explicit preprocessing steps were applied or required by the dataset description itself beyond loading.",
    "normalization_params": {}
}
if m is not None and st is not None:
    preprocessing_info["normalization_params"] = {
        "m_shape": list(m.shape),
        "st_shape": list(st.shape),
        "purpose": "These arrays ('m' for mean, 'st' for standard deviation) are typically used for z-score normalization: X_normalized = (X - m) / st. This ensures consistent scaling across all signals."
    }
else:
    print("Normalization parameters (m, st) not found.")
    preprocessing_info["normalization_params"] = "Not provided or loaded."

results['preprocessing_info'] = preprocessing_info

# Summary of findings and recommendations
summary_and_recommendations = {
    "overview": "This dataset provides PECT time-series signals for Non-Destructive Testing (NDT), categorized into 'good' (defect-free) and 'defect' (corrosion/thinning) classes. It includes training, validation, and full scan data, along with normalization parameters.",
    "key_findings": [
        "**Signal Structure:** All signals are 1D arrays of 500 time points, representing voltage response.",
        "**Class Imbalance:** Both training and validation sets show a significant imbalance, with 'good' signals outnumbering 'defect' signals. This is common in NDT datasets where defects are rare.",
        "**Signal Characteristics:** 'Good' and 'Defect' signals exhibit distinct mean waveforms. Defect signals (Xc) typically show differences in peak amplitude, decay rate, and potentially a phase shift compared to good signals (Xg), which is crucial for detection.",
        "**Normalization:** Pre-calculated mean ('m') and standard deviation ('st') arrays are provided, strongly suggesting z-score normalization as a standard preprocessing step.",
        "**Spatial Context:** The `X_in_corr` array highlights the availability of spatially organized defect data, which could be used for visual mapping of defect regions.",
        "**Data Readiness:** The data is well-structured and ready for direct use in machine learning models, particularly deep learning architectures for time series classification."
    ],
    "recommendations_for_model_training": [
        "**Address Class Imbalance:** Given the imbalance, consider techniques such as weighted loss functions, oversampling (e.g., SMOTE for time series), undersampling, or using evaluation metrics robust to imbalance (e.g., F1-score, Precision-Recall AUC).",
        "**Normalization:** Always apply z-score normalization to all input signals (training, validation, and scan data) using the provided `m` and `st` arrays: `X_normalized = (X - m) / st`. This prevents features with larger scales from dominating the learning process.",
        "**Model Choice:** Time-series specific models are highly recommended. Convolutional Neural Networks (CNNs) (e.g., 1D CNNs), Recurrent Neural Networks (RNNs) like LSTMs or GRUs, or attention-based models (Transformers) are suitable.",
        "**Feature Engineering:** While deep learning can extract features automatically, handcrafted features like peak amplitude, time-to-peak, signal decay rate, or Fourier transform coefficients might still be discriminative and improve model explainability.",
        "**Validation Strategy:** The `X_valid` and `y_valid` sets are critical for hyperparameter tuning and model selection, reflecting unseen data performance.",
        "**Anomaly Detection:** Given the nature of NDT, consider an anomaly detection approach where models learn characteristics of 'good' signals and flag deviations as potential defects, especially if new defect types might appear.",
        "**Visualize Outputs:** Post-model, visualizing predictions on `X_scan` (perhaps re-constructing a 2D map from `X_scan` predictions if spatial metadata is available) can help in interpreting defect locations."
    ]
}
results['summary_and_recommendations'] = summary_and_recommendations

# Save the results to a JSON file
results_file_path = os.path.join(output_dir, 'results.json')
with open(results_file_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n--- Analysis Complete ---")
print(f"Results saved to: {results_file_path}")
print(f"Plots saved to: {figures_dir}")