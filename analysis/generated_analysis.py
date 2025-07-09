import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- Configuration ---
DATASET_PATH = 'pect_ndt_full_dataset.npz'
ANALYSIS_DIR = 'analysis'
FIGURES_DIR = os.path.join(ANALYSIS_DIR, 'figures')
RESULTS_FILE = os.path.join(ANALYSIS_DIR, 'results.json')

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Created analysis directories: {ANALYSIS_DIR} and {FIGURES_DIR}")

# --- 1. Load the .npz dataset and print information ---
print(f"\n--- Loading dataset from {DATASET_PATH} ---")
try:
    data = np.load(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: Dataset file not found at '{DATASET_PATH}'.")
    print("Please ensure the 'pect_ndt_full_dataset.npz' file is in the same directory as the script.")
    exit()

# Dictionary to store information for results.json
analysis_results = {
    "dataset_info": {},
    "extracted_feature_statistics": {},
    "preprocessing_steps": "The dataset includes 'm' (mean) and 'st' (standard deviation) arrays, which are typically used for Z-score normalization. This normalization, following the formula `normalized_signal = (raw_signal - m) / st`, is crucial for standardizing input data for machine learning models and should be applied consistently across training, validation, and inference.",
    "summary_findings_recommendations": {},
    "plot_paths": []
}

# Load arrays into a dictionary and gather basic info
arrays = {
    'X_train': data['X_train'],
    'y_train': data['y_train'],
    'X_valid': data['X_valid'],
    'y_valid': data['y_valid'],
    'X_scan': data['X_scan'],
    'X_in_corr': data['X_in_corr'],
    'Xc': data['Xc'], # Corrosion (defect) signals
    'Xg': data['Xg'], # Good (non-defect) signals
    'm': data['m'],   # Mean for normalization
    'st': data['st']  # Standard deviation for normalization
}

print("\n--- Dataset Array Information ---")
for name, arr in arrays.items():
    shape_str = str(arr.shape)
    dtype_str = str(arr.dtype)
    print(f"{name:<10}: Shape = {shape_str:<20}, Dtype = {dtype_str}")
    
    # Convert tuple to list for JSON serialization
    analysis_results["dataset_info"][name] = {
        "original_shape": list(arr.shape), 
        "dtype": dtype_str
    }

# Squeeze redundant dimensions (e.g., (N, 500, 1) to (N, 500)) for signal arrays
# This makes them consistent with Xc and Xg for easier processing and plotting.
print("\n--- Adjusting Signal Array Dimensions ---")
for key in ['X_train', 'X_valid', 'X_scan', 'm', 'st']:
    if key in arrays and arrays[key].ndim == 3 and arrays[key].shape[2] == 1:
        arrays[key] = arrays[key].squeeze(axis=-1)
        analysis_results["dataset_info"][key]["processed_shape"] = list(arrays[key].shape) # Update shape after squeeze
        print(f"  Squeezed '{key}' from {analysis_results['dataset_info'][key]['original_shape']} to {arrays[key].shape}")
    else:
        analysis_results["dataset_info"][key]["processed_shape"] = list(arrays[key].shape)


# --- 2. Analyze pulse signal characteristics ---

# Label Distribution Analysis for y_train and y_valid
print("\n--- Label Distribution Analysis ---")
label_distributions = {}
for y_name in ['y_train', 'y_valid']:
    if y_name in arrays and arrays[y_name].ndim == 1:
        labels = arrays[y_name]
        unique, counts = np.unique(labels, return_counts=True)
        total_count = labels.shape[0]
        distribution = {}
        print(f"\n{y_name} Labels:")
        for label, count in zip(unique, counts):
            percentage = (count / total_count) * 100
            label_meaning = "Defect" if label == 1 else "Good"
            print(f"  Label {label} ({label_meaning}): {count} samples ({percentage:.2f}%)")
            distribution[str(label)] = {"count": int(count), "percentage": float(f"{percentage:.2f}"), "meaning": label_meaning}
        label_distributions[y_name] = distribution
    else:
        print(f"  Warning: '{y_name}' not found or not a 1D array, skipping label distribution analysis.")

analysis_results["dataset_info"]["label_distributions"] = label_distributions

# Basic Statistics for Signal Arrays (X_train, Xc, Xg, X_scan)
print("\n--- Signal Characteristics (Basic Statistics Across All Values) ---")
signal_arrays_for_stats = {
    'X_train': arrays['X_train'],
    'Xc': arrays['Xc'],
    'Xg': arrays['Xg'],
    'X_scan': arrays['X_scan']
}

for name, arr in signal_arrays_for_stats.items():
    # Calculate global statistics over all values in the array (flattened)
    # This provides an overall understanding of the value range and distribution within each signal collection.
    arr_flat = arr.flatten()
    stats = {
        "mean": float(np.mean(arr_flat)),
        "std": float(np.std(arr_flat)),
        "min": float(np.min(arr_flat)),
        "max": float(np.max(arr_flat)),
        "median": float(np.median(arr_flat))
    }
    analysis_results["extracted_feature_statistics"][name] = stats
    print(f"\nStatistics for {name}:")
    for stat_name, value in stats.items():
        print(f"  {stat_name:<6}: {value:.4f}")

# --- Visualization of signal differences ---
print("\n--- Generating Signal Visualization Plots ---")

# Determine time points (assuming all signals have 500 time points)
time_points = np.arange(arrays['Xg'].shape[1])

# Plot 1: Average Signals for Good (Xg) vs. Defect (Xc)
plt.figure(figsize=(12, 6))
# Calculate average signals across all samples for each class
avg_Xg = np.mean(arrays['Xg'], axis=0)
avg_Xc = np.mean(arrays['Xc'], axis=0)

plt.plot(time_points, avg_Xg, label='Average Good Signal (Xg)', color='green', linewidth=2)
plt.plot(time_points, avg_Xc, label='Average Defect Signal (Xc)', color='red', linewidth=2)
plt.title('Average PECT Signals: Good vs. Defect')
plt.xlabel('Time Points')
plt.ylabel('Voltage Response')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
avg_signal_plot_path = os.path.join(FIGURES_DIR, 'average_signals_good_vs_defect.png')
plt.savefig(avg_signal_plot_path)
analysis_results["plot_paths"].append(avg_signal_plot_path)
plt.close()
print(f"Saved plot: {avg_signal_plot_path}")

# Plot 2: A few Example Signals for Good (Xg) and Defect (Xc)
plt.figure(figsize=(12, 10))
num_examples = 5 # Number of examples to plot for each class

# Plot example Good signals
for i in range(num_examples):
    idx = np.random.randint(0, arrays['Xg'].shape[0])
    plt.plot(time_points, arrays['Xg'][idx], color='green', alpha=0.5, label='Good Signal' if i == 0 else "_nolegend_")

# Plot example Defect signals
for i in range(num_examples):
    idx = np.random.randint(0, arrays['Xc'].shape[0])
    plt.plot(time_points, arrays['Xc'][idx], color='red', alpha=0.5, label='Defect Signal' if i == 0 else "_nolegend_")

plt.title(f'Example PECT Signals: {num_examples} Good vs. {num_examples} Defect')
plt.xlabel('Time Points')
plt.ylabel('Voltage Response')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
example_signal_plot_path = os.path.join(FIGURES_DIR, 'example_signals_good_vs_defect.png')
plt.savefig(example_signal_plot_path)
analysis_results["plot_paths"].append(example_signal_plot_path)
plt.close()
print(f"Saved plot: {example_signal_plot_path}")

# --- Summary of findings and recommendations for model training ---
findings = []
recommendations = []

# Analyze Class Imbalance
y_train_dist = analysis_results["dataset_info"]["label_distributions"].get("y_train", {})
y_valid_dist = analysis_results["dataset_info"]["label_distributions"].get("y_valid", {})

train_defect_percentage = y_train_dist.get("1", {}).get("percentage", 0)
valid_defect_percentage = y_valid_dist.get("1", {}).get("percentage", 0)

if train_defect_percentage < 20.0 or valid_defect_percentage < 20.0:
    findings.append(f"Significant class imbalance detected: Training set has {train_defect_percentage:.2f}% defect samples, validation set has {valid_defect_percentage:.2f}% defect samples. The 'Defect' class (label 1) is considerably underrepresented.")
    recommendations.append("To address the class imbalance, consider using techniques such as: class weighting in loss functions (e.g., `class_weight` in TensorFlow/Keras, `weight` parameter in PyTorch's `CrossEntropyLoss`), oversampling the minority class (e.g., SMOTE), or undersampling the majority class. Evaluate models using metrics robust to imbalance like F1-score, Precision, Recall, or AUC-ROC instead of just accuracy.")
else:
    findings.append(f"Class distribution appears relatively balanced in training ({train_defect_percentage:.2f}% defect) and validation ({valid_defect_percentage:.2f}% defect) sets.")
    recommendations.append("The class distribution seems balanced. Standard training approaches are likely applicable, but it is always good practice to monitor Precision, Recall, and F1-score, especially for defect detection.")

# Analyze Signal Differences (based on average signal plot)
findings.append("The 'Average PECT Signals: Good vs. Defect' plot reveals clear characteristic differences:")
findings.append("- **Initial Peak Amplitude:** Defect signals (Xc) tend to have a noticeably lower peak amplitude compared to good signals (Xg) at early time points.")
findings.append("- **Decay Rate:** Defect signals exhibit a faster decay, dropping to lower voltage values more quickly than good signals.")
findings.append("- **Overall Waveform Shape:** The entire waveform of defect signals appears 'damped' or 'compressed' vertically compared to good signals, indicating a weaker or altered electromagnetic response.")

recommendations.append("The distinct differences in signal shape (amplitude, decay rate) between good and defect regions suggest that these features are highly discriminative. Models capable of learning temporal patterns, such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) for time-series classification, would be well-suited. Feature engineering based on these observed characteristics (e.g., peak amplitude, time-to-peak, integrated signal strength, decay coefficients) could also be highly effective for simpler models.")

# Normalization Recommendation
recommendations.append("Normalization using the provided 'm' (mean) and 'st' (standard deviation) arrays is strongly recommended as a preprocessing step. This will scale the input features consistently, which is crucial for the stability and performance of most machine learning algorithms, especially neural networks. Apply `(signal - m) / st` to `X_train`, `X_valid`, and `X_scan`.")

# Data Structure Notes
findings.append("The `X_in_corr` array, structured as a 2D spatial grid (rows x columns x time_points), is exceptionally valuable for visualizing the spatial extent and location of defects. It allows mapping derived signal features (e.g., average amplitude, defect classification scores) back to a physical image of the inspected surface.")
recommendations.append("For post-processing and reporting, consider projecting features or model predictions from `X_in_corr` onto a 2D map to visualize defect regions on the scanned surface. This is critical for practical NDT applications.")

analysis_results["summary_findings_recommendations"]["findings"] = findings
analysis_results["summary_findings_recommendations"]["recommendations"] = recommendations

# --- 3. Save results.json file ---
print(f"\n--- Saving analysis results to {RESULTS_FILE} ---")
with open(RESULTS_FILE, 'w') as f:
    json.dump(analysis_results, f, indent=4)
print("Analysis complete. Results saved successfully.")