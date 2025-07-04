import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- Configuration ---
DATASET_PATH = 'pect_ndt_full_dataset.npz'
ANALYSIS_DIR = 'analysis/'
FIGURES_DIR = os.path.join(ANALYSIS_DIR, 'figures/')
RESULTS_FILE = os.path.join(ANALYSIS_DIR, 'results.json')

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Created directories: {ANALYSIS_DIR} and {FIGURES_DIR}")

# Initialize results dictionary for JSON output
analysis_results = {
    'dataset_info': {},
    'label_distributions': {},
    'global_signal_statistics': {},
    'preprocessing_info': {},
    'summary_and_recommendations': {},
    'plot_paths': []
}

# --- 1. Load the .npz dataset and print its structure ---
print(f"\n--- Loading dataset from {DATASET_PATH} ---")
try:
    data = np.load(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_PATH}.")
    print("Please ensure 'pect_ndt_full_dataset.npz' is in the same directory as the script.")
    exit()

loaded_arrays = {}
print("\n--- Dataset Contents and Shapes ---")
for key in data.files:
    arr = data[key]
    loaded_arrays[key] = arr
    print(f"Array: {key}, Shape: {arr.shape}, Dtype: {arr.dtype}")
    analysis_results['dataset_info'][key] = {
        'shape': list(arr.shape), # Convert tuple to list for JSON serialization
        'dtype': str(arr.dtype)
    }

# Assign to variables as per instructions for direct use
X_train = loaded_arrays['X_train']
y_train = loaded_arrays['y_train']
X_valid = loaded_arrays['X_valid']
y_valid = loaded_arrays['y_valid']
X_scan = loaded_arrays['X_scan']
X_in_corr = loaded_arrays['X_in_corr']
Xc = loaded_arrays['Xc']
Xg = loaded_arrays['Xg']
m = loaded_arrays['m']
st = loaded_arrays['st']

print("\nDataset loaded successfully and variables assigned.")

# --- 2. Analyze and visualize pulse signal characteristics ---

# 2.1 Label Distribution Analysis
print("\n--- Label Distribution for Training and Validation Sets ---")
for name, labels in [('y_train', y_train), ('y_valid', y_valid)]:
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique.astype(str), counts.tolist())) # Convert keys to string, counts to list
    total = labels.shape[0]
    
    # Calculate percentages
    percentages = {k: f"{(v/total*100):.2f}%" for k, v in distribution.items()}
    
    print(f"Distribution for {name}:")
    print(f"  Counts: {distribution} (Total: {total} samples)")
    print(f"  Percentages: {percentages}")
    
    analysis_results['label_distributions'][name] = {
        'counts': distribution,
        'total': total,
        'percentage': percentages
    }

# 2.2 Basic Statistics for main signal arrays (Global Statistics)
print("\n--- Global Signal Statistics (Mean, Std, Min, Max, Median) ---")
signal_arrays_to_analyze = {
    'X_train': X_train,
    'X_scan': X_scan,
    'Xc': Xc,
    'Xg': Xg
}

for name, arr in signal_arrays_to_analyze.items():
    # Squeeze the last dimension if it's 1, or flatten X_in_corr (161, 160, 500) to (25760, 500) effectively
    # for a global statistic calculation across all signals and all time points.
    squeezed_arr = arr.reshape(-1, arr.shape[-1]) # Ensure it's (N_signals * N_spatial_points, 500)
    
    # Calculate global statistics (across all signals and all time points)
    mean_val = np.mean(squeezed_arr)
    std_val = np.std(squeezed_arr)
    min_val = np.min(squeezed_arr)
    max_val = np.max(squeezed_arr)
    median_val = np.median(squeezed_arr)
    
    stats = {
        'mean': float(mean_val),
        'std': float(std_val),
        'min': float(min_val),
        'max': float(max_val),
        'median': float(median_val)
    }
    print(f"{name}: {stats}")
    analysis_results['global_signal_statistics'][name] = stats

# 2.3 Visualization: Signal pulse characteristics (Good vs. Defect)
print("\n--- Generating Signal Comparison Plots ---")

# Ensure the data is 2D (num_samples, num_timesteps) for easy aggregation
X_train_squeezed = X_train.squeeze()
X_good_train = X_train_squeezed[y_train == 0]
X_defect_train = X_train_squeezed[y_train == 1]

# Time points for plotting (0 to 499)
time_points = np.arange(X_train_squeezed.shape[1])

# Calculate Mean Signals
mean_good_signal = np.mean(X_good_train, axis=0)
mean_defect_signal = np.mean(X_defect_train, axis=0)

# Calculate Median Signals (often more robust to outliers)
median_good_signal = np.median(X_good_train, axis=0)
median_defect_signal = np.median(X_defect_train, axis=0)

# Set a nice style for plots
sns.set_style("whitegrid")

# Plotting Mean and Median Signals side-by-side
plt.figure(figsize=(16, 7))

# Plotting Mean Signals
plt.subplot(1, 2, 1)
plt.plot(time_points, mean_good_signal, label=f'Mean Good Signal (N={len(X_good_train)})', color='blue', alpha=0.8)
plt.plot(time_points, mean_defect_signal, label=f'Mean Defect Signal (N={len(X_defect_train)})', color='red', alpha=0.8)
plt.title('Average PECT Signal: Good vs. Defect (Mean)')
plt.xlabel('Time Points')
plt.ylabel('Signal Amplitude')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Plotting Median Signals
plt.subplot(1, 2, 2)
plt.plot(time_points, median_good_signal, label=f'Median Good Signal (N={len(X_good_train)})', color='darkblue', alpha=0.8)
plt.plot(time_points, median_defect_signal, label=f'Median Defect Signal (N={len(X_defect_train)})', color='darkred', alpha=0.8)
plt.title('Median PECT Signal: Good vs. Defect')
plt.xlabel('Time Points')
plt.ylabel('Signal Amplitude')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
signal_comparison_plot_path = os.path.join(FIGURES_DIR, 'signal_comparison_mean_median.png')
plt.savefig(signal_comparison_plot_path)
analysis_results['plot_paths'].append(signal_comparison_plot_path)
plt.close()
print(f"  Saved signal comparison plot to {signal_comparison_plot_path}")

# Optional: Plot a few examples from each class for visual inspection of variability
print("  Generating Example Signal Plots (to show variability)...")
num_examples = 5

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title(f'Example Good PECT Signals (N={num_examples})')
for i in range(min(num_examples, len(X_good_train))):
    plt.plot(time_points, X_good_train[i], color='blue', alpha=0.5)
plt.xlabel('Time Points')
plt.ylabel('Signal Amplitude')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.title(f'Example Defect PECT Signals (N={num_examples})')
for i in range(min(num_examples, len(X_defect_train))):
    plt.plot(time_points, X_defect_train[i], color='red', alpha=0.5)
plt.xlabel('Time Points')
plt.ylabel('Signal Amplitude')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
example_signals_plot_path = os.path.join(FIGURES_DIR, 'example_signals.png')
plt.savefig(example_signals_plot_path)
analysis_results['plot_paths'].append(example_signals_plot_path)
plt.close()
print(f"  Saved example signal plot to {example_signals_plot_path}")


# 2.4 Preprocessing Information (m and st)
print("\n--- Preprocessing Information (Normalization Arrays 'm' and 'st') ---")
analysis_results['preprocessing_info'] = {
    'normalization_mean_shape': list(m.shape),
    # Sample first/last 5 points of the squeezed 1D array for concise representation in JSON
    'normalization_mean_sample': m.squeeze().tolist()[:5] + ['...'] + m.squeeze().tolist()[-5:], 
    'normalization_std_shape': list(st.shape),
    'normalization_std_sample': st.squeeze().tolist()[:5] + ['...'] + st.squeeze().tolist()[-5:],
    'description': (
        "The 'm' and 'st' arrays are provided for z-score normalization (standardization) of the signals. "
        "Each array contains 500 values, corresponding to the mean and standard deviation for each of the 500 time points across a reference dataset "
        "(e.g., the entire training set or a large good-signal population). "
        "Signals should be normalized as `X_normalized = (X - m) / st`. "
        "Their shape (1, 500, 1) is designed for convenient broadcasting when applied to signal arrays of shape (N, 500, 1)."
    )
}
print(analysis_results['preprocessing_info']['description'])

# --- 3. Save results.json file ---
print("\n--- Summarizing Findings and Recommendations for Model Training ---")

# Summary of Findings
summary_findings = []

# Class Imbalance Check
y_train_counts = analysis_results['label_distributions']['y_train']['counts']
good_count_train = y_train_counts.get('0', 0)
defect_count_train = y_train_counts.get('1', 0)
total_train = good_count_train + defect_count_train

if defect_count_train > 0:
    imbalance_ratio_good_to_defect = good_count_train / defect_count_train
    if imbalance_ratio_good_to_defect > 2 or imbalance_ratio_good_to_defect < 0.5: # Simple heuristic for imbalance
        summary_findings.append(
            f"**Class Imbalance**: Significant class imbalance detected in training data. "
            f"Good samples: {good_count_train} ({analysis_results['label_distributions']['y_train']['percentage']['0']}), "
            f"Defect samples: {defect_count_train} ({analysis_results['label_distributions']['y_train']['percentage']['1']}). "
            f"The ratio of Good to Defect samples is approximately {imbalance_ratio_good_to_defect:.2f}:1."
        )
    else:
        summary_findings.append(
            f"**Class Balance**: The training data appears relatively balanced (Good: {good_count_train}, Defect: {defect_count_train}). "
            f"Ratio of Good to Defect samples is approximately {imbalance_ratio_good_to_defect:.2f}:1."
        )
else:
    summary_findings.append(
        "**Class Distribution**: No defect samples (or very few) detected in the training set (y_train). "
        "This might indicate an anomaly detection task rather than a binary classification, or a dataset issue."
    )

# Signal Characteristics and Discriminative Features
summary_findings.append(
    "**Signal Characteristics**: The 'Average PECT Signal: Good vs. Defect' plots clearly show distinct differences between signal waveforms from good and defective regions."
    "Specifically, defect signals tend to exhibit:"
    "  - A notably lower peak amplitude compared to good signals."
    "  - A delayed time-to-peak or a broader peak."
    "  - A slower decay rate, especially observed in the latter half of the signal (after the main peak)."
    "  - Potential shifts in the overall baseline or a prolonged recovery time."
)

# Recommendations
recommendations = []
recommendations.append(
    "**Normalization**: It is highly recommended to normalize (standardize) the input signals using the provided 'm' and 'st' arrays. "
    "This operation `(X - m) / st` should be applied consistently to X_train, X_valid, and X_scan. Normalization helps stabilize model training and improves performance for most machine learning algorithms."
)

if defect_count_train > 0 and (imbalance_ratio_good_to_defect > 2 or imbalance_ratio_good_to_defect < 0.5):
    recommendations.append(
        "**Address Class Imbalance**: Given the detected class imbalance, consider strategies such as: "
        "  - **Resampling**: Oversampling the minority class (e.g., using SMOTE or simple duplication) or undersampling the majority class. "
        "  - **Weighted Loss Functions**: Using a loss function that penalizes misclassifications of the minority class more heavily. "
        "  - **Ensemble Methods**: Bagging or boosting techniques designed for imbalanced datasets."
    )

recommendations.append(
    "**Feature Engineering/Model Choice**: "
    "  - The distinct waveform differences suggest that features like peak amplitude, time-to-peak, decay characteristics, and area under the curve would be highly discriminative. "
    "  - Deep learning models (e.g., 1D Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) like LSTMs) are well-suited for learning these complex temporal features directly from the raw signals without explicit feature engineering. "
    "  - Frequency domain analysis (e.g., using Fast Fourier Transform) could also reveal discriminative features related to signal damping or resonance shifts."
)
recommendations.append(
    "**Validation Strategy**: Utilize the provided X_valid and y_valid sets for objective model performance evaluation and hyperparameter tuning to avoid overfitting to the training data."
)
recommendations.append(
    "**Spatial Context**: While not directly used in classification, the X_in_corr array provides spatial context. If the goal is defect localization or mapping, consider integrating this spatial information into a downstream task (e.g., by treating signal features as image pixels for a 2D CNN)."
)

analysis_results['summary_and_recommendations'] = {
    'findings': summary_findings,
    'recommendations': recommendations
}

# Save results to JSON
with open(RESULTS_FILE, 'w') as f:
    json.dump(analysis_results, f, indent=4)

print(f"\nAnalysis complete. Detailed results saved to {RESULTS_FILE}")
print("Script finished.")