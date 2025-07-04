def analyze_dataset_prompt(data):
    """
    Generate a prompt for analyzing the PECT-NDT dataset (.npz format)
    """
    return f"""
**Instructions for Coder LLM: Analyze PECT-NDT Dataset and Save Results**

You are tasked with writing a Python script to analyze a preprocessed Pulsed Eddy Current Testing (PECT) dataset for Non-Destructive Testing (NDT) at `{data}`. The dataset is stored in a single `.npz` file and contains the following arrays:

###Information about the dataset:

This dataset consists of thousands of time-series signals collected from a grid scan over a large metal surface (such as a steel plate or pipeline). Each signal represents the electromagnetic response measured at a specific location when a pulsed eddy current is applied. The goal is to detect and characterize subsurface defects (such as corrosion or thinning) without damaging the material.

**How is the data structured?**

- Each signal is a 1D array of 500 time points, representing the voltage response over time at one scan position.
- The scan covers tens of thousands of positions, forming a 2D grid over the inspected surface.
- Signals from defect-free (good) regions and defective (corroded) regions are both included.
- Labels are provided to indicate whether each signal comes from a normal or defective area.

**File contents and shapes:**

- **X_train**: Training signals, shape `(15456, 500, 1)`  
  15,456 signals for model training, each with 500 time points.
- **y_train**: Training labels, shape `(15456,)`  
  0 = good, 1 = defect.
- **X_valid**: Validation signals, shape `(10304, 500, 1)`  
  10,304 signals for model validation.
- **y_valid**: Validation labels, shape `(10304,)`
- **X_scan**: All scan signals, shape `(25761, 500, 1)`  
  The complete set of measured signals from the entire scan area.
- **X_in_corr**: Signals inside corrosion regions, shape `(161, 160, 500)`  
  A 2D spatial grid (161 rows × 160 columns), each cell contains a 500-point signal.
- **Xc**: Corrosion (defect) signals, shape `(711, 500)`  
  All signals identified as coming from defective areas.
- **Xg**: Good (non-defect) signals, shape `(25049, 500)`  
  All signals from normal regions.
- **m**: Mean values for normalization, shape `(1, 500, 1)`  
  Used for standardizing signals.
- **st**: Standard deviation values for normalization, shape `(1, 500, 1)`

**How to use this dataset?**

- Each signal can be visualized as a waveform (voltage vs. time).
- You can analyze the difference in signal shape between good and defective regions.
- The dataset is suitable for training and evaluating machine learning models for defect detection, signal classification, or feature extraction.
- The spatial grid (`X_in_corr`) allows for visualization of defect regions as "images" of signal features.

###

All arrays retain their original shapes and are ready for direct use in deep learning frameworks or scientific analysis.
this is the required code you must use when loading the dataset:
 ```python
 import numpy as np

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
---
## Task

Write a Python script that:

1. **Loads the `.npz` dataset** and prints the shape and type of each array.
2. **Analyzes, visualize pulse signal characteristics** for each main array (e.g., X_train, Xc, Xg, X_scan), including:
    - Basic statistics (mean, std, min, max, median)
    - Distribution of labels (for y_train, y_valid)
    - visualization to describe the different between signal pusle of two class. draw both signals fron both class in the same plot to show the difference. 
3. **Saves a `results.json` file** in `analysis/` with:
    - Dataset info: shapes, types, label distribution
    - Extracted feature statistics (mean, std, min, max for each feature)
    - Description of preprocessing steps (if any)
    - Summary of findings and recommendations for model training (e.g., which features are most discriminative, any class imbalance, normalization advice)
    - Paths to the saved plots

---

## Requirements

- Use only the provided `.npz` file (do not generate random data).
- Use numpy, matplotlib, seaborn, and json for analysis and visualization.
- Ensure all plots are saved as `.png` in `analysis/figures/`.
- The `results.json` must be well-structured and easy to parse for downstream code generation.
- Provide clear comments and reasoning in the script for each analysis step.
---

## Notes

- The script should be general and reusable for similar PECT-NDT datasets.
- The analysis and plots should help a downstream user quickly understand the data and write code for model training.
- All reasoning and recommendations should be included in the `results.json` for transparency and reproducibility.

### Requirements
- Do **not** use random data or random seed and synthetic data; only analyze the provided `.npz` file.
- Ensure `results.json` is well-structured and comprehensive for PECT-NDT data.
- Focus on analyzing dataset characteristics, not a specific task like anomaly detection.

---
"""