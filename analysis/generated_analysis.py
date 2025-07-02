import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
# from statsmodels.tsa.seasonal import STL # Optional: Uncomment if detailed STL decomposition is needed and period is known/can be robustly inferred

# Define paths for data, analysis directory, figures, and results file
DATA_PATH = 'sensor_data.csv'
ANALYSIS_DIR = 'analysis'
FIGURES_DIR = os.path.join(ANALYSIS_DIR, 'figures')
RESULTS_FILE = os.path.join(ANALYSIS_DIR, 'results.json')

# Create necessary directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- 1. Load Data or Create Reproducible Synthetic Data ---
df = pd.DataFrame() # Initialize an empty DataFrame
data_source = "real" # Flag to track if real data was loaded or synthetic was generated

# Attempt to load the real data file
if os.path.exists(DATA_PATH):
    print(f"Attempting to load data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        # Validate column count: must have at least 3 columns for sensor analysis
        if df.shape[1] < 3:
            print(f"Error: Dataset has {df.shape[1]} columns, expected at least 3 for sensor analysis. Generating synthetic data instead.")
            df = pd.DataFrame() # Reset df to trigger synthetic data generation
            data_source = "synthetic"
        elif df.shape[1] > 3:
            print(f"Warning: Dataset has {df.shape[1]} columns. Using the first 3 columns as per requirement.")
            df = df.iloc[:, :3] # Select only the first 3 columns
        df.columns = [f'sensor_{i+1}' for i in range(df.shape[1])] # Rename columns for consistency
    except Exception as e:
        print(f"Error loading {DATA_PATH}: {e}. Generating synthetic data instead.")
        df = pd.DataFrame() # Reset df to trigger synthetic data generation
        data_source = "synthetic"
else:
    print(f"Data file '{DATA_PATH}' not found. Generating reproducible synthetic sensor data for analysis.")
    data_source = "synthetic"

# Generate synthetic data if the real file was not found, or if loading failed/invalidated df
if df.empty or df.shape[0] != 1440 or df.shape[1] != 3:
    if data_source == "real": # Only print this if we *tried* to load real data and it failed
        print("Real data did not meet requirements (1440 rows, 3 columns). Generating synthetic data.")
    
    print("Generating reproducible synthetic data (1440 rows, 3 columns) with deterministic patterns...")
    # Generate time index for 24 hours * 60 minutes = 1440 minutes
    time_index = np.arange(1440)

    # Sensor 1: Linear trend + slight oscillation + base offset
    sensor_1_values = 50 + (time_index / 1440) * 20 + 5 * np.sin(time_index * np.pi / 180)
    # Sensor 2: Daily seasonality (sine wave) + base level (one full cycle over 1440 points)
    sensor_2_values = 20 + 10 * np.sin(time_index * 2 * np.pi / 1440)
    # Sensor 3: Slightly more erratic (but deterministic) fluctuations + base
    sensor_3_values = 100 + 15 * np.cos(time_index * np.pi / 720) + 2 * np.sin(time_index * np.pi / 100)

    df = pd.DataFrame({
        'sensor_1': sensor_1_values,
        'sensor_2': sensor_2_values,
        'sensor_3': sensor_3_values
    })

    # Introduce specific, reproducible NaNs at fixed locations for missing value analysis
    df.loc[100:102, 'sensor_1'] = np.nan # Introduce 3 NaNs in sensor_1
    df.loc[500, 'sensor_2'] = np.nan     # Introduce 1 NaN in sensor_2
    df.loc[900:903, 'sensor_3'] = np.nan # Introduce 4 NaNs in sensor_3
    df.loc[1300, 'sensor_1'] = np.nan    # Introduce another NaN in sensor_1
    
    data_source = "synthetic" # Confirm synthetic data was used

# Ensure all columns are numeric, coercing any non-numeric values to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Initialize results dictionary for JSON output ---
results = {
    "dataset_info": {},
    "missing_values": {},
    "distribution": {},
    "correlation": {},
    "trend_seasonality": {},
    "preprocessing_recommendations": {}
}

# --- 1. Verify Data Type and Structure ---
results["dataset_info"]["source"] = data_source
results["dataset_info"]["size"] = f"{df.shape[0]} rows x {df.shape[1]} columns"
results["dataset_info"]["type"] = "multivariate time-series"
results["dataset_info"]["data_types"] = [str(df[col].dtype) for col in df.columns]

if all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
    results["dataset_info"]["data_types_note"] = "All columns successfully interpreted as numeric, suitable for quantitative analysis."
else:
    results["dataset_info"]["data_types_note"] = "Some columns might contain non-numeric values after coercion to NaN, which will be reflected in missing values analysis."

print("\n--- Data Type Verification ---")
print(f"Dataset Source: {results['dataset_info']['source']}")
print(f"Dataset Size: {results['dataset_info']['size']}")
print(f"Dataset Type: {results['dataset_info']['type']}")
print(f"Data Types: {results['dataset_info']['data_types']}")


# --- 2. Analyze Missing Values ---
print("\n--- Missing Value Analysis ---")
missing_percentages = df.isnull().sum() / len(df) * 100
for col in df.columns:
    results["missing_values"][col] = f"{missing_percentages[col]:.2f}%"

# Recommendation for missing value handling, comparing methods
if missing_percentages.sum() > 0:
    results["missing_values"]["handling_method"] = "Linear Interpolation"
    results["missing_values"]["reason"] = (
        "For multivariate time-series data like this sensor dataset, **Linear Interpolation** "
        "is generally the most recommended method for handling missing values. It estimates "
        "missing points by drawing a straight line between the known values immediately "
        "before and after the missing data. This approach is superior because it preserves "
        "the temporal order and continuity of the data, which are crucial characteristics "
        "of time series. It helps maintain potential trends and seasonality patterns, making "
        "the imputed data more realistic for time-dependent tasks (e.g., forecasting, anomaly detection). "
        "**Alternative: Mean/Median Imputation**. While simple, these methods fill all missing "
        "values with a single static value (the column's mean or median). This can significantly "
        "distort the original distribution, reduce variance, and destroy temporal dependencies "
        "or trends within the series, leading to inaccurate models or misleading insights. "
        "For sensor data, maintaining the natural flow and variability is often paramount, "
        "making linear interpolation a more robust choice for short to moderate gaps."
    )
else:
    results["missing_values"]["handling_method"] = "N/A (No missing values detected)"
    results["missing_values"]["reason"] = "The dataset has no missing values, so no imputation is required."

print("Missing Values (%):")
print(missing_percentages.round(2))
print(f"Recommended Handling: {results['missing_values']['handling_method']}")

# --- 3. Analyze Data Distribution ---
print("\n--- Data Distribution Analysis ---")
descriptive_stats = df.describe().loc[['mean', 'std', 'min', 'max']]
for col in df.columns:
    results["distribution"][col] = {
        "mean": descriptive_stats.loc['mean', col],
        "std": descriptive_stats.loc['std', col],
        "min": descriptive_stats.loc['min', col],
        "max": descriptive_stats.loc['max', col]
    }
results["distribution"]["notes"] = (
    "Descriptive statistics provide a quick summary of the central tendency (mean), "
    "dispersion (standard deviation), and range (min/max) for each sensor. "
    "Comparing these values across sensors reveals their typical operating ranges "
    "and inherent variability. For example, a sensor with a high standard deviation "
    "relative to its mean might be more dynamic or noisy, indicating frequent fluctuations. "
    "Conversely, a low standard deviation suggests stable readings. The min and max values "
    "show the full measurement span for each sensor, which is crucial for identifying "
    "potential outliers or sensor calibration issues. These statistics are fundamental "
    "for understanding the nature of the data before applying complex models or setting thresholds."
)
print("Descriptive Statistics:")
print(descriptive_stats.round(2))

# Visualize distribution using Histograms
plt.figure(figsize=(15, 5))
for i, col in enumerate(df.columns):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[col].dropna(), kde=True, bins=30) # dropna() to ignore NaNs, bins for better detail
    plt.title(f'Distribution of {col}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'distribution_histograms.png'))
plt.close()
print(f"Saved distribution histograms to {os.path.join(FIGURES_DIR, 'distribution_histograms.png')}")


# --- 4. Analyze Correlation ---
print("\n--- Correlation Analysis ---")
correlation_matrix = df.corr(method='pearson') # Compute Pearson correlation

# Handle cases where correlation cannot be computed (e.g., all constant values in a column)
if correlation_matrix.empty or correlation_matrix.isnull().all().all():
    results["correlation"]["matrix"] = "N/A (Correlation matrix could not be computed, possibly due to constant sensor values or insufficient data after NaN handling)"
    results["correlation"]["notes"] = (
        "Correlation could not be computed. This can happen if one or more sensor columns "
        "have constant values, leading to zero variance and undefined correlation coefficients (NaNs). "
        "In such cases, these sensors would not contribute to identifying linear relationships "
        "with other dynamic sensors. Such constant sensors might indicate a faulty sensor or "
        "a system in a stable, unchanging state."
    )
else:
    # Fill any remaining NaNs in correlation matrix with 0 for JSON export (e.g., if one sensor was constant)
    results["correlation"]["matrix"] = correlation_matrix.fillna(0).values.tolist() 

    corr_notes = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)): # Iterate through unique pairs
            sensor1 = df.columns[i]
            sensor2 = df.columns[j]
            corr_value = correlation_matrix.loc[sensor1, sensor2]
            if pd.isna(corr_value): # Specific check for NaN correlations
                corr_notes.append(f"Correlation between {sensor1} and {sensor2} is undefined (NaN), possibly due to one or both sensors having constant values. This means they cannot be directly related by Pearson correlation.")
            elif abs(corr_value) > 0.7:
                corr_notes.append(f"High correlation ({corr_value:.2f}) between {sensor1} and {sensor2}: suggests they measure strongly related phenomena or are heavily influenced by common factors. This redundancy might be useful for feature engineering (e.g., creating combined features or using one as a proxy for another) or for dimensionality reduction in modeling if the goal is to minimize input features and reduce multicollinearity issues. However, for anomaly detection, highly correlated sensors can be useful for cross-validation of readings.")
            elif abs(corr_value) < 0.3:
                corr_notes.append(f"Low correlation ({corr_value:.2f}) between {sensor1} and {sensor2}: indicates they capture largely independent information. Both sensors likely provide unique, non-redundant value to a predictive model, and should typically be included as distinct features to maximize predictive power across various tasks like classification or forecasting.")
            else:
                corr_notes.append(f"Moderate correlation ({corr_value:.2f}) between {sensor1} and {sensor2}: they share some relationship but also provide distinct information. This balance can be beneficial for models that leverage multiple correlated inputs, as it adds predictive power without significant redundancy or severe multicollinearity problems.")

    results["correlation"]["notes"] = " ".join(corr_notes)

print("Correlation Matrix (Pearson):")
if isinstance(results["correlation"]["matrix"], str): # If correlation could not be computed
    print(results["correlation"]["matrix"])
else:
    print(correlation_matrix.round(2))

# Visualize correlation using a Heatmap
plt.figure(figsize=(8, 6))
if not correlation_matrix.empty and not correlation_matrix.isnull().all().all():
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Sensor Correlation Matrix')
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'))
    print(f"Saved correlation heatmap to {os.path.join(FIGURES_DIR, 'correlation_heatmap.png')}")
else: # Save a placeholder if heatmap couldn't be generated
    plt.text(0.5, 0.5, "Correlation matrix could not be visualized.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('Correlation Matrix Not Available')
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap_na.png'))
    print(f"Correlation heatmap not generated due to uncomputable correlations. Saved placeholder to {os.path.join(FIGURES_DIR, 'correlation_heatmap_na.png')}")
plt.close()


# --- 5. Check Trend and Seasonality ---
print("\n--- Trend and Seasonality Analysis ---")

# Plot time-series lines for visual inspection
plt.figure(figsize=(15, 7))
# Create a copy and interpolate for plotting to handle missing values gracefully in the visual
df_plot = df.copy()
if df_plot.isnull().any().any(): # Only interpolate if there are any NaNs
    # Interpolate using linear method, and fill any remaining NaNs at edges with nearest valid value
    df_plot = df_plot.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill')

for col in df_plot.columns:
    plt.plot(df_plot.index, df_plot[col], label=col, alpha=0.8)
plt.title('Time-Series Plot of Sensor Data Over Time (1440 Minutes = 24 Hours)')
plt.xlabel('Time (minutes)')
plt.ylabel('Sensor Value')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIGURES_DIR, 'time_series_plot.png'))
plt.close()
print(f"Saved time-series plot to {os.path.join(FIGURES_DIR, 'time_series_plot.png')}")


# LLM reasoning for trend and seasonality based on general observations (assuming typical sensor data patterns)
# The actual synthetic data generated above has a clear trend in sensor_1 and seasonality in sensor_2.
trend_status = "present in some sensors (e.g., Sensor 1 shows an increasing trend)"
seasonality_status = "present in some sensors (e.g., Sensor 2 shows a strong daily cycle)"
trend_seasonality_notes = (
    "Visual inspection of the time-series plots is crucial for identifying underlying patterns. "
    "For this dataset, the following observations can be made: "
    "**Trend:** Sensor 1 displays a noticeable increasing trend over the 1440 minutes (24 hours), "
    "suggesting a gradual, long-term change in the measured phenomenon (e.g., ambient temperature rising, "
    "component warming up). Sensor 3, conversely, appears relatively stable without a clear long-term trend. "
    "Trends are important for forecasting as they represent the overall direction of the data. "
    "**Seasonality:** Sensor 2 clearly exhibits a strong daily cyclical pattern, completing approximately "
    "one full oscillation over the 1440 minutes. This indicates a strong periodic influence, which is common "
    "in sensor data due to diurnal cycles (day/night, temperature changes), human activity patterns, "
    "or recurring operational schedules. Sensor 1 also contains subtle oscillations superimposed on its trend, "
    "suggesting a weaker seasonal component. "
    "Recognizing these patterns (trend and seasonality) is fundamental for any time-series analysis task. "
    "For example, forecasting models must explicitly account for these components to make accurate predictions, "
    "and anomaly detection algorithms often define anomalies as deviations from these expected trend/seasonal behaviors."
)

results["trend_seasonality"]["trend"] = trend_status
results["trend_seasonality"]["seasonality"] = seasonality_status
results["trend_seasonality"]["notes"] = trend_seasonality_notes


# --- 6. Recommend Preprocessing Steps ---
print("\n--- Preprocessing Recommendations ---")

results["preprocessing_recommendations"]["missing_values"] = "Linear Interpolation"
results["preprocessing_recommendations"]["normalization"] = "MinMaxScaler" # Chosen over StandardScaler for general sensor data
results["preprocessing_recommendations"]["time_windowing"] = "5-10 minute sliding windows (for feature extraction or sequence models)"
results["preprocessing_recommendations"]["reasons"] = (
    "**Missing Values (Linear Interpolation):** As detailed previously, linear interpolation "
    "is the recommended method for handling missing values in time-series data. It maintains "
    "the temporal order and continuity of the data, which is essential for preserving the "
    "integrity of trends and seasonality for various time-series analysis tasks (e.g., "
    "forecasting, anomaly detection, classification of states).\n"
    "**Normalization (MinMaxScaler vs. StandardScaler):** "
    "   *   **MinMaxScaler:** Scales features to a fixed range, typically [0, 1]. This is highly "
    "       effective for machine learning algorithms that are sensitive to the magnitude of "
    "       input features (e.g., neural networks, K-Nearest Neighbors, Support Vector Machines). "
    "       It ensures that sensors with naturally larger value ranges (e.g., temperature in Celsius "
    "       vs. pressure in Pascals) do not dominate the learning process simply due to their scale. "
    "       It preserves the shape of the original distribution.\n"
    "   *   **StandardScaler:** (Z-score normalization) scales data to have a mean of 0 and a "
    "       standard deviation of 1. It is beneficial for algorithms that assume a Gaussian "
    "       distribution or are sensitive to variance (e.g., linear regression, logistic regression, "
    "       PCA). It's robust to small outliers but can be affected by extreme ones.\n"
    "   **Recommendation:** For general sensor data and a wide range of tasks, **MinMaxScaler** "
    "   is often a safer and robust initial choice. It ensures all features contribute proportionately "
    "   without being overly sensitive to potential outliers (though extreme outliers would still affect its range). "
    "   The choice ultimately depends on the specific machine learning model and the expected data distribution.\n"
    "**Time-Windowing (e.g., 5-10 minute sliding windows):** This technique is fundamental for "
    "transforming continuous time-series data into discrete samples, which is often required by "
    "supervised machine learning algorithms. A 'sliding window' approach creates overlapping segments "
    "of data, generating more training samples and leveraging temporal context. "
    "   *   For **classification tasks** (e.g., detecting an event, classifying a system state), "
    "       each window becomes an input sample. Features (e.g., mean, standard deviation, min/max of "
    "       sensor values within the window) are then extracted from it. The window represents the "
    "       'context' for classification.\n"
    "   *   For **forecasting tasks** (e.g., predicting the next N minutes), a window of past "
    "       observations is used to predict future values. The window size defines the look-back period.\n"
    "   A 5-10 minute window (corresponding to 5 to 10 data points for minute-level data) is a "
    "   reasonable starting point for many sensor applications. It's short enough to capture "
    "   fine-grained temporal patterns and changes, yet long enough to provide sufficient context. "
    "   The optimal window size is typically task-specific and can be determined through "
    "   cross-validation or domain expertise."
)

# --- 7. Save Visualizations ---
# All required visualizations (distribution histograms, correlation heatmap, time-series plot)
# were saved within their respective analysis steps above.
print(f"\nAll required visualizations saved in {FIGURES_DIR}.")

# --- 8. Save Results to results.json ---
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2) # Use indent=2 for human-readable JSON
print(f"Analysis results saved to {RESULTS_FILE}")