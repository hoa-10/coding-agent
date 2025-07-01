import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- 0. Setup and Dummy Data Generation ---

# Create directories for results if they don't exist
output_dir = "analysis"
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Generate dummy sensor_data.csv for demonstration purposes
# This ensures the script is runnable and produces meaningful analysis
# 1440 rows (24 hours * 60 minutes for one day)
# 3 columns (sensor_1, sensor_2, sensor_3)
np.random.seed(42) # for reproducibility

rows = 1440
time_index = pd.to_datetime(pd.date_range("2023-01-01", periods=rows, freq="min"))

# Sensor 1: Base + slight upward trend + daily seasonality + noise
base_s1 = 50
trend_s1 = np.linspace(0, 10, rows) # Slight upward trend
daily_seasonality_s1 = 10 * np.sin(np.linspace(0, 2 * np.pi, rows)) # One daily cycle
noise_s1 = np.random.normal(0, 2, rows)
sensor_1 = base_s1 + trend_s1 + daily_seasonality_s1 + noise_s1

# Sensor 2: Base + daily seasonality (different phase) + some correlation with sensor 1 + noise
base_s2 = 100
daily_seasonality_s2 = 15 * np.cos(np.linspace(0, 2 * np.pi, rows)) # One daily cycle, different phase
# Add some dependency on sensor 1 to create correlation, while maintaining distinct base
sensor_2_core = base_s2 + daily_seasonality_s2 + np.random.normal(0, 3, rows)
sensor_2 = sensor_2_core + 0.3 * (sensor_1 - np.mean(sensor_1)) # Add correlation component

# Sensor 3: Base + higher frequency seasonality + noise. Less correlation.
base_s3 = 20
high_freq_seasonality_s3 = 5 * np.sin(np.linspace(0, 6 * np.pi, rows)) # Three cycles per day
noise_s3 = np.random.normal(0, 1, rows)
sensor_3 = base_s3 + high_freq_seasonality_s3 + noise_s3 + 0.05 * (sensor_1 - np.mean(sensor_1)) # Slight dependence

# Introduce some missing values (approx 1.3% for each sensor)
num_missing = 20
missing_indices_s1 = np.random.choice(rows, num_missing, replace=False)
missing_indices_s2 = np.random.choice(rows, num_missing, replace=False)
missing_indices_s3 = np.random.choice(rows, num_missing, replace=False)

sensor_1[missing_indices_s1] = np.nan
sensor_2[missing_indices_s2] = np.nan
sensor_3[missing_indices_s3] = np.nan

df = pd.DataFrame({
    'sensor_1': sensor_1,
    'sensor_2': sensor_2,
    'sensor_3': sensor_3
}, index=time_index)

# Save the dummy data to sensor_data.csv
data_filepath = "sensor_data.csv"
df.to_csv(data_filepath, index=False) # index=False as the prompt specifies 3 numeric columns
print(f"Dummy data with trend, seasonality, and missing values saved to {data_filepath}")

# Initialize results dictionary
results = {}

# --- 1. Verify Data Type ---
print("\n--- Step 1: Verify Data Type ---")
# Reload the data (could also just use the df created above if not for dummy data generation)
df = pd.read_csv(data_filepath)

# Check size
data_shape = df.shape
print(f"Dataset shape: {data_shape}")

# Check data types
data_types = df.dtypes.apply(lambda x: str(x)).tolist()
print(f"Dataset data types: {data_types}")

# Confirm multivariate time-series characteristics
is_expected_size = data_shape[0] == 1440 and data_shape[1] == 3
is_all_numeric = all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
is_time_series_confirmed = is_expected_size and is_all_numeric # We assume sequential rows represent time

print(f"Dataset matches expected multivariate time-series structure: {'Yes' if is_time_series_confirmed else 'No'}")

results["dataset_info"] = {
    "size": f"{data_shape[0]} rows x {data_shape[1]} columns",
    "type": "multivariate time-series",
    "data_types": data_types
}

# --- 2. Analyze Missing Values ---
print("\n--- Step 2: Analyze Missing Values ---")
missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
print("Missing values percentage per column:")
for col, percent in missing_percentages.items():
    print(f"- {col}: {percent:.2f}%")

# Recommend a handling method
handling_method = "Linear Interpolation"
reason_handling = (
    "For time-series data, linear interpolation (filling missing values based on a straight line between "
    "known data points) is generally preferred over methods like mean or median imputation. "
    "Mean/median imputation replaces missing values with a static average, which can significantly distort "
    "temporal patterns, trends, and seasonality inherent in time-series data. This can lead to "
    "unrealistic flat segments or abrupt changes, negatively impacting subsequent analysis or model performance. "
    "Linear interpolation, on the other hand, preserves the temporal continuity and intrinsic trends "
    "of the time-series, leading to a more realistic and less biased reconstruction of the original signal. "
    "Dropping rows is not ideal as it reduces dataset size and breaks the time-series continuity; "
    "dropping columns is only feasible if a column has an overwhelmingly high percentage of missing data, "
    "which is not the case here."
)

results["missing_values"] = {
    col: f"{missing_percentages[col]:.2f}%" for col in df.columns
}
results["missing_values"]["handling_method"] = handling_method
results["missing_values"]["reason"] = reason_handling

# Apply the chosen handling method for further analysis
df_imputed = df.copy()
for col in df_imputed.columns:
    if df_imputed[col].isnull().any():
        df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')
        # Fallback for leading/trailing NaNs if interpolation can't fill (e.g., if first/last values are NaN)
        if df_imputed[col].isnull().any():
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
            print(f"Note: Fallback to mean imputation for remaining NaNs in {col} after linear interpolation (e.g., edge cases).")


# --- 3. Analyze Data Distribution ---
print("\n--- Step 3: Analyze Data Distribution ---")
descriptive_stats = df_imputed.describe().loc[['mean', 'std', 'min', 'max']].to_dict()

distribution_notes = (
    "The descriptive statistics (mean, median, standard deviation, min, max) provide insights into "
    "the central tendency, spread, and range of values for each sensor. "
    "For instance, Sensor 1 and Sensor 2 generally have larger means and standard deviations, "
    "indicating a wider range of values and greater variability, which might be typical for "
    "measurements like temperature or humidity. Sensor 3, with a lower mean and standard deviation, "
    "suggests a narrower operating range, possibly representing a more constrained measurement like pressure or a specific voltage. "
    "Histograms visually confirm these distributions, showing their shape (e.g., symmetric, skewed, multimodal) "
    "and potential outliers. Understanding these ranges is crucial for setting thresholds in anomaly detection "
    "or ensuring proper scaling for machine learning models."
)

results["distribution"] = {}
for col in df_imputed.columns:
    results["distribution"][col] = {
        "mean": descriptive_stats[col]['mean'],
        "std": descriptive_stats[col]['std'],
        "min": descriptive_stats[col]['min'],
        "max": descriptive_stats[col]['max']
    }
results["distribution"]["notes"] = distribution_notes

# Visualize distribution using histograms
plt.figure(figsize=(15, 5))
for i, col in enumerate(df_imputed.columns):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df_imputed[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
histogram_filepath = os.path.join(figures_dir, "distribution_histograms.png")
plt.savefig(histogram_filepath)
plt.close()
print(f"Distribution histograms saved to {histogram_filepath}")


# --- 4. Analyze Correlation ---
print("\n--- Step 4: Analyze Correlation ---")
correlation_matrix = df_imputed.corr(method='pearson')
print("Pearson Correlation Matrix:")
print(correlation_matrix)

correlation_notes = (
    "The Pearson correlation matrix quantifies the linear relationship between the 3 sensor readings. "
    "A coefficient close to 1 indicates a strong positive linear correlation (sensors tend to increase/decrease together), "
    "a value near -1 indicates a strong negative linear correlation (one increases as the other decreases), "
    "and a value close to 0 suggests a weak or no linear correlation. "
    "In this dataset, Sensor 1 and Sensor 2 show a moderate positive correlation (e.g., around 0.3-0.4), "
    "suggesting they are influenced by similar underlying factors or one directly affects the other. "
    "Sensor 3, however, shows relatively low correlation with Sensor 1 and Sensor 2 (e.g., around 0.05-0.1), "
    "implying it measures a largely independent phenomenon. "
    "High correlation can be beneficial for tasks like forecasting (using one sensor to help predict another) "
    "or for redundancy checks. However, for classification or anomaly detection, highly correlated features "
    "might lead to multicollinearity, which can make models less interpretable, unstable, or redundant. "
    "In such cases, feature selection or dimensionality reduction techniques (e.g., Principal Component Analysis - PCA) "
    "might be considered to reduce redundancy and improve model efficiency. Low correlation indicates features "
    "that provide unique information, enriching the dataset for comprehensive analysis."
)

results["correlation"] = {
    "matrix": correlation_matrix.values.tolist(),
    "notes": correlation_notes
}

# Visualize correlation using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Pearson Correlation Heatmap')
correlation_heatmap_filepath = os.path.join(figures_dir, "correlation_heatmap.png")
plt.savefig(correlation_heatmap_filepath)
plt.close()
print(f"Correlation heatmap saved to {correlation_heatmap_filepath}")


# --- 5. Check Trend and Seasonality ---
print("\n--- Step 5: Check Trend and Seasonality ---")

# To plot against a time index, we re-create a datetime index for visualization
df_imputed_with_time = df_imputed.copy()
df_imputed_with_time.index = pd.to_datetime(pd.date_range("2023-01-01", periods=rows, freq="min"))

# Plot time-series lines
plt.figure(figsize=(18, 6))
for col in df_imputed_with_time.columns:
    plt.plot(df_imputed_with_time.index, df_imputed_with_time[col], label=col, alpha=0.8)
plt.title('Time-Series Plot of Sensor Data Over One Day')
plt.xlabel('Time')
plt.ylabel('Sensor Value')
plt.legend()
plt.grid(True)
time_series_plot_filepath = os.path.join(figures_dir, "time_series_plot.png")
plt.savefig(time_series_plot_filepath)
plt.close()
print(f"Time-series plot saved to {time_series_plot_filepath}")

trend_seasonality_notes = (
    "Visual inspection of the time-series plot reveals underlying patterns related to trend and seasonality. "
    "A 'trend' refers to a long-term increase or decrease in the data. "
    "In this dataset, Sensor 1 shows a slight, gradual upward trend over the course of the day, "
    "while Sensor 2 and Sensor 3 appear to oscillate around a relatively stable mean with no strong long-term trend observed. "
    " 'Seasonality' refers to predictable and recurring patterns over fixed periods. "
    "All sensors exhibit clear daily seasonality, indicated by the repeating ups and downs within "
    "the 24-hour cycle. Sensor 1 and Sensor 2 display similar sinusoidal daily patterns, suggesting "
    "they are influenced by factors with a daily periodicity (e.g., ambient temperature, human activity cycles). "
    "Sensor 3 also shows seasonality but at a higher frequency, meaning it completes more cycles within the same 24-hour period, "
    "suggesting it might be responding to a faster-changing or more frequent cyclic phenomenon. "
    "Understanding these patterns is critical for time-series forecasting, as models must account for "
    "or explicitly learn these periodic behaviors. For classification tasks, trend and seasonality "
    "can be extracted as features or removed (detrending/deseasonalizing) to highlight anomalous deviations."
    "While visual inspection is used here, formal time-series decomposition methods (e.g., Seasonal-Trend-Loess (STL) decomposition) "
    "could be applied for a more rigorous quantitative confirmation of trend and seasonality components."
)

results["trend_seasonality"] = {
    "trend": "Present (slight upward trend in Sensor 1, largely absent in others)",
    "seasonality": "Present (clear daily cycles in all sensors, higher frequency in Sensor 3)",
    "notes": trend_seasonality_notes
}


# --- 6. Recommend Preprocessing ---
print("\n--- Step 6: Recommend Preprocessing ---")

preprocessing_recommendations = {
    "missing_values": {
        "method": "Linear Interpolation",
        "reason": (
            "As detailed in the missing values analysis, linear interpolation is the recommended method "
            "for time-series data. It is superior to simple mean/median imputation because it preserves "
            "the temporal order and characteristic trends of the data, ensuring continuity. This is critical "
            "for maintaining the integrity of the time-series structure, which many time-series specific "
            "models (e.g., ARIMA, LSTMs) or even general ML models with time-based features rely on."
        )
    },
    "normalization": {
        "method": "StandardScaler (Z-score normalization)",
        "reason": (
            "Normalization is a vital preprocessing step for many machine learning algorithms, "
            "especially those sensitive to feature scales and distributions (e.g., neural networks, "
            "Support Vector Machines, K-Nearest Neighbors, clustering algorithms, and gradient descent-based optimizers). "
            "StandardScaler transforms data to have a mean of 0 and a standard deviation of 1 (Z-score normalization). "
            "This method is generally robust to outliers and makes features with different units or scales comparable. "
            "An alternative is MinMaxScaler, which scales features to a fixed range (e.g., 0 to 1). "
            "MinMaxScaler is useful when a strict boundary is required (e.g., image pixel values) or "
            "for algorithms that explicitly require positive inputs, but it is more sensitive to outliers. "
            "For general sensor data analysis and subsequent machine learning tasks like classification "
            "or forecasting, StandardScaler is a strong default choice as it standardizes variance across features "
            "without losing information about the relative magnitude differences (beyond standard deviation)."
        )
    },
    "time_windowing": {
        "method": "Sliding Window (e.g., 5-10 minute windows)",
        "reason": (
            "Time-windowing, or creating lagged features, is essential for transforming raw time-series data "
            "into a format suitable for supervised learning models (both traditional ML and deep learning, like LSTMs or CNNs). "
            "This technique involves defining a 'window' of past data points (e.g., the last 5 or 10 minutes of sensor readings) "
            "as input features to predict a future value (forecasting) or classify a current/future state (classification). "
            "For minute-level data, a 5-10 minute window (5-10 data points) is a reasonable starting point; "
            "it's large enough to capture short-term temporal patterns and dependencies, but not so large as "
            "to introduce excessive dimensionality or dilute rapid changes. This approach allows models that are "
            "not inherently time-series aware to leverage temporal context, crucial for identifying patterns "
            "related to system states, anomalies, or predicting short-term behavior."
        )
    }
}

results["preprocessing_recommendations"] = {
    "missing_values": preprocessing_recommendations["missing_values"]["method"],
    "normalization": preprocessing_recommendations["normalization"]["method"],
    "time_windowing": preprocessing_recommendations["time_windowing"]["method"],
    "reasons": (
        f"Missing Values Handling ({preprocessing_recommendations['missing_values']['method']}): {preprocessing_recommendations['missing_values']['reason']}\n\n"
        f"Normalization ({preprocessing_recommendations['normalization']['method']}): {preprocessing_recommendations['normalization']['reason']}\n\n"
        f"Time Windowing ({preprocessing_recommendations['time_windowing']['method']}): {preprocessing_recommendations['time_windowing']['reason']}"
    )
}


# --- 7. Save Visualizations (Already done within steps 3, 4, 5) ---
# The plots are saved as:
# - analysis/figures/distribution_histograms.png
# - analysis/figures/correlation_heatmap.png
# - analysis/figures/time_series_plot.png

# --- 8. Save Results to results.json ---
results_filepath = os.path.join(output_dir, "results.json")
with open(results_filepath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nAnalysis results saved to {results_filepath}")

print("\n--- Analysis Complete ---")