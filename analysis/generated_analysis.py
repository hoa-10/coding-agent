import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- 0. Setup: Create synthetic data and directories ---
# This section generates a dummy sensor_data.csv to make the script runnable
# and demonstrate its functionality. In a real scenario, sensor_data.csv
# would already exist.

# Define paths for outputs
output_dir = "analysis"
figures_dir = os.path.join(output_dir, "figures")
results_path = os.path.join(output_dir, "results.json")
sensor_data_path = "sensor_data.csv"

# Create output directories if they don't exist
os.makedirs(figures_dir, exist_ok=True)

# Generate synthetic sensor data for 1440 minutes (1 day)
np.random.seed(42) # for reproducibility
time_points = 1440 # minute-level data for one day

# Sensor 1: Base value + daily sine wave + noise (simulating temperature)
t = np.linspace(0, 2 * np.pi, time_points) # Represents 24 hours of a cycle
sensor1_data = 25 + 5 * np.sin(t) + np.random.normal(0, 1.5, time_points)

# Sensor 2: Base value + increasing trend + daily sine wave (phase shifted) + noise (simulating pressure)
sensor2_data = 1000 + 0.1 * np.arange(time_points) + 50 * np.sin(t + np.pi/4) + np.random.normal(0, 5, time_points)

# Sensor 3: Correlated with sensor 1, but with a different range and noise (simulating humidity related to temperature)
sensor3_data = 0.7 * sensor1_data + 30 + np.random.normal(0, 2, time_points)
sensor3_data = np.clip(sensor3_data, 10, 95) # Keep within a reasonable humidity range

# Introduce some missing values randomly
for _ in range(30): # Add 30 NaNs spread out (approx. 0.7% missing)
    row = np.random.randint(0, time_points)
    col = np.random.randint(0, 3)
    if col == 0: sensor1_data[row] = np.nan
    elif col == 1: sensor2_data[row] = np.nan
    else: sensor3_data[row] = np.nan

data = pd.DataFrame({
    'sensor_1': sensor1_data,
    'sensor_2': sensor2_data,
    'sensor_3': sensor3_data
})

# Save the synthetic data to sensor_data.csv
data.to_csv(sensor_data_path, index=False)
print(f"Synthetic '{sensor_data_path}' created for analysis.")

# Initialize results dictionary to store all analysis findings
results = {}

# --- 1. Verify Data Type ---
df = pd.read_csv(sensor_data_path)

dataset_info = {
    "size": f"{df.shape[0]} rows x {df.shape[1]} columns",
    "type": "multivariate time-series",
    "data_types": [str(df[col].dtype) for col in df.columns]
}
results["dataset_info"] = dataset_info

print("\n--- 1. Data Verification ---")
print(f"Dataset shape: {df.shape}")
print(f"Data types:\n{df.dtypes}")
print(f"Dataset info: {results['dataset_info']}")

# --- 2. Analyze Missing Values ---
missing_percentage = df.isnull().sum() / len(df) * 100
missing_values_info = {col: f"{missing_percentage[col]:.2f}%" for col in df.columns}

# LLM Reasoning: Comparison of missing value handling methods
# 1. Linear Interpolation: Fills missing values by drawing a straight line between the known points.
#    Pros: Maintains temporal continuity, essential for time-series data where the sequence of values holds meaning. Good for short, sporadic gaps.
#    Cons: Can be inaccurate if gaps are large, or if underlying data pattern is complex/non-linear.
# 2. Mean/Median Imputation: Fills missing values with the mean or median of the column.
#    Pros: Simple to implement, computationally inexpensive.
#    Cons: Distorts time-series patterns, reduces variance, and can lead to biased estimates by treating missing values as typical values, ignoring temporal context. Not ideal for trend or seasonality analysis.

chosen_handling_method = "linear interpolation"
chosen_reason = "For time-series data like sensor readings, linear interpolation is generally the superior method for handling missing values. It preserves the temporal sequence and estimates missing points based on the trends and values of adjacent data points, which is crucial for maintaining the integrity of time-dependent patterns (like daily cycles or trends). Unlike mean/median imputation, it does not artificially flatten data or introduce biases by ignoring the temporal context, making it more suitable for subsequent time-series analysis and modeling tasks (e.g., forecasting, anomaly detection) where temporal continuity is paramount. Given that the data is minute-level, interpolation over small gaps is likely to be accurate and represent the real physical process well."

missing_values_info.update({
    "handling_method": chosen_handling_method,
    "reason": chosen_reason
})
results["missing_values"] = missing_values_info

print("\n--- 2. Missing Values Analysis ---")
print(f"Missing values percentage per column:\n{missing_percentage.round(2)}")
print(f"Recommended handling: {chosen_handling_method} - {chosen_reason}")

# Impute missing values for subsequent analysis to ensure continuity
df_imputed = df.copy()
df_imputed.interpolate(method='linear', inplace=True)
# If there are any NaNs at the beginning or end after linear interpolation, fill with ffill/bfill
df_imputed.fillna(method='bfill', inplace=True) # Backward fill for leading NaNs
df_imputed.fillna(method='ffill', inplace=True) # Forward fill for trailing NaNs


# --- 3. Analyze Data Distribution ---
descriptive_stats = df_imputed.describe().loc[['mean', 'std', 'min', 'max']]
distribution_info = {}
notes_distribution = []

for col in df_imputed.columns:
    col_stats = descriptive_stats[col].to_dict()
    distribution_info[col] = {k: round(v, 2) for k, v in col_stats.items()}
    notes_distribution.append(f"Sensor '{col}' has a mean value of {col_stats['mean']:.2f} and a standard deviation of {col_stats['std']:.2f}, indicating its typical operating point and variability. The min ({col_stats['min']:.2f}) and max ({col_stats['max']:.2f}) values define the observed operating range for this sensor within the dataset.")

# Visualize distribution using histograms
plt.figure(figsize=(18, 5))
for i, col in enumerate(df_imputed.columns):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df_imputed[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(f'{col} Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "distribution_histograms.png"))
plt.close()

notes_distribution.append("Histograms provide a visual representation of how sensor values are distributed. They reveal the value ranges, central tendency, spread, and potential skewness or multimodal patterns. For example, a bell-shaped distribution suggests normal operation, while skewed distributions might indicate operational limits or specific environmental influences. Analyzing value ranges (min/max) is crucial for understanding sensor limitations and data quality. Significant differences in ranges between sensors highlight the need for normalization prior to many machine learning algorithms.")

results["distribution"] = {
    "sensor_1": distribution_info['sensor_1'],
    "sensor_2": distribution_info['sensor_2'],
    "sensor_3": distribution_info['sensor_3'],
    "notes": " ".join(notes_distribution)
}

print("\n--- 3. Data Distribution Analysis ---")
print("Descriptive statistics (after imputation):\n", descriptive_stats.round(2))
print("Distribution notes added to results.json. Histograms saved.")


# --- 4. Analyze Correlation ---
correlation_matrix = df_imputed.corr(method='pearson')
correlation_info = {
    "matrix": correlation_matrix.round(3).values.tolist()
}

notes_correlation = []
notes_correlation.append("Pearson correlation quantifies the linear relationship between pairs of sensor readings. A value close to 1 indicates a strong positive linear correlation, -1 indicates a strong negative linear correlation, and 0 indicates no linear correlation.")

# LLM Reasoning: Interpret correlation levels and implications
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) >= 0.7:
            notes_correlation.append(f"High correlation ({corr_val:.2f}) between {col1} and {col2}. This suggests that these sensors are likely measuring closely related physical phenomena or that one sensor's reading is strongly dependent on another. Implications: For modeling tasks (e.g., regression, classification), high correlation can lead to multicollinearity, which might make models unstable, harder to interpret, and inflate the variance of coefficient estimates. It also suggests potential redundancy; if one sensor is significantly more expensive or prone to failure, the other might serve as a viable proxy. Dimensionality reduction (e.g., PCA) or feature selection might be beneficial to reduce redundancy.")
        elif abs(corr_val) >= 0.3:
            notes_correlation.append(f"Moderate correlation ({corr_val:.2f}) between {col1} and {col2}. This indicates a noticeable linear relationship, where the sensors tend to move together to some extent, but are not perfectly aligned. Implications: These sensors likely capture related but distinct aspects. Both could provide valuable, complementary information for complex tasks like state classification (e.g., 'normal', 'warning', 'critical' states) or advanced forecasting, as they provide more nuanced insights than highly correlated pairs.")
        else:
            notes_correlation.append(f"Low correlation ({corr_val:.2f}) between {col1} and {col2}. This implies that the linear movements of these sensors are largely independent. Implications: These sensors are likely measuring entirely different physical properties or aspects of the environment. Each sensor provides unique information, which is beneficial for building robust predictive models or for gaining a comprehensive understanding of the system, as they contribute distinct features without significant overlap or redundancy.")

correlation_info["notes"] = " ".join(notes_correlation)
results["correlation"] = correlation_info

# Visualize correlation using a heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Pearson Correlation Matrix of Sensor Data', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)
plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"))
plt.close()

print("\n--- 4. Correlation Analysis ---")
print("Correlation matrix (after imputation):\n", correlation_matrix.round(2))
print("Correlation heatmap saved. Notes added to results.json.")


# --- 5. Check Trend and Seasonality ---
# Set a proper time index for time-series analysis and visualization
# Assuming minute-level data starting from 2023-01-01 00:00:00
df_ts = df_imputed.copy()
df_ts.index = pd.to_datetime('2023-01-01') + pd.to_timedelta(df_ts.index, unit='min')

trend_seasonality_info = {
    "trend": "Likely present in some sensors (e.g., a slight increase/decrease over the day), but not a sustained long-term trend due to single-day data.",
    "seasonality": "Visually observable daily cycle (24-hour period).",
    "notes": ""
}

notes_ts = []
notes_ts.append("Trend refers to the long-term upward or downward movement of the data. Seasonality refers to repeating patterns or cycles within a fixed and known period (e.g., daily, weekly, yearly).")

# Visualize time-series lines for each sensor
plt.figure(figsize=(16, 8))
for col in df_ts.columns:
    plt.plot(df_ts.index, df_ts[col], label=col, alpha=0.8, linewidth=1.5)
plt.title('Time Series Plot of Sensor Data (1 Minute Intervals - 24 Hours)', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Sensor Value', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "time_series_plot.png"))
plt.close()

# LLM Reasoning: Interpretation of Trend and Seasonality for a single day dataset
notes_ts.append("The time series plot allows for visual inspection of trends and seasonality. For a single day's worth of data (1440 minutes), a clear long-term trend (e.g., indicating sensor degradation) is unlikely to be fully established or generalized. However, shorter-term trends (e.g., values generally increasing or decreasing throughout that specific 24-hour period) can be observed. Daily seasonality, manifesting as a repeating pattern over a 24-hour cycle, is a common characteristic of many sensor types (e.g., temperature fluctuating with day/night). While this dataset only contains one full daily cycle, the characteristic shape of this cycle is visible and important for understanding the sensor behavior.")
notes_ts.append("For robust time-series decomposition (e.g., using STL), a dataset spanning multiple seasonal cycles (e.g., several days or weeks) is typically required to accurately separate *repeated* seasonal patterns from the trend and residual components. Given this dataset covers exactly one day (1440 minutes), applying decomposition for 'repeated' seasonality might not be meaningful or accurately reflect future cycles, as there's only one full period to analyze. Therefore, visual inspection is the primary and most reliable method for identifying a daily cycle within this specific dataset, and any overarching trend observed is specific to this single 24-hour period.")

trend_seasonality_info["notes"] = " ".join(notes_ts)
results["trend_seasonality"] = trend_seasonality_info

print("\n--- 5. Trend and Seasonality Check ---")
print("Time series plot saved. Notes added to results.json regarding visual inspection and decomposition limitations for this dataset size.")


# --- 6. Recommend Preprocessing ---
preprocessing_recommendations = {
    "missing_values": {
        "method": "Linear interpolation (followed by backward/forward fill for edges)",
        "reason": "This method is superior for time-series data because it preserves the temporal order and estimates missing points based on the values of surrounding known data points. This maintains the natural flow and trends of the sensor readings, which is critical for accurate analysis and modeling. Compared to simple methods like mean/median imputation, it avoids distorting the data's variance and temporal correlations, making it suitable for tasks like forecasting, anomaly detection, or system state classification."
    },
    "normalization": {
        "method": "StandardScaler (Z-score normalization)",
        "reason": "StandardScaler transforms data to have a mean of 0 and a standard deviation of 1. This is crucial when sensor features have widely different scales or units (e.g., temperature in Celsius vs. pressure in kPa). Many machine learning algorithms (e.g., SVMs, neural networks, k-Nearest Neighbors, PCA, clustering algorithms) are sensitive to feature scales and perform poorly if not normalized. Compared to MinMaxScaler (which scales to a fixed range like [0,1]), StandardScaler is generally preferred as it is less affected by outliers and preserves original distribution shapes better, making it robust for diverse sensor data and suitable for a wide array of tasks like anomaly detection (where deviation from the mean is key) or forecasting."
    },
    "time_windowing": {
        "method": "Sliding windows (e.g., 5-10 minute windows for minute-level data)",
        "reason": "Time-windowing (also known as sequence generation or rolling features) transforms raw continuous time-series data into fixed-size input vectors or sequences. This is essential for many supervised machine learning tasks, as it allows capturing temporal patterns within a defined time frame. For classification tasks (e.g., identifying a specific operational state or anomaly), a window can be used to extract aggregated features (e.g., mean, max, standard deviation, variance, slope, or even FFT components) over that period, which act as inputs for classification models. For forecasting, it creates input-output sequences for recurrent neural networks (RNNs like LSTMs) or traditional time-series models. A 10-minute window (10 data points) for minute-level data, for instance, provides a concise summary of recent dynamics, useful for capturing sub-hourly trends or transient events relevant to various sensor-based applications."
    },
    "reasons": "These preprocessing steps collectively prepare sensor data for robust machine learning analysis. Handling missing values ensures data completeness and integrity. Normalization standardizes feature scales, preventing algorithms from implicitly favoring features with larger numerical ranges. Time-windowing transforms the continuous data stream into discrete, meaningful samples that capture temporal dependencies, making the data suitable for both traditional and deep learning models across diverse tasks such as system state classification, predictive maintenance, anomaly detection, or forecasting of future sensor values."
}
results["preprocessing_recommendations"] = preprocessing_recommendations

print("\n--- 6. Preprocessing Recommendations ---")
print("Recommendations added to results.json.")


# --- 7. Save Results to results.json ---
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nAnalysis complete. Detailed results saved to '{results_path}'")
print(f"Visualizations saved to '{figures_dir}'")