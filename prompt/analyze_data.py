def analyze_dataset_prompt(data):
    """
    Generate a prompt for analyzing a dataset
    """
    return f"""
**Instructions for Coder LLM: Analyze Sensor Dataset and Save Results**

You are tasked with writing a Python script to analyze a multivariate time-series dataset at `{data}`, using code to compute and visualize characteristics, and combining LLM reasoning to draw general conclusions. Results must be saved in a `results.json` file with a comprehensive dataset description, suitable for various sensor types and tasks. The dataset has:
- **Structure**: 1,440 rows (minute-level data for one day), 3 columns (continuous numeric values from 3 sensors).

---

### Task
Write a Python script using pandas, numpy, matplotlib, and seaborn to analyze the dataset generally. The script must:
- Compute characteristics (missing values, distribution, correlation, trend, seasonality).
- Visualize to support analysis.
- Use LLM reasoning to explain findings.
- Save results in `results.json` with a complete dataset description.

**Steps to Perform**:

1. **Verify Data Type**
   - Check size (1,440 rows, 3 columns) and data types (numeric) using pandas.
   - Confirm the dataset is a multivariate time-series.

2. **Analyze Missing Values**
   - Calculate the percentage of missing values per column.
   - Recommend a handling method (e.g., linear interpolation or mean imputation) and explain why.

3. **Analyze Data Distribution**
   - Compute descriptive statistics (mean, median, std, min, max) for each column.
   - Visualize distribution using histograms or boxplots to understand value ranges and potential outliers.

4. **Analyze Correlation**
   - Compute the Pearson correlation matrix for the 3 sensor columns.
   - Comment on correlation levels (high/low) and implications for potential tasks.

5. **Check Trend and Seasonality**
   - Plot time-series lines for each sensor to identify trends (increasing/decreasing) or seasonality (cycles).
   - Optionally use time-series decomposition (e.g., STL) to confirm trends and seasonality.

6. **Recommend Preprocessing**
   - Suggest general preprocessing steps for sensor data:
     - Handle missing values (e.g., linear interpolation or mean imputation).
     - Normalize (e.g., StandardScaler or MinMaxScaler).
     - Apply time-windowing (e.g., 5-10 minute windows for temporal patterns).
   - Explain why each step is suitable for various tasks (e.g., classification, forecasting).

7. **Save Visualizations**
   - Save at least two plots:
     - Time-series plot for all 3 sensors.
     - Correlation heatmap or distribution histogram.
   - Save as `.png` files in `analysis/figures/` (create if not exists).

8. **Save Results to `results.json`**
   - Save in JSON format with the structure:
     ```json
     
       "dataset_info": 
         "size": "1440 rows x 3 columns",
         "type": "multivariate time-series",
         "data_types": ["numeric", "numeric", "numeric"]
       ,
       "missing_values": 
         "sensor_1": "<percent>%",
         "sensor_2": "<percent>%",
         "sensor_3": "<percent>%",
         "handling_method": "<e.g., linear interpolation>",
         "reason": "<why chosen, e.g., maintains temporal continuity>"
       ,
       "distribution": 
         "sensor_1": "mean": <value>, "std": <value>, "min": <value>, "max": <value>,
         "sensor_2": "mean": <value>, "std": <value>, "min": <value>, "max": <value>,
         "sensor_3": "mean": <value>, "std": <value>, "min": <value>, "max": <value>,
         "notes": "<e.g., sensor 1 has wider range>"
       ,
       "correlation": 
         "matrix": [[1.0, <corr12>, <corr13>], [<corr21>, 1.0, <corr23>], [<corr31>, <corr32>, 1.0]],
         "notes": "<e.g., moderate correlation between sensor 1 and 2>"
       ,
       "trend_seasonality": 
         "trend": "<present/absent, e.g., no clear trend>",
         "seasonality": "<present/absent, e.g., daily cycle>",
         "notes": "<details, e.g., 24-hour periodicity observed>"
       ,
       "preprocessing_recommendations": 
         "missing_values": "<method>",
         "normalization": "<e.g., StandardScaler>",
         "time_windowing": "<e.g., 10-minute windows>",
         "reasons": "<why chosen, e.g., suitable for time-series tasks>"
       
     
     ```

---

### Reasoning Requirements
- **Compare Methods**: For each step (e.g., missing value handling, normalization), compare at least 2 methods (e.g., linear interpolation vs. mean imputation) and explain the chosen method in `results.json`.
- **Generalize**: Ensure analysis is broad, applicable to various sensor types (e.g., temperature, pressure) and tasks (e.g., classification, forecasting).
- **Self-Reflection**: Review choices (e.g., is a 10-minute window suitable for all tasks?) and adjust if needed.
- **Interpret Findings**: Comment on characteristics (e.g., how high correlation might impact modeling) in `results.json`.

---

### Notes
- Use pandas, numpy, matplotlib, and seaborn for analysis and visualization.
- Save plots in `analysis/figures/` and `results.json` in `analysis/`.
- Ensure `results.json` is well-structured and comprehensive for general sensor data.
- Focus on analyzing dataset characteristics, not a specific task like anomaly detection.
"""