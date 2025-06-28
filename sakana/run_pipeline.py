# run_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplo t as plt
import json
import os

# --- Configuration ---
DATA_PATH = '../sensor_data.csv'
LOOK_BACK_WINDOW = 10  # minutes
FORECAST_HORIZON = 10  # minutes
TRAIN_SPLIT_RATIO = 0.8 # 80% for training, 20% for testing

# Model Hyperparameters
LSTM_UNITS = 60       # Units for the LSTM layer (between 50-100)
DENSE_UNITS = 25      # Units for the hidden Dense layer
EPOCHS = 80           # Number of training epochs (between 50-100)
BATCH_SIZE = 32       # Batch size for training (between 32-64)
EARLY_STOPPING_PATIENCE = 10 # Patience for EarlyStopping callback

# Output Directories and Files
OUTPUT_FIGURES_DIR = 'figures'
OUTPUT_MODELS_DIR = 'models'
OUTPUT_RESULTS_FILE = 'results.json'

# --- Setup Output Directories ---
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

print("Starting Sensor Data Time-Series Forecasting Pipeline...")
print(f"Configuration: Look-back Window={LOOK_BACK_WINDOW} min, Forecast Horizon={FORECAST_HORIZON} min")

# 1. Setup and Data Loading
print(f"\n1. Loading data from '{DATA_PATH}'...")
try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    print(f"Initial DataFrame shape: {df.shape}")
    print("First 5 rows of data:\n", df.head())
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}.")
    print("Please ensure 'sensor_data.csv' is located in the parent directory.")
    exit()

# Identify sensor columns for processing
SENSOR_COLS = ['SensorA', 'SensorB', 'SensorC']
NUM_FEATURES = len(SENSOR_COLS)

# 2. Data Preprocessing
print("\n2. Preprocessing data...")
# Convert non-numeric entries to NaN
for col in SENSOR_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(f"Missing values after coercing non-numeric:\n{df[SENSOR_COLS].isnull().sum()}")

# Apply linear interpolation to fill NaN values
print("Applying linear interpolation for missing values...")
df[SENSOR_COLS] = df[SENSOR_COLS].interpolate(method='linear', limit_direction='both', axis=0)
print(f"Missing values after interpolation:\n{df[SENSOR_COLS].isnull().sum()}")
print("Interpolation complete.")

# Feature Scaling: Fit StandardScaler on the entire dataset and transform.
# This scaler will be used later for inverse transformation.
print("Scaling sensor data using StandardScaler...")
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[SENSOR_COLS]), columns=SENSOR_COLS, index=df.index)
print("Data scaled successfully.")
print("First 5 rows of scaled data:\n", df_scaled.head())


# 3. Data Preparation for LSTM (Windowing)
print(f"\n3. Preparing data for LSTM (windowing) with LOOK_BACK_WINDOW={LOOK_BACK_WINDOW} and FORECAST_HORIZON={FORECAST_HORIZON}...")
X, y = [], []

# Iterate through the scaled data to create input sequences (X) and target values (y)
# The loop runs until there's enough data for a full look-back window AND a future forecast horizon.
for i in range(len(df_scaled) - LOOK_BACK_WINDOW - FORECAST_HORIZON + 1):
    # X: Input sequence (LOOK_BACK_WINDOW minutes of all sensor readings)
    # The window starts at 'i' and ends at 'i + LOOK_BACK_WINDOW - 1'.
    X.append(df_scaled.iloc[i : i + LOOK_BACK_WINDOW][SENSOR_COLS].values)

    # y: Target value (SensorA reading at 'FORECAST_HORIZON' minutes into the future
    # from the end of the input window)
    # The end of the input window is at index `i + LOOK_BACK_WINDOW - 1`.
    # The target index is `(i + LOOK_BACK_WINDOW - 1) + FORECAST_HORIZON`.
    y.append(df_scaled.iloc[i + LOOK_BACK_WINDOW - 1 + FORECAST_HORIZON]['SensorA'])

X = np.array(X)
y = np.array(y)

print(f"Shape of X (input sequences): {X.shape} (samples, look_back_window, num_features)")
print(f"Shape of y (target values): {y.shape} (samples,)")


# 4. Train/Test Split (Chronological)
print(f"\n4. Splitting data into training and testing sets ({TRAIN_SPLIT_RATIO*100}% train, {1-TRAIN_SPLIT_RATIO:.0%}% test)...")
train_size = int(len(X) * TRAIN_SPLIT_RATIO)

X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

print(f"Train set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")


# 5. Scaling Training and Test Data
# This step was implicitly handled by scaling the entire DataFrame (df_scaled) before windowing.
# X_train, y_train, X_test, y_test are already in their scaled form.


# 6. Model Building (LSTM)
print("\n6. Building LSTM model...")
model = Sequential([
    # LSTM layer: input_shape matches (LOOK_BACK_WINDOW, NUM_FEATURES)
    # return_sequences=False because we predict a single output value (not a sequence)
    LSTM(LSTM_UNITS, input_shape=(LOOK_BACK_WINDOW, NUM_FEATURES), return_sequences=False),
    # Optional Dense hidden layer with ReLU activation
    Dense(DENSE_UNITS, activation='relu'),
    # Output layer: 1 unit for single-value regression, linear activation for continuous output
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['mean_squared_error', 'mean_absolute_error'])

print("Model Summary:")
model.summary()


# 7. Model Training
print(f"\n7. Training model for {EPOCHS} epochs with batch_size={BATCH_SIZE}...")
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15, # Use a portion of the training data for validation
    callbacks=[early_stopping],
    verbose=1
)
print("Model training complete.")


# 8. Model Evaluation
print("\n8. Evaluating model performance on the test set...")
predictions_scaled = model.predict(X_test)

# --- Inverse Scaling ---
# Both y_test (actual values) and predictions are in scaled form.
# To calculate metrics and plot in the original scale, they must be inverse transformed.
# The scaler was fitted on a 3-feature array (SensorA, SensorB, SensorC).
# To inverse transform a single column (SensorA, which is at index 0), we must
# create a temporary 3-feature array, place the single column values into the
# SensorA position (index 0), then inverse_transform, and finally extract index 0.

# Create dummy arrays of shape (num_samples, NUM_FEATURES)
dummy_y_test_scaled = np.zeros((len(y_test), NUM_FEATURES))
dummy_predictions_scaled = np.zeros((len(predictions_scaled), NUM_FEATURES))

# Place the scaled SensorA values/predictions into the first column (index 0)
dummy_y_test_scaled[:, 0] = y_test
dummy_predictions_scaled[:, 0] = predictions_scaled.flatten() # Flatten predictions (N,1) to (N,)

# Inverse transform using the fitted scaler
y_test_original = scaler.inverse_transform(dummy_y_test_scaled)[:, 0]
predictions_original = scaler.inverse_transform(dummy_predictions_scaled)[:, 0]

# Calculate metrics on original scale
mae = mean_absolute_error(y_test_original, predictions_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))

print(f"\n--- Model Evaluation Results (Original Scale) ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("--------------------------------------------------")


# 9. Output Generation

# Save Results to JSON
results_filepath = OUTPUT_RESULTS_FILE
print(f"\nSaving evaluation results to '{results_filepath}'...")
results = {"MAE": mae, "RMSE": rmse}
with open(results_filepath, 'w') as f:
    json.dump(results, f, indent=4)
print("Results saved.")

# Save Plot: Actual vs. Predicted SensorA Values
plot_filepath = os.path.join(OUTPUT_FIGURES_DIR, 'predicted_vs_actual_sensorA.png')
print(f"Saving prediction plot to '{plot_filepath}'...")
plt.figure(figsize=(15, 7))
plt.plot(y_test_original, label='Actual SensorA Value', alpha=0.8)
plt.plot(predictions_original, label='Predicted SensorA Value', alpha=0.8)
plt.title(f'SensorA Actual vs. Predicted Values (Test Set)\nMAE: {mae:.2f}, RMSE: {rmse:.2f}')
plt.xlabel('Time Index (Test Set Samples)')
plt.ylabel('SensorA Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_filepath)
plt.close()
print("Plot saved.")

# Save Model
model_filepath = os.path.join(OUTPUT_MODELS_DIR, 'lstm_forecaster.h5')
print(f"Saving trained Keras model to '{model_filepath}'...")
model.save(model_filepath)
print("Model saved.")

print("\nPipeline execution complete. Check 'figures/' and 'models/' directories, and 'results.json'.")