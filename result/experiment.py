import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)
import json

# Define dataset path
dataset_path = 'C:\\Users\\Admin\\Desktop\\coding-agent\\pect_ndt_full_dataset.npz'

print("Starting NDT Deep Learning Pipeline...")

# 1. Data Loading
print(f"Loading data from '{dataset_path}'...")
if not os.path.exists(dataset_path):
    print(f"Error: The dataset file '{dataset_path}' was not found.")
    print("Please ensure the file is in the correct directory.")
    exit()

try:
    data = np.load(dataset_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_valid']
    y_valid = data['y_valid']
    print("Data loaded successfully.")
except KeyError as e:
    print(f"Error: Missing expected key in the dataset file: {e}.")
    print("Please ensure 'X_train', 'y_train', 'X_valid', 'y_valid' keys are present.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the dataset: {e}")
    exit()

# Determine the task type (binary classification, multi-class classification, or regression)
unique_y_train = np.unique(y_train)
num_unique_y_train = len(unique_y_train)
is_integer_labels = np.issubdtype(y_train.dtype, np.integer)

task_type = None
num_classes = None

if is_integer_labels and num_unique_y_train == 2:
    task_type = "binary_classification"
    print(f"Detected task: Binary Classification (unique labels: {unique_y_train}).")
elif is_integer_labels and num_unique_y_train > 2:
    task_type = "multi_class_classification"
    num_classes = num_unique_y_train
    print(f"Detected task: Multi-class Classification (unique labels: {unique_y_train}, {num_classes} classes).")
else:
    task_type = "regression"
    print(f"Detected task: Regression (y_train dtype: {y_train.dtype}, {num_unique_y_train} unique values).")

# 2. Preprocessing
print("Applying feature scaling and target preprocessing...")

# Store original shapes for Keras input layer if needed
original_X_train_shape = X_train.shape
original_X_valid_shape = X_valid.shape

# Flatten for StandardScaler (StandardScaler expects 2D input: samples, features)
X_train_reshaped_for_scaler = X_train.reshape(X_train.shape[0], -1)
X_valid_reshaped_for_scaler = X_valid.reshape(X_valid.shape[0], -1)

scaler = StandardScaler()
X_train_scaled_flat = scaler.fit_transform(X_train_reshaped_for_scaler)
X_valid_scaled_flat = scaler.transform(X_valid_reshaped_for_scaler)

# Reshape scaled data for LSTM input: (samples, timesteps, features)
# Here, we treat each sample as a sequence of 1 timestep.
X_train_scaled = X_train_scaled_flat.reshape(X_train_scaled_flat.shape[0], 1, X_train_scaled_flat.shape[1])
X_valid_scaled = X_valid_scaled_flat.reshape(X_valid_scaled_flat.shape[0], 1, X_valid_scaled_flat.shape[1])

# Target Preprocessing
y_train_processed = y_train
y_valid_processed = y_valid

if task_type == "multi_class_classification":
    y_train_processed = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_valid_processed = tf.keras.utils.to_categorical(y_valid, num_classes=num_classes)

print("Preprocessing complete.")

# 3. Deep Learning Model Configuration
print("Configuring deep learning model...")

model = keras.Sequential()

# Determine input shape for LSTM layer: (timesteps, features)
# Based on the reshaping above, timesteps will be 1 and features will be total_features.
lstm_input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2]) # (1, total_features)

model.add(keras.layers.LSTM(128, activation='relu', input_shape=lstm_input_shape))
model.add(keras.layers.Dense(64, activation='relu')) # A Dense layer after LSTM output

# Output layer
output_activation = None
loss_function = None
metrics_list = []

if task_type == "binary_classification":
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    output_activation = 'sigmoid'
    loss_function = 'binary_crossentropy'
    metrics_list.append('accuracy')
elif task_type == "multi_class_classification":
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    output_activation = 'softmax'
    loss_function = 'categorical_crossentropy'
    metrics_list.append('accuracy')
elif task_type == "regression":
    model.add(keras.layers.Dense(1, activation='linear'))
    output_activation = 'linear'
    loss_function = 'mse'
    metrics_list.append('mae')

# Compile the model
model.compile(optimizer='adam', loss=loss_function, metrics=metrics_list)

epochs = 30
batch_size = 32

model_architecture_summary = []
model.summary(print_fn=lambda x: model_architecture_summary.append(x))
model_architecture_summary_str = "\n".join(model_architecture_summary)

print("Model configured:")
print(model_architecture_summary_str)
print(f"Hyperparameters: Epochs={epochs}, Batch Size={batch_size}")
print(f"Loss function: {loss_function}, Metrics: {metrics_list}")


# 4. Training Process
print("Training model...")
history = model.fit(
    X_train_scaled, y_train_processed,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_valid_scaled, y_valid_processed),
    verbose=1
)
print("Model training complete.")

# 5. Evaluation
print("Evaluating model...")

results = {}
evaluation_metrics = {}

# Keras evaluate results
keras_eval_results = model.evaluate(X_valid_scaled, y_valid_processed, verbose=0)
for i, metric_name in enumerate(model.metrics_names):
    evaluation_metrics[f"val_{metric_name}"] = keras_eval_results[i]

# Sklearn metrics for classification/regression
y_pred = model.predict(X_valid_scaled)

if task_type == "binary_classification":
    y_pred_binary = (y_pred > 0.5).astype(int)
    evaluation_metrics['accuracy'] = float(accuracy_score(y_valid, y_pred_binary))
    evaluation_metrics['precision'] = float(precision_score(y_valid, y_pred_binary, zero_division=0))
    evaluation_metrics['recall'] = float(recall_score(y_valid, y_pred_binary, zero_division=0))
    evaluation_metrics['f1_score'] = float(f1_score(y_valid, y_pred_binary, zero_division=0))
    evaluation_metrics['roc_auc'] = float(roc_auc_score(y_valid, y_pred))
    conf_matrix = confusion_matrix(y_valid, y_pred_binary)
    evaluation_metrics['confusion_matrix'] = conf_matrix.tolist() # Convert to list for JSON serialization

    print(f"Validation Loss: {evaluation_metrics['val_loss']:.4f}")
    print(f"Validation Accuracy (Keras): {evaluation_metrics['val_accuracy']:.4f}")
    print(f"Validation Accuracy (Sklearn): {evaluation_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {evaluation_metrics['precision']:.4f}")
    print(f"Validation Recall: {evaluation_metrics['recall']:.4f}")
    print(f"Validation F1-Score: {evaluation_metrics['f1_score']:.4f}")
    print(f"Validation ROC AUC: {evaluation_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

elif task_type == "multi_class_classification":
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_valid_classes = np.argmax(y_valid_processed, axis=1)
    evaluation_metrics['accuracy'] = float(accuracy_score(y_valid_classes, y_pred_classes))
    evaluation_metrics['precision_weighted'] = float(precision_score(y_valid_classes, y_pred_classes, average='weighted', zero_division=0))
    evaluation_metrics['recall_weighted'] = float(recall_score(y_valid_classes, y_pred_classes, average='weighted', zero_division=0))
    evaluation_metrics['f1_score_weighted'] = float(f1_score(y_valid_classes, y_pred_classes, average='weighted', zero_division=0))
    evaluation_metrics['roc_auc_ovr'] = float(roc_auc_score(y_valid_processed, y_pred, multi_class='ovr'))
    conf_matrix = confusion_matrix(y_valid_classes, y_pred_classes)
    evaluation_metrics['confusion_matrix'] = conf_matrix.tolist() # Convert to list for JSON serialization

    print(f"Validation Loss: {evaluation_metrics['val_loss']:.4f}")
    print(f"Validation Accuracy (Keras): {evaluation_metrics['val_accuracy']:.4f}")
    print(f"Validation Accuracy (Sklearn): {evaluation_metrics['accuracy']:.4f}")
    print(f"Validation Precision (weighted): {evaluation_metrics['precision_weighted']:.4f}")
    print(f"Validation Recall (weighted): {evaluation_metrics['recall_weighted']:.4f}")
    print(f"Validation F1-Score (weighted): {evaluation_metrics['f1_score_weighted']:.4f}")
    print(f"Validation ROC AUC (OVR): {evaluation_metrics['roc_auc_ovr']:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

elif task_type == "regression":
    evaluation_metrics['mae'] = float(mean_absolute_error(y_valid, y_pred))
    evaluation_metrics['mse'] = float(mean_squared_error(y_valid, y_pred))
    evaluation_metrics['r2_score'] = float(r2_score(y_valid, y_pred))
    print(f"Validation Loss (MSE): {evaluation_metrics['val_loss']:.4f}")
    print(f"Validation MAE (Keras): {evaluation_metrics['val_mae']:.4f}")
    print(f"Validation MAE (Sklearn): {evaluation_metrics['mae']:.4f}")
    print(f"Validation MSE: {evaluation_metrics['mse']:.4f}")
    print(f"Validation R2 Score: {evaluation_metrics['r2_score']:.4f}")

print("Model evaluation complete.")

# 6. Save Results
print("Saving results...")
results_filepath = 'results.json'

results['evaluation_metrics'] = evaluation_metrics
results['hyperparameters'] = {
    'epochs': epochs,
    'batch_size': batch_size,
    'optimizer': 'adam',
    'loss_function': loss_function,
    'model_metrics': metrics_list,
    'task_type': task_type
}
results['model_architecture_summary'] = model_architecture_summary_str
results['training_history'] = {key: [float(val) for val in values] for key, values in history.history.items()}

try:
    with open(results_filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved successfully to '{results_filepath}'.")
except Exception as e:
    print(f"Error saving results to JSON: {e}")

print("NDT Deep Learning Pipeline finished.")
