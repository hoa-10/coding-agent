import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import json
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

try:
    data = np.load('pect_ndt_full_dataset.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_valid']
    y_valid = data['y_valid']
    m = data['m']
    st = data['st']
except FileNotFoundError:
    print("Error: 'pect_ndt_full_dataset.npz' not found. Please ensure the dataset file is in the same directory.")
    exit()
except KeyError as e:
    print(f"Error: Missing expected key in the dataset file: {e}. Please check the dataset integrity.")
    exit()

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

X_train_normalized = (X_train - m) / st
X_valid_normalized = (X_valid - m) / st

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(500, 1)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy',
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc')])

epochs = 30
batch_size = 64

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

history = model.fit(
    X_train_normalized,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_valid_normalized, y_valid),
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

loss, accuracy, precision, recall, auc_score_from_tf = model.evaluate(X_valid_normalized, y_valid, verbose=0)

y_pred_proba = model.predict(X_valid_normalized)
y_pred_binary = (y_pred_proba > 0.5).astype(int)

val_accuracy = accuracy_score(y_valid, y_pred_binary)
val_precision = precision_score(y_valid, y_pred_binary, pos_label=1)
val_recall = recall_score(y_valid, y_pred_binary, pos_label=1)
val_f1_score = f1_score(y_valid, y_pred_binary, pos_label=1)
val_roc_auc = roc_auc_score(y_valid, y_pred_proba)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision (Defect): {val_precision:.4f}")
print(f"Validation Recall (Defect): {val_recall:.4f}")
print(f"Validation F1-Score (Defect): {val_f1_score:.4f}")
print(f"Validation ROC-AUC: {val_roc_auc:.4f}")


results_data = {
    "metrics": {
        "accuracy": val_accuracy,
        "precision": val_precision,
        "recall": val_recall,
        "f1_score": val_f1_score,
        "roc_auc": val_roc_auc,
        "train_loss": history.history['loss'][-1],
        "train_accuracy": history.history['accuracy'][-1],
        "val_loss": history.history['val_loss'][-1],
        "val_accuracy": history.history['val_accuracy'][-1]
    },
    "hyperparameters": {
        "epochs": epochs,
        "batch_size": batch_size,
        "random_seed": RANDOM_SEED,
        "optimizer": "Adam",
        "early_stopping_patience": early_stopping.patience,
        "reduce_lr_factor": reduce_lr.factor,
        "reduce_lr_patience": reduce_lr.patience,
        "reduce_lr_min_lr": reduce_lr.min_lr
    },
    "model_architecture_description": "LSTM model with 2 LSTM layers (64 units, return_sequences=True; 32 units), with 0.2 dropout after each LSTM layer. Followed by 2 Dense layers (64, 32 units, relu activation) with 0.5 dropout after each. Final Dense layer with 1 unit and sigmoid activation."
}

with open(os.path.join(results_dir, 'results.json'), 'w') as f:
    json.dump(results_data, f, indent=4)

print(f"Results saved to '{results_dir}/results.json'")
