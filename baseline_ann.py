import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================================
# 1. REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data" / "teleconnect_clean.csv"

results_dir = project_dir / "results"
models_dir = project_dir / "models"

results_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD DATA
# ==========================================================

df = pd.read_csv(data_path)

print("=" * 65)
print("OTOMOTO BASELINE ARTIFICIAL NEURAL NETWORK")
print("=" * 65)

print("\nDataset shape:", df.shape)

# Encode target: No = 0, Yes = 1
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in the Churn column.")

X = df.drop(columns=["Churn"])

print("\nTarget distribution:")
print(y.value_counts().sort_index())


# ==========================================================
# 3. IDENTIFY FEATURE TYPES
# ==========================================================

categorical_columns = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:", len(categorical_columns))
print("Numerical features:", len(numerical_columns))


# ==========================================================
# 4. SPLIT THE DATA
# ==========================================================

# 70% training, 15% validation, 15% testing.

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp,
)

print("\nDataset split:")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))


# ==========================================================
# 5. PREPROCESS THE FEATURES
# ==========================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ]
)

# Fit preprocessing on training data only.
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrices to dense arrays for TensorFlow.
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_val_processed, "toarray"):
    X_val_processed = X_val_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)
X_val_processed = np.asarray(
    X_val_processed, dtype=np.float32
)
X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

y_train_array = y_train.to_numpy(dtype=np.float32)
y_val_array = y_val.to_numpy(dtype=np.float32)


print("\nProcessed feature count:", X_train_processed.shape[1])


# ==========================================================
# 6. BUILD THE BASELINE ANN
# ==========================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(X_train_processed.shape[1],)
        ),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    ],
)

print("\nBaseline model architecture:")
model.summary()


# ==========================================================
# 7. TRAIN THE MODEL
# ==========================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    X_train_processed,
    y_train_array,
    validation_data=(X_val_processed, y_val_array),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)


# ==========================================================
# 8. EVALUATE THE BASELINE MODEL
# ==========================================================

y_probability = model.predict(
    X_test_processed,
    verbose=0,
).ravel()

# Use 0.5 as the initial classification threshold.
y_predicted = (y_probability >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_predicted)

precision = precision_score(
    y_test, y_predicted, zero_division=0
)

recall = recall_score(
    y_test, y_predicted, zero_division=0
)

f1 = f1_score(
    y_test, y_predicted, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_probability)

test_loss = float(
    tf.keras.losses.binary_crossentropy(
        y_test.to_numpy(dtype=np.float32),
        y_probability,
    ).numpy().mean()
)

print("\n" + "=" * 65)
print("BASELINE ANN TEST RESULTS")
print("=" * 65)

print(f"Test loss:  {test_loss:.4f}")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-score:   {f1:.4f}")
print(f"ROC-AUC:    {roc_auc:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_predicted))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_predicted,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
)


# ==========================================================
# 9. SAVE THE BASELINE MODEL AND RESULTS
# ==========================================================

model.save(models_dir / "baseline_ann.keras")

baseline_results = {
    "model": "Baseline ANN",
    "optimizer": "Adam",
    "test_loss": float(test_loss),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "classification_threshold": 0.5,
    "training_records": int(len(X_train)),
    "validation_records": int(len(X_val)),
    "test_records": int(len(X_test)),
    "processed_feature_count": int(X_train_processed.shape[1]),
}

with open(
    results_dir / "baseline_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(baseline_results, file, indent=4)

# Save the fitted preprocessing pipeline separately.
import joblib

joblib.dump(
    preprocessor,
    models_dir / "baseline_preprocessor.joblib",
)

print("\nSaved baseline model to:")
print(models_dir / "baseline_ann.keras")

print("\nSaved evaluation results to:")
print(results_dir / "baseline_metrics.json")

print("\nSaved preprocessing pipeline to:")
print(models_dir / "baseline_preprocessor.joblib")

print("\nBaseline ANN experiment completed successfully.")