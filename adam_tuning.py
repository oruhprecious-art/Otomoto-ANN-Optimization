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
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "teleconnect_clean.csv"

RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in the Churn column.")

print("=" * 65)
print("ADAM HYPERPARAMETER TUNING")
print("=" * 65)


# ==========================================================
# 3. FIXED TRAINING, VALIDATION, AND TEST SPLITS
# ==========================================================

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

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))


# ==========================================================
# 4. PREPROCESS THE DATA
# ==========================================================

categorical_columns = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)


def dense_array(values):
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


X_train = dense_array(X_train)
X_val = dense_array(X_val)
X_test = dense_array(X_test)

y_train = y_train.to_numpy(dtype=np.float32)
y_val = y_val.to_numpy(dtype=np.int32)
y_test = y_test.to_numpy(dtype=np.int32)


# ==========================================================
# 5. BUILD THE SAME BASELINE ARCHITECTURE
# ==========================================================

def build_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
    )

    return model


# ==========================================================
# 6. TEST THREE LEARNING RATES
# ==========================================================

learning_rates = [0.0003, 0.001, 0.003]

tuning_results = []
trained_models = {}

for learning_rate in learning_rates:

    print("\n" + "=" * 65)
    print(f"TESTING LEARNING RATE: {learning_rate}")
    print("=" * 65)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = build_model(learning_rate)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    best_val_loss = float(min(history.history["val_loss"]))

    tuning_results.append({
        "learning_rate": learning_rate,
        "best_validation_loss": best_val_loss,
        "epochs_trained": len(history.history["loss"]),
    })

    trained_models[learning_rate] = model

    print("Best validation loss:", round(best_val_loss, 4))


# ==========================================================
# 7. SELECT THE BEST LEARNING RATE USING VALIDATION LOSS
# ==========================================================

tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    RESULTS_DIR / "adam_learning_rate_tuning.csv",
    index=False,
)

best_learning_rate = float(
    tuning_df.loc[
        tuning_df["best_validation_loss"].idxmin(),
        "learning_rate",
    ]
)

best_model = trained_models[best_learning_rate]

print("\nLearning-rate comparison:")
print(tuning_df.to_string(index=False))

print("\nSelected learning rate:", best_learning_rate)


# ==========================================================
# 8. SELECT A CLASSIFICATION THRESHOLD USING VALIDATION DATA
# ==========================================================

val_probabilities = best_model.predict(
    X_val,
    verbose=0,
).ravel()

threshold_results = []

# Use a prespecified range of thresholds.
thresholds = np.arange(0.20, 0.61, 0.05)

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": float(threshold),
        "precision": precision_score(
            y_val,
            val_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_val,
            val_predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_val,
            val_predictions,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_val,
            val_predictions,
        ),
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    RESULTS_DIR / "threshold_comparison.csv",
    index=False,
)

# Choose the threshold with the highest validation F1-score.
# If tied, prefer the higher precision.
best_threshold = float(
    threshold_df.sort_values(
        ["f1_score", "precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]["threshold"]
)

print("\nValidation threshold comparison:")
print(
    threshold_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSelected classification threshold:", best_threshold)


# ==========================================================
# 9. FINAL TEST EVALUATION
# ==========================================================

test_probabilities = best_model.predict(
    X_test,
    verbose=0,
).ravel()

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_results = {
    "model": "Tuned Adam",
    "learning_rate": best_learning_rate,
    "classification_threshold": best_threshold,
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "f1_score": float(
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
    "confusion_matrix": confusion_matrix(
        y_test,
        test_predictions,
    ).tolist(),
}

print("\n" + "=" * 65)
print("TUNED ADAM: FINAL TEST RESULTS")
print("=" * 65)

for key, value in test_results.items():
    print(f"{key}: {value}")

with open(
    RESULTS_DIR / "tuned_adam_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(test_results, file, indent=4)

best_model.save(MODELS_DIR / "tuned_adam_ann.keras")

joblib.dump(
    preprocessor,
    MODELS_DIR / "tuned_adam_preprocessor.joblib",
)

print("\nSaved learning-rate results:")
print(RESULTS_DIR / "adam_learning_rate_tuning.csv")

print("\nSaved threshold results:")
print(RESULTS_DIR / "threshold_comparison.csv")

print("\nSaved final test metrics:")
print(RESULTS_DIR / "tuned_adam_metrics.json")

print("\nExperiment completed successfully.")

