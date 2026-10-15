import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize

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


# ==========================================================
# 1. PROJECT SETTINGS
# ==========================================================

SEED = 42
EPOCHS = 50
BATCH_SIZE = 32

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
    raise ValueError("Unexpected target values in Churn.")

print("Dataset:", df.shape)


# ==========================================================
# 3. CREATE FIXED DATA SPLITS
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
# 4. PREPROCESS FEATURES
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


def to_dense_float32(values):
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


X_train = to_dense_float32(X_train)
X_val = to_dense_float32(X_val)
X_test = to_dense_float32(X_test)

y_train = y_train.to_numpy(dtype=np.float32)
y_val = y_val.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.int32)

print("Processed feature count:", X_train.shape[1])


# ==========================================================
# 5. MODEL FACTORY
# ==========================================================

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    return model


# ==========================================================
# 6. EVALUATION FUNCTION
# ==========================================================

def evaluate_model(model, optimizer_name):
    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "optimizer": optimizer_name,
        "test_loss": float(
            tf.keras.losses.binary_crossentropy(
                y_test.astype(np.float32),
                probabilities,
            ).numpy().mean()
        ),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print(f"\n{optimizer_name} results:")
    for name, value in metrics.items():
        if name != "optimizer":
            print(f"{name}: {value:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return metrics


# ==========================================================
# 7. EXPERIMENT 1: ADAM
# ==========================================================

print("\n" + "=" * 55)
print("EXPERIMENT 1: ADAM")
print("=" * 55)

tf.keras.backend.clear_session()
tf.random.set_seed(SEED)

adam_model = build_model()

adam_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
)

adam_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ],
    verbose=1,
)

adam_metrics = evaluate_model(adam_model, "Adam")

adam_model.save(MODELS_DIR / "adam_ann.keras")


# ==========================================================
# 8. EXPERIMENT 2: SGD
# ==========================================================

print("\n" + "=" * 55)
print("EXPERIMENT 2: SGD")
print("=" * 55)

tf.keras.backend.clear_session()
tf.random.set_seed(SEED)

sgd_model = build_model()

sgd_model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9,
    ),
    loss="binary_crossentropy",
)

sgd_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ],
    verbose=1,
)

sgd_metrics = evaluate_model(sgd_model, "SGD")

sgd_model.save(MODELS_DIR / "sgd_ann.keras")


# ==========================================================
# 9. EXPERIMENT 3: L-BFGS
# ==========================================================

print("\n" + "=" * 55)
print("EXPERIMENT 3: L-BFGS")
print("=" * 55)

tf.keras.backend.clear_session()
tf.random.set_seed(SEED)

lbfgs_model = build_model()

# L-BFGS is a full-batch optimizer.
# Use float64 for its numerical optimization variables.
train_x_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
train_y_tf = tf.convert_to_tensor(
    y_train.reshape(-1, 1), dtype=tf.float64
)

def get_parameters(model):
    return [
        np.asarray(variable.numpy(), dtype=np.float64)
        for variable in model.trainable_variables
    ]


def set_parameters(model, parameters):
    for variable, parameter in zip(
        model.trainable_variables, parameters
    ):
        variable.assign(parameter)


initial_parameters = get_parameters(lbfgs_model)
parameter_shapes = [p.shape for p in initial_parameters]
parameter_sizes = [p.size for p in initial_parameters]


def flatten_parameters(parameters):
    return np.concatenate([p.ravel() for p in parameters])


def unflatten_parameters(flat_parameters):
    parameters = []
    offset = 0

    for shape, size in zip(parameter_shapes, parameter_sizes):
        parameters.append(
            flat_parameters[offset:offset + size].reshape(shape)
        )
        offset += size

    return parameters


def lbfgs_objective(flat_parameters):
    parameters = unflatten_parameters(flat_parameters)
    set_parameters(lbfgs_model, parameters)

    with tf.GradientTape() as tape:
        predictions = lbfgs_model(train_x_tf, training=True)

        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                train_y_tf, predictions
            )
        )

    gradients = tape.gradient(
        loss, lbfgs_model.trainable_variables
    )

    flat_gradients = np.concatenate([
        np.asarray(g.numpy(), dtype=np.float64).ravel()
        for g in gradients
    ])

    return float(loss.numpy()), flat_gradients


initial_flat = flatten_parameters(initial_parameters)

optimization_result = minimize(
    fun=lbfgs_objective,
    x0=initial_flat,
    method="L-BFGS-B",
    jac=True,
    options={
        "maxiter": 200,
        "maxfun": 3000,
        "ftol": 1e-9,
    },
)

set_parameters(
    lbfgs_model,
    unflatten_parameters(optimization_result.x),
)

print("L-BFGS convergence:", optimization_result.success)
print("L-BFGS message:", optimization_result.message)
print("L-BFGS iterations:", optimization_result.nit)
print("L-BFGS training loss:", optimization_result.fun)

lbfgs_metrics = evaluate_model(lbfgs_model, "L-BFGS")

lbfgs_model.save(MODELS_DIR / "lbfgs_ann.keras")


# ==========================================================
# 10. SAVE THE COMPARISON
# ==========================================================

comparison = pd.DataFrame([
    adam_metrics,
    sgd_metrics,
    lbfgs_metrics,
])

comparison.to_csv(
    RESULTS_DIR / "optimizer_comparison.csv",
    index=False,
)

print("\n" + "=" * 65)
print("FINAL OPTIMIZER COMPARISON")
print("=" * 65)

print(
    comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nComparison saved to:")
print(RESULTS_DIR / "optimizer_comparison.csv")

print("\nExperiment completed.")