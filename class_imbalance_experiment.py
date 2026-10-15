from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "teleconnect.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Candidate thresholds selected using validation data only.
THRESHOLDS = np.round(
    np.arange(0.10, 0.91, 0.05),
    2
)

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if "Churn" not in df.columns:
    raise ValueError(
        "The dataset must contain a column named 'Churn'."
    )

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["Churn"].value_counts())

print("\nClass percentages:")
print(
    (
        df["Churn"].value_counts(normalize=True) * 100
    ).round(2)
)

# Convert TotalCharges to numeric.
# Blank strings become missing values and are imputed below.
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Encode the target:
# No churn = 0
# Churn = 1

y = (
    df["Churn"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"no": 0, "yes": 1})
)

if y.isna().any():
    raise ValueError(
        "Unexpected values were found in the Churn column."
    )

# Exclude the target and customer identifier.
X = df.drop(
    columns=["Churn", "customerID"],
    errors="ignore"
)


# ============================================================
# 3. CREATE TRAINING, VALIDATION AND TEST SETS
# ============================================================

# 70% training, 30% temporary.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

# Divide temporary data equally:
# 70% training, 15% validation, 15% test.

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation class distribution:")
print(y_val.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# 4. PREPROCESS THE DATA
# ============================================================

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]),
            numeric_columns
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            categorical_columns
        )
    ],
    sparse_threshold=0.0
)


def to_dense_float32(data):
    """Convert transformed features to a dense float32 array."""

    if hasattr(data, "toarray"):
        data = data.toarray()

    return np.asarray(data, dtype=np.float32)


# Fit the preprocessor on training data only.
X_train_processed = preprocessor.fit_transform(X_train)

# Transform validation and test data without refitting.
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = to_dense_float32(
    X_train_processed
)

X_val_processed = to_dense_float32(
    X_val_processed
)

X_test_processed = to_dense_float32(
    X_test_processed
)

y_train_array = np.asarray(
    y_train,
    dtype=np.float32
)

y_val_array = np.asarray(
    y_val,
    dtype=np.int32
)

y_test_array = np.asarray(
    y_test,
    dtype=np.int32
)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed validation shape:", X_val_processed.shape)
print("Processed test shape:", X_test_processed.shape)


# ============================================================
# 5. DEFINE THE ANN
# ============================================================

def build_model(input_features):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(input_features,)
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.003
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 6. CALCULATE CLASS WEIGHTS
# ============================================================

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.asarray(y_train, dtype=int)
)

class_weights = {
    int(label): float(weight)
    for label, weight in zip(classes, weights)
}

print("\n" + "=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

print("Class 0 = No churn")
print("Class 1 = Churn")
print(class_weights)


# ============================================================
# 7. SELECT THRESHOLD USING VALIDATION DATA
# ============================================================

validation_results = []
final_test_results = []


def find_best_threshold(
    model_name,
    validation_probabilities
):
    """
    Select the threshold with the highest validation F1-score.

    The test set is not used for threshold selection.
    """

    model_threshold_results = []

    for threshold in THRESHOLDS:

        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val_array,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val_array,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val_array,
            predictions,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_val_array,
            predictions
        )

        row = {
            "Model": model_name,
            "Threshold": float(threshold),
            "Validation Accuracy": accuracy,
            "Validation Precision": precision,
            "Validation Recall": recall,
            "Validation F1-score": f1
        }

        model_threshold_results.append(row)
        validation_results.append(row)

    threshold_df = pd.DataFrame(
        model_threshold_results
    )

    # Select the threshold with the highest validation F1-score.
    # If tied, prefer the higher threshold to avoid unnecessary
    # positive predictions.
    threshold_df = threshold_df.sort_values(
        by=[
            "Validation F1-score",
            "Threshold"
        ],
        ascending=[False, False]
    )

    best_row = threshold_df.iloc[0]

    best_threshold = float(
        best_row["Threshold"]
    )

    print("\n" + "-" * 70)
    print("VALIDATION THRESHOLD SELECTION:", model_name)
    print("-" * 70)

    print(
        threshold_df.head(5).round(4).to_string(
            index=False
        )
    )

    print("\nSelected threshold:", best_threshold)

    print(
        "Best validation F1-score:",
        round(
            float(best_row["Validation F1-score"]),
            4
        )
    )

    return best_threshold


# ============================================================
# 8. TRAIN AND EVALUATE EACH MODEL
# ============================================================

def train_evaluate_model(
    model_name,
    use_class_weights
):

    print("\n" + "#" * 70)
    print("TRAINING:", model_name)
    print("#" * 70)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = build_model(
        X_train_processed.shape[1]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    fit_arguments = {}

    if use_class_weights:
        fit_arguments["class_weight"] = class_weights

    history = model.fit(
        X_train_processed,
        y_train_array,
        validation_data=(
            X_val_processed,
            y_val_array.astype(np.float32)
        ),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
        **fit_arguments
    )

    print(
        "Training completed in",
        len(history.history["loss"]),
        "epochs."
    )

    # Generate validation probabilities.
    validation_probabilities = model.predict(
        X_val_processed,
        verbose=0
    ).ravel()

    # Choose threshold using validation F1-score only.
    best_threshold = find_best_threshold(
        model_name,
        validation_probabilities
    )

    # Save the trained model.
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_path = (
        MODELS_DIR / f"{safe_name}_validation_tuned.keras"
    )

    model.save(model_path)

    print("\nModel saved to:", model_path)

    # --------------------------------------------------------
    # Final test evaluation at the selected threshold.
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FINAL TEST EVALUATION:", model_name)
    print("-" * 70)

    # Test probabilities are generated only after threshold selection.
    test_probabilities = model.predict(
        X_test_processed,
        verbose=0
    ).ravel()

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test_array,
        test_predictions
    )

    precision = precision_score(
        y_test_array,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test_array,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test_array,
        test_predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test_array,
        test_probabilities
    )

    cm = confusion_matrix(
        y_test_array,
        test_predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    result = {
        "Model": model_name,
        "Selected Threshold": best_threshold,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1-score": f1,
        "Test ROC-AUC": auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Epochs trained": len(
            history.history["loss"]
        )
    }

    final_test_results.append(result)

    print(
        f"Selected threshold: {best_threshold:.2f}"
    )

    print(
        f"Accuracy:  {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall:    {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-score:  {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"ROC-AUC:   {auc:.4f} "
        f"({auc * 100:.2f}%)"
    )

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")

    print(
        classification_report(
            y_test_array,
            test_predictions,
            labels=[0, 1],
            target_names=[
                "No churn",
                "Churn"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save confusion matrix figure.
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        cm,
        interpolation="nearest"
    )

    ax.set_title(
        f"{model_name}\n"
        f"Selected threshold = {best_threshold:.2f}"
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    ax.set_xticks(
        [0, 1],
        ["No churn", "Churn"]
    )

    ax.set_yticks(
        [0, 1],
        ["No churn", "Churn"]
    )

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=12
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    figure_path = (
        FIGURES_DIR
        / f"{safe_name}_final_test_confusion_matrix.png"
    )

    fig.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nConfusion matrix saved to:", figure_path)


# ============================================================
# 9. RUN BOTH EXPERIMENTS
# ============================================================

train_evaluate_model(
    model_name="Unweighted ANN",
    use_class_weights=False
)

train_evaluate_model(
    model_name="Class-Weighted ANN",
    use_class_weights=True
)


# ============================================================
# 10. SAVE VALIDATION THRESHOLD RESULTS
# ============================================================

validation_df = pd.DataFrame(
    validation_results
)

validation_path = (
    RESULTS_DIR
    / "validation_threshold_search.csv"
)

validation_df.to_csv(
    validation_path,
    index=False
)

print("\n" + "=" * 90)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 90)

for model_name in validation_df["Model"].unique():

    print("\n", model_name)

    model_rows = validation_df[
        validation_df["Model"] == model_name
    ]

    print(
        model_rows.sort_values(
            by="Validation F1-score",
            ascending=False
        ).head(5).round(4).to_string(
            index=False
        )
    )

print("\nValidation results saved to:")
print(validation_path)


# ============================================================
# 11. SAVE FINAL TEST RESULTS
# ============================================================

test_df = pd.DataFrame(
    final_test_results
)

test_path = (
    RESULTS_DIR
    / "final_validation_selected_test_results.csv"
)

test_df.to_csv(
    test_path,
    index=False
)

print("\n" + "=" * 90)
print("FINAL TEST RESULTS")
print("=" * 90)

print(
    test_df.round(4).to_string(index=False)
)

print("\nFinal test results saved to:")
print(test_path)


# ============================================================
# 12. SAVE EXPERIMENT SETTINGS
# ============================================================

settings = {
    "random_seed": SEED,
    "dataset": str(DATA_PATH),
    "train_validation_test_split": "70/15/15",
    "architecture": [
        "Dense(32, activation='relu')",
        "Dense(1, activation='sigmoid')"
    ],
    "optimizer": "Adam",
    "learning_rate": 0.003,
    "loss": "binary_crossentropy",
    "batch_size": 32,
    "maximum_epochs": 50,
    "early_stopping_patience": 5,
    "threshold_selection": (
        "Highest validation F1-score; ties prefer higher threshold"
    ),
    "candidate_thresholds": [
        float(value) for value in THRESHOLDS
    ],
    "class_weights": class_weights
}

settings_path = (
    RESULTS_DIR
    / "validation_threshold_experiment_settings.json"
)

with open(
    settings_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        settings,
        file,
        indent=4
    )

print("\nExperiment settings saved to:")
print(settings_path)

print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY.")