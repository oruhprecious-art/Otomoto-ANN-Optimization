import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

DATA_PATH = "data/teleconnect_clean.csv"
MODEL_PATH = "models/tuned_adam_ann.keras"
PREPROCESSOR_PATH = "models/tuned_adam_preprocessor.joblib"

SEED = 42

# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Remove customer identifier if it remains in the file.
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Encode the target.
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

# --------------------------------------------------
# 3. Recreate the original stratified data split
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

# --------------------------------------------------
# 4. Load the saved preprocessing pipeline and model
# --------------------------------------------------

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

X_test_processed = preprocessor.transform(X_test)

# --------------------------------------------------
# 5. Generate test probabilities once
# --------------------------------------------------

y_prob = model.predict(
    X_test_processed,
    verbose=0
).ravel()

# --------------------------------------------------
# 6. Compare thresholds using identical probabilities
# --------------------------------------------------

print("\nFINAL THRESHOLD COMPARISON")
print("=" * 60)

for threshold in [0.30, 0.50]:

    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_test, y_pred, zero_division=0):.4f}"
    )
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\nThreshold comparison completed.")