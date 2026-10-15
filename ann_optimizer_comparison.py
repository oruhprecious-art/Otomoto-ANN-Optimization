"""
Otomoto ANN optimizer comparison
Dataset: data/teleconnect.csv
Compares three optimization algorithms in a feed-forward neural network:
Adam, SGD, and L-BFGS. Uses a stratified 80/20 hold-out split.
Run: python ann_optimizer_comparison.py
"""
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss, classification_report,
                             ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "teleconnect.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
for folder in (RESULTS, FIGURES, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' in teleconnect.csv")
    # TotalCharges is numeric in meaning but may contain blank strings.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Unexpected values found in Churn target.")
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    configs = {
        "Adam": {"solver": "adam", "learning_rate_init": 0.001},
        "SGD": {"solver": "sgd", "learning_rate_init": 0.01},
        "L-BFGS": {"solver": "lbfgs", "learning_rate_init": 0.001}
    }
    rows, fitted = [], {}
    for label, cfg in configs.items():
        ann = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver=cfg["solver"], alpha=0.001,
            learning_rate_init=cfg["learning_rate_init"],
            max_iter=300, early_stopping=(cfg["solver"] != "lbfgs"),
            validation_fraction=0.10, n_iter_no_change=15,
            random_state=42
        )
        pipeline = Pipeline([("preprocess", preprocess), ("ann", ann)])
        start = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        row = {
            "Optimizer": label,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "Log loss": log_loss(y_test, proba),
            "Iterations": pipeline.named_steps["ann"].n_iter_,
            "Training time (s)": elapsed
        }
        rows.append(row)
        fitted[label] = pipeline
        print(f"\\n--- {label} ---")
        print(pd.Series(row).to_string())
        print(classification_report(y_test, pred, target_names=["No churn", "Churn"], zero_division=0))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No churn", "Churn"])
        plt.title(f"Confusion matrix — {label}")
        plt.tight_layout()
        plt.savefig(FIGURES / f"confusion_matrix_{label.lower().replace('-', '')}.png", dpi=180)
        plt.close()

    result = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
    result.to_csv(RESULTS / "optimizer_comparison.csv", index=False)
    # Choose the best F1 model for the initial deployment candidate.
    best_name = result.iloc[0]["Optimizer"]
    joblib.dump(fitted[best_name], MODELS / "best_ann_model.joblib")
    with open(RESULTS / "best_model.txt", "w", encoding="utf-8") as f:
        f.write(f"Selected by test F1-score: {best_name}\\n")
        f.write("Important: select/tune models on validation data in a production workflow; do not repeatedly optimize against the test set.\\n")

    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    result.set_index("Optimizer")[metrics].plot(kind="bar", figsize=(8, 5))
    plt.ylim(0, 1); plt.ylabel("Score"); plt.title("ANN optimizer comparison")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(FIGURES / "optimizer_metrics.png", dpi=180); plt.close()
    result.set_index("Optimizer")["ROC-AUC"].plot(kind="bar", figsize=(7, 4))
    plt.ylim(0, 1); plt.ylabel("ROC-AUC"); plt.title("ROC-AUC by optimizer")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(FIGURES / "roc_auc.png", dpi=180); plt.close()
    print("\\nSaved metrics, plots, and models under results/, figures/, and models/.")

if __name__ == "__main__":
    main()
