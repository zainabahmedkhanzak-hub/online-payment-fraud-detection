import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def main():
    csv_path = "transactions.csv"  # Apni downloaded CSV ka naam yahan check kar lo
    df = pd.read_csv(csv_path)

    # Target column standardize karo
    if "isFraud" in df.columns and "is_fraud" not in df.columns:
        df = df.rename(columns={"isFraud": "is_fraud"})

    # Extra/ID columns drop karo
    drop_cols = [c for c in ["nameOrig", "nameDest", "isFlaggedFraud"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = df.drop_duplicates().dropna(subset=["is_fraud"])

    # Agar dataset 10,000 rows se bara hai to stratified sample lo (fast & clean comparison ke liye)
    if len(df) > 10000:
        df, _ = train_test_split(
            df, train_size=10000, stratify=df["is_fraud"].astype(int), random_state=42
        )

    target_col = "is_fraud"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    print("Dataset shape used:", df.shape)
    print("Class distribution:\n", y.value_counts(normalize=True).round(4))

    # 70% Train, 15% Validation, 15% Test (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, stratify=y_train_full, random_state=42
    )

    cat_cols = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # SMOTE sirf training data par (data leakage se bachne ke liye)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_and_grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.1, 1.0, 10.0]}
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [5, 10, None], "min_samples_split": [2, 10]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}
        )
    }

    results = []
    roc_data = {}
    best_model = None
    best_f1 = -1.0
    best_name = ""

    for name, (estimator, param_grid) in models_and_grids.items():
        grid = GridSearchCV(estimator, param_grid, scoring="f1", cv=cv, n_jobs=-1)

        t0 = time.perf_counter()
        grid.fit(X_train_bal, y_train_bal)
        train_time = time.perf_counter() - t0

        best_est = grid.best_estimator_

        t1 = time.perf_counter()
        y_pred = best_est.predict(X_test_proc)
        pred_time = time.perf_counter() - t1

        y_prob = best_est.predict_proba(X_test_proc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Best Params": str(grid.best_params_),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Train Time (s)": round(train_time, 3),
            "Predict Time (s)": round(pred_time, 4)
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)

        if f1 > best_f1:
            best_f1 = f1
            best_model = best_est
            best_name = name

    res_df = pd.DataFrame(results)
    print("\n=== TEST SET RESULTS ===")
    print(res_df.to_string(index=False))
    res_df.to_csv("model_comparison_results.csv", index=False)

    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=300)
    plt.close()

    cm = confusion_matrix(y_test, best_model.predict(X_test_proc))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("best_model_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"\nBest model by F1-Score: {best_name} (F1 = {best_f1:.4f})")


if __name__ == "__main__":
    main()
