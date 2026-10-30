# ============================================================
#  Smart Parking System – ML Model Training
#  Trains Decision Tree, Random Forest & Logistic Regression
#  and saves the best model for use by the dashboard.
# ============================================================

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "buffer") else sys.stdout
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import (accuracy_score, precision_score,
                                        recall_score, f1_score,
                                        confusion_matrix, classification_report,
                                        roc_auc_score)

from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

FEATURE_PATH  = os.path.join(os.path.dirname(__file__), "..", "data",   "parking_features.csv")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "reports")
BEST_MODEL    = os.path.join(MODELS_DIR, "best_model.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Load feature dataset ─────────────────────────────────────
def load_features(path: str = FEATURE_PATH):
    df = pd.read_csv(path)
    X  = df[FEATURE_COLUMNS].fillna(0)
    y  = df[TARGET_COLUMN]
    print(f"[✓] Features loaded : X{X.shape}  y{y.shape}")
    print(f"    Class balance   : 0={int((y==0).sum()):,}  1={int((y==1).sum()):,}")
    return X, y


# ── Build model pipelines ────────────────────────────────────
def build_pipelines() -> dict:
    return {
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            ))
        ]),
    }


# ── Evaluate a single model ──────────────────────────────────
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, "predict_proba") else None)

    metrics = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score (y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall"   : round(recall_score   (y_test, y_pred) * 100, 2),
        "F1-Score" : round(f1_score       (y_test, y_pred) * 100, 2),
        "ROC-AUC"  : round(roc_auc_score  (y_test, y_prob) * 100, 2)
                     if y_prob is not None else "N/A",
    }
    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"{'-'*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:12s}: {v}%")
    print(classification_report(y_test, y_pred,
                                 target_names=["Vacant","Occupied"]))
    return metrics


# ── Plot confusion matrix ────────────────────────────────────
def plot_confusion_matrix(name: str, model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Vacant","Occupied"],
                yticklabels=["Vacant","Occupied"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    safe = name.replace(" ", "_").lower()
    path = os.path.join(REPORTS_DIR, f"cm_{safe}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Confusion matrix saved → {path}")


# ── Feature importance (Random Forest) ──────────────────────
def plot_feature_importance(model, X):
    rf_clf = model.named_steps["clf"]
    if not hasattr(rf_clf, "feature_importances_"):
        return
    imp  = rf_clf.feature_importances_
    feat = X.columns.tolist()
    fi   = pd.Series(imp, index=feat).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances – Random Forest")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Feature importance saved → {path}")


# ── Compare models bar chart ─────────────────────────────────
def plot_model_comparison(results: list):
    df  = pd.DataFrame(results).set_index("Model")
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x  = np.arange(len(df))
    w  = 0.18
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*w, df[m], width=w, label=m, color=c)
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison – Classification Metrics")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Model comparison chart saved → {path}")


# ── Cross-validation ─────────────────────────────────────────
def cross_validate_models(pipelines: dict, X_train, y_train):
    print("\n[Cross-Validation – 5-fold F1 scores]")
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=5, scoring="f1", n_jobs=-1)
        print(f"  {name:22s}: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")


# ── Main training routine ─────────────────────────────────────
def train(save_best: bool = True):
    X, y    = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[✓] Train : {len(X_train):,}  Test : {len(X_test):,}")

    pipelines = build_pipelines()
    cross_validate_models(pipelines, X_train, y_train)

    results     = []
    trained     = {}

    for name, pipe in pipelines.items():
        print(f"\n[→] Training {name} …")
        pipe.fit(X_train, y_train)
        trained[name] = pipe

        metrics = evaluate_model(name, pipe, X_test, y_test)
        results.append(metrics)
        plot_confusion_matrix(name, pipe, X_test, y_test)

        # Save individual model
        safe = name.replace(" ", "_").lower()
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"{safe}.pkl"))
        print(f"[✓] Saved → models/{safe}.pkl")

    # Feature importance for RF
    plot_feature_importance(trained["Random Forest"], X_train)
    plot_model_comparison(results)

    # Pick best model by F1
    best_name = max(results, key=lambda r: r["F1-Score"])["Model"]
    best_pipe = trained[best_name]
    print(f"\n[Best model] {best_name}")

    if save_best:
        joblib.dump(best_pipe, BEST_MODEL)
        print(f"[✓] Best model saved → {BEST_MODEL}")

    # Save metrics table
    df_res = pd.DataFrame(results)
    path   = os.path.join(REPORTS_DIR, "model_metrics.csv")
    df_res.to_csv(path, index=False)
    print(f"[✓] Metrics table saved → {path}")

    return trained, results


if __name__ == "__main__":
    train()
