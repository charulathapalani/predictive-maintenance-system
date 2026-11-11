"""Train the AI/ML predictive-intelligence engine.

Three models are trained and saved in one bundle:
  * health model   - HEALTHY / WARNING / CRITICAL (Random Forest by default)
  * fault model    - fault classification for non-healthy windows
  * anomaly model  - Isolation Forest fitted on healthy data only

Usage:  python -m src.train --samples 600 --model random_forest
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import METRICS_PATH, MODEL_DIR, MODEL_PATH, STATES
from .data_generator import generate_dataset
from .features import FEATURE_COLUMNS


def make_model(name: str, seed: int = 42):
    """Model factory - the engine is not tied to Random Forest."""
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    raise ValueError(f"Unknown model '{name}'")


def _pipeline(name: str, seed: int) -> Pipeline:
    # StandardScaler = normalisation / scaling step
    return Pipeline([("scaler", StandardScaler()), ("model", make_model(name, seed))])


def _save_confusion_plot(cm, labels, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Motor health - confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def train(samples_per_class: int = 600, model_name: str = "random_forest",
          seed: int = 42, model_path=MODEL_PATH, verbose: bool = True) -> dict:
    data = generate_dataset(samples_per_class, seed)
    X, y_state, y_fault = data[FEATURE_COLUMNS], data["state"], data["fault"]

    X_tr, X_te, ys_tr, ys_te, yf_tr, yf_te = train_test_split(
        X, y_state, y_fault, test_size=0.25, stratify=y_state, random_state=seed)

    health = _pipeline(model_name, seed).fit(X_tr, ys_tr)
    pred = health.predict(X_te)
    cm = confusion_matrix(ys_te, pred, labels=STATES)

    # Fault classifier (only on non-healthy windows)
    bad_tr, bad_te = ys_tr != "HEALTHY", ys_te != "HEALTHY"
    fault = _pipeline(model_name, seed).fit(X_tr[bad_tr], yf_tr[bad_tr])
    fault_acc = accuracy_score(yf_te[bad_te], fault.predict(X_te[bad_te]))

    # Anomaly detector trained only on healthy behaviour
    healthy = X_tr[ys_tr == "HEALTHY"]
    a_scaler = StandardScaler().fit(healthy)
    anomaly = IsolationForest(n_estimators=200, contamination=0.02, random_state=seed)
    anomaly.fit(a_scaler.transform(healthy))

    importances = {}
    inner = health.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        top = np.argsort(inner.feature_importances_)[::-1][:10]
        importances = {FEATURE_COLUMNS[i]: round(float(inner.feature_importances_[i]), 4) for i in top}

    metrics = {
        "model": model_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data": "synthetic (simulated) sensor windows",
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "health_accuracy": round(float(accuracy_score(ys_te, pred)), 4),
        "fault_accuracy": round(float(fault_acc), 4),
        "classification_report": classification_report(ys_te, pred, output_dict=True),
        "confusion_matrix": {"labels": STATES, "matrix": cm.tolist()},
        "top_features": importances,
    }

    bundle = {
        "health_model": health, "fault_model": fault,
        "anomaly_model": anomaly, "anomaly_scaler": a_scaler,
        "feature_columns": FEATURE_COLUMNS,
        "medians": X_tr.median().to_dict(),
        "healthy_mean": dict(zip(FEATURE_COLUMNS, a_scaler.mean_)),
        "healthy_std": dict(zip(FEATURE_COLUMNS, a_scaler.scale_)),
        "metrics": metrics,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    _save_confusion_plot(cm, STATES, MODEL_DIR / "confusion_matrix.png")

    if verbose:
        print(f"Model: {model_name} | health accuracy: {metrics['health_accuracy']:.3f} "
              f"| fault accuracy: {metrics['fault_accuracy']:.3f}")
        print(classification_report(ys_te, pred))
        print(f"Saved -> {model_path}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train predictive-maintenance models")
    ap.add_argument("--samples", type=int, default=600, help="windows per health class")
    ap.add_argument("--model", default="random_forest",
                    choices=["random_forest", "gradient_boosting", "logistic_regression"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train(a.samples, a.model, a.seed)
