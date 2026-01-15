"""
Plot Model Metrics
Loads saved models from ./models, computes classification reports and precision-recall / ROC curves,
and saves images under ./models/metrics/.

Usage: python itps_project/plot_model_metrics.py
"""
import os
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)

from implement_advanced_models import AdvancedModelTrainer


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def transform_and_flatten(X: np.ndarray, scaler):
    """Apply scaler to (n, seq_len, feat_dim) and return flattened (n, seq_len*feat_dim)."""
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    X_flat = X_scaled.reshape(X.shape[0], -1)
    return X_flat


def evaluate_saved_model(name: str, model_data: dict, X: np.ndarray, y: np.ndarray, test_idx):
    """Evaluate a saved model and return test labels, preds, scores and text report."""
    model = model_data.get("model")
    if model is None:
        raise ValueError(f"Model {name} has no model object saved")

    # For ensemble model, use lstm_scaler which is used in training
    if model_data.get("model_type") == "ensemble":
        scaler = model_data.get("lstm_scaler")
    else:
        scaler = model_data.get("scaler")

    if scaler is None:
        raise ValueError(f"Model {name} has no scaler saved; cannot prepare inputs")

    X_flat = transform_and_flatten(X, scaler)
    X_test = X_flat[test_idx]
    y_test = y[test_idx]

    # Autoencoder case: model is PCA and we use reconstruction error
    if model_data.get("model_type", "").startswith("autoencoder"):
        pca = model
        try:
            X_recon = pca.inverse_transform(pca.transform(X_test))
            rec_errors = np.mean((X_test - X_recon) ** 2, axis=1)
            # prediction: malicious if error > threshold
            threshold = model_data.get("threshold", np.percentile(rec_errors, 95))
            y_pred = np.where(rec_errors > threshold, 0, 1)
            y_score = rec_errors  # higher -> more likely malicious
        except Exception as e:
            raise ValueError(f"Error processing autoencoder model: {e}")
    else:
        # Typical classifier with predict / predict_proba
        try:
            y_pred = model.predict(X_test)
            # probability of malicious class (0)
            y_score = model.predict_proba(X_test)[:, 0]
        except Exception as e:
            raise ValueError(f"Error running model predictions: {e}")

    report = classification_report(y_test, y_pred, target_names=["malicious(0)", "benign(1)"], output_dict=False)

    return y_test, y_pred, y_score, report


def plot_classification_report(text: str, outpath: Path, title: str = "Classification Report"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0, 0.95, title, fontsize=12, fontweight="bold")
    ax.text(0, 0.02, text, fontsize=10, family="monospace")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_precision_recall(y_test, y_score, outpath: Path, title: str = "Precision-Recall Curve"):
    if y_score is None:
        print(f"Skipping PR plot for {title}: no score/probabilities available.")
        return

    y_true = (y_test == 0).astype(int)  # malicious as positive
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_test, y_score, outpath: Path, title: str = "ROC Curve"):
    if y_score is None:
        print(f"Skipping ROC plot for {title}: no score/probabilities available.")
        return

    y_true = (y_test == 0).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    models_dir = Path("./models")
    metrics_dir = models_dir / "metrics"
    ensure_dir(metrics_dir)

    trainer = AdvancedModelTrainer()
    X, y = trainer.load_and_process_data(max_sequences=2000)

    n = X.shape[0]
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    # List of model files to evaluate
    files = {
        "lstm": models_dir / "lstm_model.pkl",
        "transformer": models_dir / "transformer_model.pkl",
        "autoencoder": models_dir / "autoencoder_model.pkl",
        "ensemble": models_dir / "ensemble_model.pkl",
    }

    for name, path in files.items():
        if not path.exists():
            print(f"Model file not found: {path} — skipping {name}")
            continue

        model_data = joblib.load(path)
        try:
            y_test, y_pred, y_score, report = evaluate_saved_model(name, model_data, X, y, test_idx)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue

        # Save classification report as image
        cr_path = metrics_dir / f"classification_report_{name}.png"
        plot_classification_report(report, cr_path, title=f"Classification Report ({name})")
        print(f"Saved classification report image to {cr_path}")

        # Save PR curve
        pr_path = metrics_dir / f"precision_recall_{name}.png"
        plot_precision_recall(y_test, y_score, pr_path, title=f"Precision-Recall ({name})")
        print(f"Saved precision-recall image to {pr_path}")

        # Save ROC curve
        roc_path = metrics_dir / f"roc_{name}.png"
        plot_roc_curve(y_test, y_score, roc_path, title=f"ROC ({name})")
        print(f"Saved ROC image to {roc_path}")


if __name__ == "__main__":
    main()
