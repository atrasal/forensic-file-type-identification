"""
Ensemble model for file-type fragment classification.

Combines predictions from Random Forest, XGBoost, and ResNet using
weighted soft voting for improved classification accuracy.

Usage:
  python -m models.ensemble.evaluate
  python models/ensemble/evaluate.py
"""

import os
import sys
import json
import time
import numpy as np
import joblib
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_loader import load_fragments_batched, get_all_labels, LazyFragmentDataset, CHUNK_SIZE
from utils.feature_engineering import extract_features_batch

# ========== CONFIG ==========
TRAIN_DIR = "datasets/train"
VAL_DIR = "datasets/val"
TEST_DIR = "datasets/test"
TRAIN_MAPPING = os.path.join(TRAIN_DIR, "fragment_mapping.csv")
VAL_MAPPING = os.path.join(VAL_DIR, "fragment_mapping.csv")
TEST_MAPPING = os.path.join(TEST_DIR, "fragment_mapping.csv")

RESULTS_PATH = "results/ensemble_results.json"
BATCH_SIZE = 50000

# Model paths
RF_MODEL_PATH = "saved_models/random_forest/rf_model.joblib"
XGB_MODEL_PATH = "saved_models/xgboost/xgb_model.joblib"
RESNET_MODEL_PATH = "saved_models/resnet/resnet_model.pth"
RESNET_RESULTS_PATH = "results/resnet_results.json"

# Ensemble weights (tunable — higher weight to better models)
WEIGHTS = {
    'rf': 0.40,
    'xgboost': 0.35,
    'resnet': 0.25,
}


def load_sklearn_model(path):
    """Load a sklearn/xgboost model saved with joblib."""
    saved = joblib.load(path)
    return saved['model'], saved['label_encoder']


def load_resnet_model(model_path, results_path, num_classes):
    """Load the ResNet model."""
    from models.resnet.train import ResNet1D

    model = ResNet1D(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def get_sklearn_probs(model, X_features, label_enc, num_classes):
    """Get probability predictions from a sklearn model."""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_features)
    else:
        # LinearSVC doesn't have predict_proba — use decision function
        decision = model.decision_function(X_features)
        # Convert to pseudo-probabilities via softmax
        exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probs


def get_resnet_probs(model, X_raw, batch_size=256):
    """Get probability predictions from ResNet on raw bytes."""
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_raw), batch_size):
            batch = X_raw[i:i + batch_size]
            x = torch.FloatTensor(batch).unsqueeze(1)  # (batch, 1, 4096)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1).numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


def main():
    # Check that required models exist
    missing = []
    for name, path in [('RF', RF_MODEL_PATH), ('XGBoost', XGB_MODEL_PATH), ('ResNet', RESNET_MODEL_PATH)]:
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print("❌ Missing models — train them first:")
        for m in missing:
            print(m)
        sys.exit(1)

    # Fit label encoder
    print("📦 Fitting label encoder...")
    all_train_labels = get_all_labels(TRAIN_MAPPING)
    label_enc = LabelEncoder()
    label_enc.fit(all_train_labels)
    class_names = label_enc.classes_
    num_classes = len(class_names)
    train_label_counts = dict(Counter(all_train_labels))

    # Load models
    print("📦 Loading models...")
    rf_model, rf_label_enc = load_sklearn_model(RF_MODEL_PATH)
    xgb_model, xgb_label_enc = load_sklearn_model(XGB_MODEL_PATH)
    resnet_model = load_resnet_model(RESNET_MODEL_PATH, RESNET_RESULTS_PATH, num_classes)
    print(f"  ✅ RF, XGBoost, ResNet loaded")

    # Load test data (need both raw bytes and engineered features)
    print("\n📦 Loading test data...")

    # Load raw bytes for ResNet
    all_X_raw = []
    all_y = []
    for X_batch, y_batch in load_fragments_batched(TEST_MAPPING, TEST_DIR, batch_size=BATCH_SIZE):
        X_norm = (X_batch / 255.0).astype(np.float32)
        all_X_raw.append(X_norm)
        all_y.extend(y_batch)
    X_raw = np.vstack(all_X_raw)
    y_test_str = np.array(all_y)
    y_test = label_enc.transform(y_test_str)
    del all_X_raw

    # Extract features for RF + XGBoost
    print("🔧 Extracting engineered features for sklearn models...")
    X_features = extract_features_batch(X_raw)

    n_test = len(y_test)
    print(f"📊 Test set: {n_test} samples, {num_classes} classes")

    # Get predictions from each model
    print("\n🤖 Getting predictions from each model...")

    print("  → Random Forest...")
    rf_probs = get_sklearn_probs(rf_model, X_features, rf_label_enc, num_classes)

    print("  → XGBoost...")
    xgb_probs = get_sklearn_probs(xgb_model, X_features, xgb_label_enc, num_classes)

    print("  → ResNet...")
    resnet_probs = get_resnet_probs(resnet_model, X_raw, batch_size=256)

    # Weighted ensemble
    print(f"\n🗳️  Combining with weights: RF={WEIGHTS['rf']}, XGB={WEIGHTS['xgboost']}, ResNet={WEIGHTS['resnet']}")
    inference_start = time.time()
    ensemble_probs = (
        WEIGHTS['rf'] * rf_probs +
        WEIGHTS['xgboost'] * xgb_probs +
        WEIGHTS['resnet'] * resnet_probs
    )
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    inference_time = time.time() - inference_start

    # Also get individual model predictions for comparison
    rf_preds = np.argmax(rf_probs, axis=1)
    xgb_preds = np.argmax(xgb_probs, axis=1)
    resnet_preds = np.argmax(resnet_probs, axis=1)

    # Evaluate ensemble
    accuracy = accuracy_score(y_test, ensemble_preds)
    precision = precision_score(y_test, ensemble_preds, average='macro', zero_division=0)
    recall = recall_score(y_test, ensemble_preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, ensemble_preds, average='macro', zero_division=0)

    print("\n📊 Ensemble Test Classification Report:")
    print(classification_report(y_test, ensemble_preds, target_names=class_names))
    print(f"✅ Ensemble Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    # Compare with individual models
    print("\n📊 Individual Model Comparison:")
    for name, preds in [("RF", rf_preds), ("XGBoost", xgb_preds), ("ResNet", resnet_preds), ("Ensemble", ensemble_preds)]:
        acc = accuracy_score(y_test, preds)
        f1_val = f1_score(y_test, preds, average='macro', zero_division=0)
        marker = "🏆" if name == "Ensemble" else "  "
        print(f"  {marker} {name:12s} Accuracy: {acc:.4f}, F1: {f1_val:.4f}")

    # Per-class metrics
    report = classification_report(y_test, ensemble_preds, target_names=class_names, output_dict=True)
    per_class_metrics = {}
    for cls in class_names:
        per_class_metrics[cls] = {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_test, ensemble_preds).tolist()

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {
        "model": "Ensemble (RF+XGBoost+ResNet)",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "parameters": {
            "method": "Weighted Soft Voting",
            "weights": WEIGHTS,
            "component_models": ["Random Forest", "XGBoost", "ResNet"],
        },
        "component_results": {
            "rf": {
                "accuracy": float(accuracy_score(y_test, rf_preds)),
                "f1_score": float(f1_score(y_test, rf_preds, average='macro', zero_division=0)),
            },
            "xgboost": {
                "accuracy": float(accuracy_score(y_test, xgb_preds)),
                "f1_score": float(f1_score(y_test, xgb_preds, average='macro', zero_division=0)),
            },
            "resnet": {
                "accuracy": float(accuracy_score(y_test, resnet_preds)),
                "f1_score": float(f1_score(y_test, resnet_preds, average='macro', zero_division=0)),
            },
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm,
        "dataset_info": {
            "test_samples": n_test,
            "num_classes": num_classes,
            "classes": list(class_names),
            "samples_per_class": {k: int(v) for k, v in train_label_counts.items()},
        },
        "inference_time_seconds": float(inference_time),
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
