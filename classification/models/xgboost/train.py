"""
XGBoost training script for file-type fragment classification.

Usage:
  python -m models.xgboost.train
  python models/xgboost/train.py
"""

import os
import sys
import time
import json
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_loader import load_fragments_batched, get_all_labels
from utils.feature_engineering import extract_features_batch, get_feature_names

# ========== CONFIG ==========
TRAIN_DIR = "datasets/train"
VAL_DIR = "datasets/val"
TEST_DIR = "datasets/test"
TRAIN_MAPPING = os.path.join(TRAIN_DIR, "fragment_mapping.csv")
VAL_MAPPING = os.path.join(VAL_DIR, "fragment_mapping.csv")
TEST_MAPPING = os.path.join(TEST_DIR, "fragment_mapping.csv")

MODEL_SAVE_PATH = "saved_models/xgboost/xgb_model.joblib"
RESULTS_PATH = "results/xgboost_results.json"
BATCH_SIZE = 50000  # Fragments per batch during loading


def load_and_extract_features(mapping_file, base_dir, label_enc=None):
    """Load fragments in batches, extract features, free raw data. Memory-efficient."""
    all_features = []
    all_labels = []
    for X_batch, y_batch in load_fragments_batched(mapping_file, base_dir, batch_size=BATCH_SIZE):
        X_batch = (X_batch / 255.0).astype(np.float32)
        feat_batch = extract_features_batch(X_batch)
        all_features.append(feat_batch)
        all_labels.extend(y_batch)
        del X_batch  # Free raw batch immediately
    X_feat = np.vstack(all_features)
    del all_features
    y = np.array(all_labels)
    if label_enc is not None:
        y = label_enc.transform(y)
    return X_feat, y


def main():
    # Fit label encoder on all training labels (just reads CSV column, no fragment data)
    print("📦 Fitting label encoder...")
    all_train_labels = get_all_labels(TRAIN_MAPPING)
    label_enc = LabelEncoder()
    label_enc.fit(all_train_labels)
    class_names = label_enc.classes_
    train_label_counts = dict(Counter(all_train_labels))
    num_classes = len(class_names)

    # Load and extract features in batches
    print("\n📦 Loading training data + extracting features...")
    X_train_feat, y_train = load_and_extract_features(TRAIN_MAPPING, TRAIN_DIR, label_enc)

    print("\n📦 Loading validation data + extracting features...")
    X_val_feat, y_val = load_and_extract_features(VAL_MAPPING, VAL_DIR, label_enc)

    print("\n📦 Loading test data + extracting features...")
    X_test_feat, y_test = load_and_extract_features(TEST_MAPPING, TEST_DIR, label_enc)

    n_train = X_train_feat.shape[0]
    n_val = X_val_feat.shape[0]
    n_test = X_test_feat.shape[0]
    print(f"\n📊 Train: {n_train}, Val: {n_val}, Test: {n_test}, Features: {X_train_feat.shape[1]}, Classes: {list(class_names)}")

    # Compute class-balanced sample weights
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    sample_weights = class_weights[y_train]

    # Train model with early stopping on validation set
    print(f"\n🚀 Training XGBoost on {n_train} samples ({X_train_feat.shape[1]} features)...")
    start_time = time.time()
    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        early_stopping_rounds=10,
    )
    model.fit(
        X_train_feat, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val_feat, y_val)],
        verbose=True,
    )
    train_time = time.time() - start_time

    # Evaluate on validation set
    y_val_pred = model.predict(X_val_feat)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)

    # Evaluate on test set
    inference_start = time.time()
    y_pred = model.predict(X_test_feat)
    inference_time = time.time() - inference_start

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n📊 Test Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"✅ Val  Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    print(f"✅ Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(f"⏱️  Training time: {train_time:.2f}s")
    print(f"⏱️  Inference time: {inference_time:.4f}s")

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump({
        'model': model,
        'label_encoder': label_enc,
        'use_engineered_features': True,
        'feature_names': get_feature_names(),
    }, MODEL_SAVE_PATH)
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")

    # Model parameters
    model_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)

    # Per-class metrics
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    per_class_metrics = {}
    for cls in class_names:
        per_class_metrics[cls] = {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {
        "model": "XGBoost",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "val_accuracy": float(val_accuracy),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "val_f1_score": float(val_f1),
        "parameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "model_size_mb": round(model_size_mb, 2),
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm,
        "dataset_info": {
            "train_samples": n_train,
            "val_samples": n_val,
            "test_samples": n_test,
            "total_samples": n_train + n_val + n_test,
            "num_classes": num_classes,
            "classes": list(class_names),
            "samples_per_class": {k: int(v) for k, v in train_label_counts.items()},
        },
        "train_time_seconds": float(train_time),
        "inference_time_seconds": float(inference_time),
        "best_iteration": int(model.best_iteration) if hasattr(model, 'best_iteration') else None,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

