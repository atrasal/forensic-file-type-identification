"""
MLP training script using engineered features for file-type fragment classification.

Unlike the raw-byte MLP (models/mlp/train.py), this model operates on the
317 hand-crafted features (byte histograms, entropy, bigrams, etc.) —
the same features used by Random Forest, XGBoost, and SVM.

This demonstrates that even a simple neural network benefits significantly
from feature engineering on this task.

Usage:
  python -m models.mlp_features.train
  python models/mlp_features/train.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

MODEL_SAVE_PATH = "saved_models/mlp_features/mlp_features_model.pth"
RESULTS_PATH = "results/mlp_features_results.json"
FEATURE_DIM = 317  # Number of engineered features
EPOCHS = 30
BATCH_SIZE = 256  # Larger batches since feature vectors are small
LR = 0.001
PATIENCE = 5
LOAD_BATCH_SIZE = 50000  # For loading fragments from disk


# ========== MLP Model on Engineered Features ==========
class FeatureMLP(nn.Module):
    """
    MLP operating on 317 engineered features instead of raw bytes.
    Uses BatchNorm for training stability and slightly wider layers
    since the input is much smaller (317 vs 4096).
    """
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_and_extract_features(mapping_file, base_dir, label_enc=None):
    """Load fragments in batches, extract features, free raw data. Memory-efficient."""
    all_features = []
    all_labels = []
    for X_batch, y_batch in load_fragments_batched(mapping_file, base_dir, batch_size=LOAD_BATCH_SIZE):
        X_batch = (X_batch / 255.0).astype(np.float32)
        feat_batch = extract_features_batch(X_batch)
        all_features.append(feat_batch)
        all_labels.extend(y_batch)
        del X_batch
    X_feat = np.vstack(all_features)
    del all_features
    y = np.array(all_labels)
    if label_enc is not None:
        y = label_enc.transform(y)
    return X_feat, y


def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


def main():
    print("📦 Fitting label encoder...")
    all_train_labels = get_all_labels(TRAIN_MAPPING)
    label_enc = LabelEncoder()
    label_enc.fit(all_train_labels)
    class_names = label_enc.classes_
    num_classes = len(class_names)
    train_label_counts = dict(Counter(all_train_labels))

    # Load and extract features
    print("\n📦 Loading training data + extracting features...")
    X_train, y_train = load_and_extract_features(TRAIN_MAPPING, TRAIN_DIR, label_enc)

    print("\n📦 Loading validation data + extracting features...")
    X_val, y_val = load_and_extract_features(VAL_MAPPING, VAL_DIR, label_enc)

    print("\n📦 Loading test data + extracting features...")
    X_test, y_test = load_and_extract_features(TEST_MAPPING, TEST_DIR, label_enc)

    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    print(f"\n📊 Train: {n_train}, Val: {n_val}, Test: {n_test}, Features: {X_train.shape[1]}, Classes: {list(class_names)}")

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Free raw arrays
    del X_train, X_val, X_test

    # Setup model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Compute class weights
    label_counts = np.bincount(label_enc.transform(all_train_labels))
    class_weights = len(all_train_labels) / (num_classes * label_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    model = FeatureMLP(FEATURE_DIM, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"📐 Parameters: {total_params:,} total, {trainable_params:,} trainable, {model_size_mb:.2f} MB")

    # Training
    print(f"\n🚀 Training MLP (features) on {n_train} samples ({FEATURE_DIM} features, patience={PATIENCE})...")
    start_time = time.time()
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  ✅ Restored best model (val_loss={best_val_loss:.4f})")

    # Evaluate
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

    inference_start = time.time()
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    inference_time = time.time() - inference_start

    test_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)

    print("\n📊 Test Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    print(f"✅ Val  Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    print(f"✅ Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"⏱️  Training time: {train_time:.2f}s")
    print(f"⏱️  Inference time: {inference_time:.4f}s")

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")

    # Per-class metrics
    report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    per_class_metrics = {}
    for cls in class_names:
        per_class_metrics[cls] = {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }

    cm = confusion_matrix(test_labels, test_preds).tolist()

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {
        "model": "MLP (Features)",
        "accuracy": float(test_acc),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1_score": float(test_f1),
        "val_accuracy": float(val_acc),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "val_f1_score": float(val_f1),
        "parameters": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": round(model_size_mb, 2),
            "input_features": FEATURE_DIM,
            "uses_batchnorm": True,
        },
        "training_history": history,
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
        "early_stopped": patience_counter >= PATIENCE,
        "best_val_loss": float(best_val_loss),
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
