"""
CNN training script for file-type fragment classification.

Usage:
  python -m models.cnn.train
  python models/cnn/train.py
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
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_loader import get_all_labels, LazyFragmentDataset, CHUNK_SIZE

# ========== CONFIG ==========
TRAIN_DIR = "datasets/train"
VAL_DIR = "datasets/val"
TEST_DIR = "datasets/test"
TRAIN_MAPPING = os.path.join(TRAIN_DIR, "fragment_mapping.csv")
VAL_MAPPING = os.path.join(VAL_DIR, "fragment_mapping.csv")
TEST_MAPPING = os.path.join(TEST_DIR, "fragment_mapping.csv")

MODEL_SAVE_PATH = "saved_models/cnn/cnn_model.pth"
RESULTS_PATH = "results/cnn_results.json"
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 5  # Early stopping patience
NUM_WORKERS = 0  # Disk I/O workers for DataLoader


# ========== CNN Model ==========
class FragmentCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FragmentCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_size // 4), 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader. Returns loss, accuracy, predictions."""
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
    # Fit label encoder on all training labels (just reads CSV, no fragment data)
    print("📦 Fitting label encoder...")
    from sklearn.preprocessing import LabelEncoder
    all_train_labels = get_all_labels(TRAIN_MAPPING)
    label_enc = LabelEncoder()
    label_enc.fit(all_train_labels)
    class_names = label_enc.classes_
    num_classes = len(class_names)
    input_size = CHUNK_SIZE

    # Create lazy-loading datasets (only indexes file paths, doesn't load data)
    print("📦 Indexing datasets...")
    train_dataset = LazyFragmentDataset(TRAIN_MAPPING, TRAIN_DIR, label_enc)
    val_dataset = LazyFragmentDataset(VAL_MAPPING, VAL_DIR, label_enc)
    test_dataset = LazyFragmentDataset(TEST_MAPPING, TEST_DIR, label_enc)

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_test = len(test_dataset)
    print(f"📊 Train: {n_train}, Val: {n_val}, Test: {n_test}, Classes: {list(class_names)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Setup model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Compute class weights from training label frequencies
    label_counts = np.bincount(label_enc.transform(all_train_labels))
    class_weights = len(all_train_labels) / (num_classes * label_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    model = FragmentCNN(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"📐 Parameters: {total_params:,} total, {trainable_params:,} trainable, {model_size_mb:.2f} MB")

    # Training with validation monitoring + early stopping
    print(f"\n🚀 Training CNN on {n_train} samples (patience={PATIENCE})...")
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

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⏹️  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    train_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  ✅ Restored best model (val_loss={best_val_loss:.4f})")

    # Evaluate on validation set
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

    # Evaluate on test set
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

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds).tolist()

    # Dataset distribution
    train_label_counts = dict(Counter(all_train_labels))

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {
        "model": "CNN",
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
