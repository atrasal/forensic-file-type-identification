"""
Generate comparison graphs from model training results.

Reads all results/*.json files and produces:
  1. Model accuracy comparison (bar chart)
  2. Training loss curves (line plots, for models with training history)
  3. Confusion matrices (heatmaps per model)
  4. Per-class F1 scores (grouped bar chart)
  5. Model size vs accuracy (scatter plot)
  6. Training time vs accuracy (scatter plot)
  7. Dataset class distribution (bar chart)

Usage:
  python generate_graphs.py             # Generate all graphs
  python generate_graphs.py --show      # Also display interactively
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RESULTS_DIR = "results"
GRAPHS_DIR = "results/graphs"

# Color map for models
MODEL_COLORS = {
    "Ensemble (RF+XGBoost+ResNet)": "#e74c3c",
    "CNN": "#3498db",
    "ResNet": "#2ecc71",
    "Random Forest": "#e74c3c",
    "XGBoost": "#f39c12",
    "SVM": "#9b59b6",
    "MLP": "#1abc9c",
    "MLP (Features)": "#16a085",
    "LeNet": "#e67e22",
    "LSTM": "#34495e",
}


def load_results():
    """Load all results JSON files."""
    results = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_results.json"))):
        with open(path) as f:
            data = json.load(f)
            data["_file"] = os.path.basename(path)
            results.append(data)
    if not results:
        print("❌ No results found in results/")
        sys.exit(1)
    print(f"📊 Loaded {len(results)} model results: {[r['model'] for r in results]}")
    return results


def get_color(model_name):
    """Get a color for a model, with fallback."""
    return MODEL_COLORS.get(model_name, "#9b59b6")


def plot_accuracy_comparison(results):
    """Bar chart comparing accuracy, precision, recall, F1 across models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = [r["model"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    x = np.arange(len(models))
    width = 0.18

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [r.get(metric, 0) for r in results]
        bars = ax.bar(x + i * width, values, width, label=label, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper left', fontsize=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_training_curves(results):
    """Training and validation loss curves for models with history."""
    models_with_history = [r for r in results if "training_history" in r]
    if not models_with_history:
        print("  ⏭️  No training history found, skipping loss curves")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1 = axes[0]
    for r in models_with_history:
        name = r["model"]
        history = r["training_history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        color = get_color(name)
        ax1.plot(epochs, history["train_loss"], '-', color=color, label=f"{name} (train)", linewidth=2)
        ax1.plot(epochs, history["val_loss"], '--', color=color, label=f"{name} (val)", linewidth=2)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)

    # Accuracy curves
    ax2 = axes[1]
    for r in models_with_history:
        name = r["model"]
        history = r["training_history"]
        epochs = range(1, len(history["val_accuracy"]) + 1)
        color = get_color(name)
        ax2.plot(epochs, history["val_accuracy"], '-o', color=color, label=name, linewidth=2, markersize=4)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Accuracy", fontsize=12)
    ax2.set_title("Validation Accuracy Over Epochs", fontsize=14, fontweight='bold')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax2.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_confusion_matrices(results):
    """Confusion matrix heatmap for each model."""
    models_with_cm = [r for r in results if "confusion_matrix" in r]
    if not models_with_cm:
        print("  ⏭️  No confusion matrices found, skipping")
        return None

    n = len(models_with_cm)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, r in enumerate(models_with_cm):
        ax = axes[idx]
        cm = np.array(r["confusion_matrix"])

        # Get class names
        if "dataset_info" in r:
            classes = r["dataset_info"]["classes"]
        elif "classes" in r:
            classes = r["classes"]
        else:
            classes = [str(i) for i in range(cm.shape[0])]

        # Normalize by row (true labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        ax.set_title(f"{r['model']} - Confusion Matrix", fontsize=13, fontweight='bold')
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.tick_params(axis='both', labelsize=8)

    # Hide unused axes
    for idx in range(len(models_with_cm), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_per_class_f1(results):
    """Grouped bar chart of per-class F1 scores."""
    models_with_pcm = [r for r in results if "per_class_metrics" in r]
    if not models_with_pcm:
        print("  ⏭️  No per-class metrics found, skipping")
        return None

    # Get all classes from first model that has them
    all_classes = sorted(models_with_pcm[0]["per_class_metrics"].keys())

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_classes))
    width = 0.8 / len(models_with_pcm)

    for i, r in enumerate(models_with_pcm):
        f1_scores = [r["per_class_metrics"].get(cls, {}).get("f1", 0) for cls in all_classes]
        ax.bar(x + i * width, f1_scores, width, label=r["model"],
               color=get_color(r["model"]), alpha=0.85)

    ax.set_xlabel("File Type", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Scores by Model", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models_with_pcm) - 1) / 2)
    ax.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "per_class_f1.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_model_size_vs_accuracy(results):
    """Scatter plot of model size (MB) vs accuracy."""
    models_with_params = [r for r in results if "parameters" in r and "model_size_mb" in r.get("parameters", {})]
    if not models_with_params:
        print("  ⏭️  No parameter info found, skipping size vs accuracy")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in models_with_params:
        size = r["parameters"]["model_size_mb"]
        acc = r["accuracy"]
        color = get_color(r["model"])
        ax.scatter(size, acc, s=200, color=color, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(r["model"], (size, acc), textcoords="offset points",
                    xytext=(10, 10), fontsize=11, fontweight='bold')

    ax.set_xlabel("Model Size (MB)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Model Size vs Accuracy", fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "size_vs_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_training_time_vs_accuracy(results):
    """Scatter plot of training time vs accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        time_s = r.get("train_time_seconds", 0)
        acc = r["accuracy"]
        color = get_color(r["model"])
        ax.scatter(time_s / 60, acc, s=200, color=color, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(r["model"], (time_s / 60, acc), textcoords="offset points",
                    xytext=(10, 10), fontsize=11, fontweight='bold')

    ax.set_xlabel("Training Time (minutes)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Training Time vs Accuracy", fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "time_vs_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_dataset_distribution(results):
    """Bar chart of samples per class (from first model that has the data)."""
    for r in results:
        if "dataset_info" in r and "samples_per_class" in r["dataset_info"]:
            info = r["dataset_info"]
            break
    else:
        print("  ⏭️  No dataset info found, skipping distribution")
        return None

    classes = sorted(info["samples_per_class"].keys())
    counts = [info["samples_per_class"][c] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = sns.color_palette("husl", len(classes))
    bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=0.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel("File Type", fontsize=12)
    ax.set_ylabel("Training Samples", fontsize=12)
    ax.set_title(f"Dataset Distribution ({info['train_samples']:,} total training samples, "
                 f"{info['num_classes']} classes)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "dataset_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_summary_table(results):
    """Summary table image with all key metrics."""
    fig, ax = plt.subplots(figsize=(14, 2 + len(results) * 0.6))
    ax.axis('off')

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "Train Time", "Size (MB)", "Samples"]
    rows = []
    for r in results:
        size = "N/A"
        if "parameters" in r and "model_size_mb" in r["parameters"]:
            size = f"{r['parameters']['model_size_mb']:.1f}"

        train_samples = r.get("dataset_info", {}).get("train_samples", r.get("train_samples", "N/A"))
        train_time = r.get("train_time_seconds", 0)
        if train_time > 60:
            time_str = f"{train_time / 60:.1f} min"
        else:
            time_str = f"{train_time:.1f}s"

        rows.append([
            r["model"],
            f"{r['accuracy']:.2%}",
            f"{r['precision']:.2%}",
            f"{r['recall']:.2%}",
            f"{r['f1_score']:.2%}",
            time_str,
            size,
            f"{train_samples:,}" if isinstance(train_samples, int) else train_samples,
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

    ax.set_title("Model Performance Summary", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_feature_importance():
    """Plot top-20 feature importances from Random Forest model."""
    import joblib
    rf_path = "saved_models/random_forest/rf_model.joblib"
    if not os.path.exists(rf_path):
        print("  ⏭️  Random Forest model not found, skipping feature importance")
        return None

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from utils.feature_engineering import get_feature_names

        saved = joblib.load(rf_path)
        model = saved['model'] if isinstance(saved, dict) else saved
        importances = model.feature_importances_

        feature_names = get_feature_names()
        if len(feature_names) != len(importances):
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        top_indices = np.argsort(importances)[-20:][::-1]
        top_names = [feature_names[i] for i in top_indices]
        top_values = importances[top_indices]

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_names)))
        bars = ax.barh(range(len(top_names)), top_values, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
        ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=14, fontweight='bold')

        for bar, val in zip(bars, top_values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va='center', fontsize=9)

        plt.tight_layout()
        path = os.path.join(GRAPHS_DIR, "feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  ✅ {path}")
        return fig
    except Exception as e:
        print(f"  ⚠️  Could not generate feature importance: {e}")
        return None


def plot_f1_ranking(results):
    """Horizontal bar chart ranking models by F1 score."""
    sorted_results = sorted(results, key=lambda r: r.get("f1_score", 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r["model"] for r in sorted_results]
    f1s = [r.get("f1_score", 0) for r in sorted_results]
    colors = [get_color(n) for n in names]

    bars = ax.barh(names, f1s, color=colors, edgecolor='black', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel("Macro F1 Score", fontsize=12)
    ax.set_title("Model Ranking by F1 Score", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "f1_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_per_class_heatmap(results):
    """Heatmap of per-class F1 scores across all models."""
    models_with_pcm = [r for r in results if "per_class_metrics" in r]
    if not models_with_pcm:
        print("  ⏭️  No per-class metrics found, skipping heatmap")
        return None

    # Sort models by F1
    models_with_pcm = sorted(models_with_pcm, key=lambda r: r.get("f1_score", 0), reverse=True)
    all_classes = sorted(models_with_pcm[0]["per_class_metrics"].keys())
    model_names = [r["model"] for r in models_with_pcm]

    # Build matrix
    matrix = []
    for r in models_with_pcm:
        row = [r["per_class_metrics"].get(cls, {}).get("f1", 0) for cls in all_classes]
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(16, max(6, len(model_names) * 0.7)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=all_classes, yticklabels=model_names, ax=ax,
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={'label': 'F1 Score'})
    ax.set_title("Per-Class F1 Scores — All Models", fontsize=14, fontweight='bold')
    ax.set_xlabel("File Type", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "per_class_f1_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def plot_individual_confusion_matrices(results):
    """Generate individual confusion matrix PNG for each model."""
    models_with_cm = [r for r in results if "confusion_matrix" in r]
    if not models_with_cm:
        print("  ⏭️  No confusion matrices found, skipping individual CMs")
        return

    for r in models_with_cm:
        cm = np.array(r["confusion_matrix"])
        if "dataset_info" in r:
            classes = r["dataset_info"]["classes"]
        elif "classes" in r:
            classes = r["classes"]
        else:
            classes = [str(i) for i in range(cm.shape[0])]

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1, linewidths=0.3)
        ax.set_title(f"{r['model']} — Confusion Matrix (Normalized)", fontsize=14, fontweight='bold')
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.tick_params(axis='both', labelsize=9)

        safe_name = r['model'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        path = os.path.join(GRAPHS_DIR, f"confusion_matrix_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ {path}")


def plot_ablation_study(results):
    """Bar chart showing MLP raw vs MLP features vs RF."""
    mlp_raw = next((r for r in results if r["model"] == "MLP"), None)
    mlp_feat = next((r for r in results if r["model"] == "MLP (Features)"), None)
    rf = next((r for r in results if r["model"] == "Random Forest"), None)

    entries = []
    if mlp_raw:
        entries.append(("MLP\n(raw bytes)", mlp_raw["f1_score"], "#e74c3c"))
    if mlp_feat:
        entries.append(("MLP\n(317 features)", mlp_feat["f1_score"], "#f39c12"))
    if rf:
        entries.append(("Random Forest\n(317 features)", rf["f1_score"], "#2ecc71"))

    if len(entries) < 2:
        print("  ⏭️  Not enough models for ablation study, skipping")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    names, vals, colors = zip(*entries)
    bars = ax.bar(names, vals, color=colors, edgecolor='black', linewidth=0.8, width=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Feature Engineering Ablation Study", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "ablation_study.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate comparison graphs from training results")
    parser.add_argument('--show', action='store_true', help='Display graphs interactively')
    args = parser.parse_args()

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    results = load_results()

    print("\n📈 Generating graphs...")
    figs = []
    figs.append(plot_accuracy_comparison(results))
    figs.append(plot_f1_ranking(results))
    figs.append(plot_training_curves(results))
    figs.append(plot_confusion_matrices(results))
    figs.append(plot_per_class_f1(results))
    figs.append(plot_per_class_heatmap(results))
    figs.append(plot_model_size_vs_accuracy(results))
    figs.append(plot_training_time_vs_accuracy(results))
    figs.append(plot_dataset_distribution(results))
    figs.append(plot_summary_table(results))
    figs.append(plot_feature_importance())
    figs.append(plot_ablation_study(results))
    plot_individual_confusion_matrices(results)

    print(f"\n✅ All graphs saved to {GRAPHS_DIR}/")

    if args.show:
        try:
            matplotlib.use('macosx')
            plt.show()
        except Exception:
            print("⚠️  Cannot display interactively. Open graphs from: results/graphs/")


if __name__ == "__main__":
    main()

