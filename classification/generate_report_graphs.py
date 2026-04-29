"""
Generate all graphs for the LaTeX research report.
Saves PNG files to report_images/ directory.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Output directory
OUT = Path("report_images")
OUT.mkdir(exist_ok=True)

# Set consistent style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})

# Load all results
results_dir = Path("results")
models = {}
for f in results_dir.glob("*_results.json"):
    with open(f) as fh:
        data = json.load(fh)
        name = data.get("model", f.stem.replace("_results", ""))
        models[name] = data

print(f"Loaded {len(models)} models: {list(models.keys())}")

# ─── 1. Model Comparison Bar Chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

model_order = [
    ("Ensemble", "Ensemble (RF+XGB+ResNet)"),
    ("XGBoost", "XGBoost"),
    ("Random Forest", "Random Forest"),
    ("ResNet", "ResNet"),
    ("MLP (Features)", "MLP (Features)"),
    ("CNN", "CNN"),
    ("LeNet", "LeNet"),
    ("SVM", "SVM"),
    ("LSTM", "LSTM"),
    ("MLP", "MLP (Raw)"),
]

names = []
accs, precs, recs, f1s = [], [], [], []

for key, label in model_order:
    if key in models:
        m = models[key]
        names.append(label)
        accs.append(m['accuracy'] * 100)
        precs.append(m['precision'] * 100)
        recs.append(m['recall'] * 100)
        f1s.append(m['f1_score'] * 100)

x = np.arange(len(names))
w = 0.2

bars1 = ax.bar(x - 1.5*w, accs, w, label='Accuracy', color='#3498db', alpha=0.85)
bars2 = ax.bar(x - 0.5*w, precs, w, label='Precision', color='#2ecc71', alpha=0.85)
bars3 = ax.bar(x + 0.5*w, recs, w, label='Recall', color='#e74c3c', alpha=0.85)
bars4 = ax.bar(x + 1.5*w, f1s, w, label='F1 Score', color='#9b59b6', alpha=0.85)

ax.set_ylabel('Score (%)')
ax.set_title('Model Performance Comparison (Test Set)')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=35, ha='right')
ax.legend(loc='upper right')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", bbox_inches='tight')
plt.close()
print("✅ model_comparison.png")

# ─── 2. F1 Score Ranking ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

sorted_models = sorted(zip(names, f1s), key=lambda x: x[1], reverse=True)
s_names, s_f1s = zip(*sorted_models)

colors = ['#2ecc71' if f > 70 else '#f39c12' if f > 50 else '#e74c3c' for f in s_f1s]
bars = ax.barh(range(len(s_names)), s_f1s, color=colors, alpha=0.85, edgecolor='white')

for i, (bar, val) in enumerate(zip(bars, s_f1s)):
    ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(s_names)))
ax.set_yticklabels(s_names)
ax.set_xlabel('Macro F1 Score (%)')
ax.set_title('Model Ranking by Macro F1 Score')
ax.set_xlim(0, 95)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "f1_ranking.png", bbox_inches='tight')
plt.close()
print("✅ f1_ranking.png")

# ─── 3. Per-Class F1 Heatmap (Top 3 Models) ──────────────────
top_models = ["Ensemble", "XGBoost", "Random Forest"]
available = [m for m in top_models if m in models]

if available:
    classes = list(models[available[0]]['per_class_metrics'].keys())

    data_matrix = []
    for m in available:
        row = [models[m]['per_class_metrics'][c]['f1'] for c in classes]
        data_matrix.append(row)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(available)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)

    for i in range(len(available)):
        for j in range(len(classes)):
            val = data_matrix[i][j]
            color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='F1 Score', shrink=0.8)
    ax.set_title('Per-Class F1 Scores — Top 3 Models')
    plt.tight_layout()
    plt.savefig(OUT / "per_class_f1_heatmap.png", bbox_inches='tight')
    plt.close()
    print("✅ per_class_f1_heatmap.png")

# ─── 4. Confusion Matrix (Ensemble) ──────────────────────────
if "Ensemble" in models and "confusion_matrix" in models["Ensemble"]:
    cm = np.array(models["Ensemble"]["confusion_matrix"])
    classes = list(models["Ensemble"]["per_class_metrics"].keys())
    
    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                annot_kws={'size': 7})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix — Ensemble Model')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix_ensemble.png", bbox_inches='tight')
    plt.close()
    print("✅ confusion_matrix_ensemble.png")

# ─── 5. Feature Engineering Ablation ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ablation_data = {
    'MLP (Raw Bytes)\n4096-dim': 34.5,
    'MLP (317 Features)\n317-dim': 61.3,
    'Random Forest\n317 Features': 77.3,
    'XGBoost\n317 Features': 79.8,
    'Ensemble\nRF+XGB+ResNet': 81.0,
}

colors_abl = ['#e74c3c', '#f39c12', '#2ecc71', '#27ae60', '#1a8a4a']
bars = ax.bar(ablation_data.keys(), ablation_data.values(), color=colors_abl, 
              alpha=0.85, edgecolor='white', width=0.6)

for bar, val in zip(bars, ablation_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
            ha='center', fontweight='bold', fontsize=10)

ax.set_ylabel('Macro F1 Score (%)')
ax.set_title('Impact of Feature Engineering on Classification Performance')
ax.set_ylim(0, 95)
ax.grid(axis='y', alpha=0.3)

# Add annotation arrow
ax.annotate('+26.8 pp', xy=(1, 61.3), xytext=(0.5, 50),
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2),
            fontsize=10, color='#e67e22', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT / "ablation_study.png", bbox_inches='tight')
plt.close()
print("✅ ablation_study.png")

# ─── 6. Training Loss Curves (DL Models) ──────────────────────
dl_models_with_history = {}
for name, data in models.items():
    if 'training_history' in data and 'train_loss' in data['training_history']:
        dl_models_with_history[name] = data

if dl_models_with_history:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors_dl = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, (name, data) in enumerate(dl_models_with_history.items()):
        hist = data['training_history']
        color = colors_dl[i % len(colors_dl)]
        epochs = range(1, len(hist['train_loss']) + 1)
        
        ax1.plot(epochs, hist['train_loss'], label=name, color=color, linewidth=1.5)
        if 'val_loss' in hist:
            ax2.plot(range(1, len(hist['val_loss'])+1), hist['val_loss'], 
                    label=name, color=color, linewidth=1.5)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Curves')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT / "training_curves.png", bbox_inches='tight')
    plt.close()
    print("✅ training_curves.png")

# ─── 7. Class Distribution ───────────────────────────────────
if "Ensemble" in models and "dataset_info" in models["Ensemble"]:
    info = models["Ensemble"]["dataset_info"]
    spc = info.get("samples_per_class", {})
    
    if spc:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        sorted_items = sorted(spc.items(), key=lambda x: x[1], reverse=True)
        cls_names = [item[0].upper() for item in sorted_items]
        cls_counts = [item[1] for item in sorted_items]
        
        colors_cls = plt.cm.viridis(np.linspace(0.2, 0.9, len(cls_names)))
        bars = ax.bar(cls_names, cls_counts, color=colors_cls, alpha=0.85, edgecolor='white')
        
        for bar, val in zip(bars, cls_counts):
            if val > 30000:
                ax.text(bar.get_x() + bar.get_width()/2, val + 1000, f'{val//1000}K',
                        ha='center', fontsize=7, fontweight='bold')
        
        ax.set_ylabel('Number of Fragments')
        ax.set_title('Dataset Distribution Across 22 File Types')
        ax.set_yscale('log')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "class_distribution.png", bbox_inches='tight')
        plt.close()
        print("✅ class_distribution.png")

# ─── 8. Pipeline Architecture Diagram ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 3))
ax.set_xlim(0, 14)
ax.set_ylim(0, 3)
ax.axis('off')

boxes = [
    (1, 1.2, 'Raw Files\n(22 types)', '#3498db'),
    (3.5, 1.2, 'Fragmenter\n(4096-byte)', '#e74c3c'),
    (6, 1.2, 'Header/Footer\nCleaner', '#f39c12'),
    (8.5, 1.2, 'Stratified\nSplitter\n(70/15/15)', '#2ecc71'),
    (11, 1.2, 'Feature Eng.\n(317 features)', '#9b59b6'),
    (13, 1.2, 'ML Models\n(10 archs.)', '#1abc9c'),
]

for x, y, text, color in boxes:
    rect = mpatches.FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=8,
            color='white', fontweight='bold')

for i in range(len(boxes)-1):
    x1 = boxes[i][0] + 0.7
    x2 = boxes[i+1][0] - 0.7
    ax.annotate('', xy=(x2, 1.2), xytext=(x1, 1.2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

ax.set_title('End-to-End Data Pipeline Architecture', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(OUT / "pipeline_architecture.png", bbox_inches='tight')
plt.close()
print("✅ pipeline_architecture.png")

print(f"\n🎉 All graphs saved to {OUT}/")
print(f"   Total: {len(list(OUT.glob('*.png')))} images")
