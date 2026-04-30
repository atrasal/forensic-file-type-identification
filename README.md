# File Type Identification Using Machine Learning

A machine learning system that identifies file types from raw binary fragments — without relying on file headers, footers, or extensions. Useful for **digital forensics**, **data recovery**, and **malware analysis**.

This project combines two complementary approaches:

| Approach | Directory | Technique | Author |
|---|---|---|---|
| **Supervised Classification** | [`classification/`](classification/) | 10 ML models trained on labelled fragments | Aadi |
| **Unsupervised Clustering** | [`clustering/`](clustering/) | DBSCAN / K-Means clustering on byte-level features | Prathamesh Jadhav |

---

## Part 1 — Supervised Classification ([`classification/`](classification/))

Trains and evaluates **10 machine learning models** (Random Forest, XGBoost, SVM, CNN, ResNet, MLP, LeNet, LSTM, MLP-features, and a Weighted Ensemble) on **22 file types** with **~1M fragments**.

**Highlights:**
- 317 engineered features (byte histograms, entropy, bigrams, trigrams, block entropy, compression ratio, chi-squared)
- Class-weighted loss functions for handling imbalanced data (up to 222:1 ratio)
- Best result: **81.0% macro F1** (Weighted Ensemble of RF + XGBoost + ResNet)
- Streamlit dashboard for interactive model comparison and file prediction
- Full data pipeline: raw files → fragmenter → header/footer cleaner → stratified splitter

📖 **[Full details & instructions →](classification/README.md)**

---

## Part 2 — Unsupervised Clustering ([`clustering/`](clustering/))

Groups unknown file fragments into clusters based purely on **byte-level statistical features**, without any training labels.

**Highlights:**
- 517-dimensional feature extraction (byte frequency + byte-pair transitions + entropy + ASCII ratio)
- DBSCAN clustering with K-Means fallback (silhouette-guided k selection)
- File-type-aware pre-grouping with optional sub-clustering
- Comprehensive visualizations: t-SNE, UMAP, silhouette plots, similarity heatmaps
- Clustering quality metrics: Purity, ARI, NMI

📖 **[Full details & instructions →](clustering/README.md)**

---

## Quick Start

Each part has its own setup. Navigate to the respective directory:

```bash
# Clone
git clone <[github](https://github.com/atrasal/forensic-file-type-identification)>
cd forensics

# --- Classification ---
cd classification
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Train a model, e.g.:
python models/random_forest/train.py
# Launch dashboard:
streamlit run frontend/app.py

# --- Clustering ---
cd ../clustering
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Run clustering:
python sequence/main.py
# Generate visualizations:
python sequence/visualization.py
```

## Supported File Types (22)

| Category | Types |
|---|---|
| Archives | 7zip, APK, GZIP, TAR |
| Documents | PDF, RTF, PPTX, EPS |
| Audio/Video | MP3, MP4, SWF |
| Images | TIF, BMP, GIF |
| Code | CSS, HTML, JavaScript, JSON |
| Executables | ELF, BIN, EXE, DLL |

## Project Structure

```
forensics/
├── classification/              # Supervised ML classification
│   ├── datasets/                # Data pipeline scripts & fragments
│   ├── models/                  # 10 model training scripts
│   │   ├── random_forest/
│   │   ├── xgboost/
│   │   ├── svm/
│   │   ├── cnn/
│   │   ├── resnet/
│   │   ├── mlp/
│   │   ├── lenet/
│   │   ├── lstm/
│   │   ├── mlp_features/
│   │   └── ensemble/
│   ├── utils/                   # Shared data loading & feature engineering
│   ├── frontend/                # Streamlit dashboard
│   ├── saved_models/            # Trained checkpoints
│   ├── results/                 # Metrics JSON + graphs
│   ├── predict.py               # CLI inference
│   ├── requirements.txt
│   ├── README.md
│   └── RESEARCH.md
│
├── clustering/                  # Unsupervised fragment clustering
│   ├── sequence/
│   │   ├── main.py              # DBSCAN + K-Means clustering
│   │   ├── clustering_improved.py  # Enhanced PCA-based clustering
│   │   └── visualization.py    # t-SNE, UMAP, heatmaps
│   ├── results/                 # Cluster assignments & visualizations
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── README.md                    # ← You are here
```

## Requirements

- Python 3.10+
- macOS / Linux
- ~4 GB RAM for classification training

## License

MIT
