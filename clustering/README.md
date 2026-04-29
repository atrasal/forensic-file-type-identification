# 📂 File Type Identification using ML (Digital Forensics)

This project focuses on File Fragment Analysis for Digital Forensics using Machine Learning. It identifies and clusters unknown file fragments based purely on byte-level statistical features, without relying on file headers or metadata.

---

## 🚀 Features

- **🔍 517-Dimensional Feature Extraction**
  - 256 → Byte Frequency
  - 256 → Byte-Pair Transitions
  - 5 → Statistical Features (Entropy, ASCII ratio, etc.)
- **🤖 Unsupervised Learning**
  - DBSCAN clustering
  - K-Means fallback (silhouette-based)
- **🧠 Type-Aware Clustering**
  - Groups fragments by inferred file type
- **📊 Automatic Result Generation**
  - Outputs clusters in `results/clusters.json`

---

## 🧠 How It Works

1. Load `.bin` file fragments.
2. Extract 517-dimensional feature vectors.
3. Apply clustering:
   - DBSCAN (primary)
   - K-Means (fallback)
4. Generate clustered output.

> 👉 **Note:** Full methodology explained in research document.

---

## 📁 Project Structure

```text
File-Type-Identification/
│
├── Train/                 # Dataset (fragment files)
├── sequence/
│   └── main.py            # Main clustering script
│
├── results/               # Output JSON
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone [https://github.com/Prathamesh-2005/File-Type-Identification.git](https://github.com/Prathamesh-2005/File-Type-Identification.git)
cd File-Type-Identification
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Activate it:**

- **Windows:**
  ```cmd
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 🔹 Default Run (Recommended)

```bash
python sequence/main.py
```

👉 **This will:**

- Automatically load fragments from `Train/`
- Extract features
- Perform clustering
- Save output in `results/clusters.json`

### 🔹 Optional Arguments

```bash
python sequence/main.py --max-fragments 50
```

**Available Options:**

| Argument          | Description                    |
| :---------------- | :----------------------------- |
| `--fragments-dir` | Custom dataset path            |
| `--max-fragments` | Limit number of fragments      |
| `--no-type-group` | Disable type-based grouping    |
| `--sub-cluster`   | Cluster within same file types |

### 🔹 Example (Advanced)

```bash
python sequence/main.py --max-fragments 100 --sub-cluster
```

---

## 📊 Output

- **Console:** Cluster summary
- **File:** `results/clusters.json`

**Example Output:**

```json
{
  "html/file1.bin": "Cluster 0",
  "pdf/file2.bin": "Cluster 1"
}
```

---

## 🧪 Dependencies (`requirements.txt`)

If not already created, your `requirements.txt` should look like this:

```text
numpy
scikit-learn
tqdm
```

---

## 📌 Key Concepts Used

- Shannon Entropy
- Byte-level Feature Engineering
- DBSCAN Clustering
- K-Means + Silhouette Score
---

## 🎯 Use Cases

- Digital Forensics
- Data Recovery
- Malware Analysis
- File Type Identification without Metadata

---

**🧑‍💻 Author:** Prathamesh Jadhav
