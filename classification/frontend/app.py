"""
File Type Identification - Model Comparison Dashboard
Main Streamlit application
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from utils import clean_file_data, create_fragments
from models import load_models, predict_file
from visualizations import (
    plot_comparison_bars, plot_radar_comparison, plot_training_history,
    plot_accuracy_history, plot_confusion_matrix, plot_per_class_metrics,
    display_model_metrics, plot_confidence_gauge, plot_confidence_comparison,
    plot_prediction_pie
)

warnings.filterwarnings('ignore')

# ── Page Configuration ──
st.set_page_config(
    page_title="ForensicID — File Type Identification Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for a clean, professional light theme ──
st.markdown("""
<style>
    /* ─── Global ─── */
    .block-container { padding-top: 1.5rem; }

    /* ─── Header banner ─── */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6aa6 50%, #3b82c4 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(30,58,95,0.15);
    }
    .hero-banner h1 { margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
    .hero-banner p  { margin: 0.4rem 0 0; opacity: 0.9; font-size: 0.95rem; }

    /* ─── Stat cards ─── */
    .stat-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
    .stat-card {
        flex: 1;
        min-width: 140px;
        background: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-card .label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .stat-card .value { font-size: 1.5rem; font-weight: 700; color: #1e3a5f; margin-top: 0.25rem; }
    .stat-card .sub   { font-size: 0.75rem; color: #9ca3af; margin-top: 0.15rem; }

    /* ─── Section headings ─── */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e3a5f;
        border-left: 4px solid #2d6aa6;
        padding-left: 0.75rem;
        margin: 1.5rem 0 0.75rem;
    }

    /* ─── Result card for predictions ─── */
    .result-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
    }
    .result-card h3 { margin: 0 0 0.25rem; color: #166534; font-size: 1.1rem; }
    .result-card .big { font-size: 2rem; font-weight: 700; color: #15803d; }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.75rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }

    /* ─── Misc polish ─── */
    .stDataFrame { border-radius: 8px; }
    div[data-testid="stMetric"] { background: #f8f9fb; border-radius: 8px; padding: 0.75rem; border: 1px solid #e5e7eb; }
    div[data-testid="stMetric"] label { font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# ── Pretty Display Names ──
MODEL_DISPLAY_NAMES = {
    'cnn': 'CNN',
    'resnet': 'ResNet',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'svm': 'SVM',
    'mlp': 'MLP (raw bytes)',
    'mlp_features': 'MLP (features)',
    'lenet': 'LeNet',
    'lstm': 'LSTM',
    'ensemble': 'Ensemble',
}

def dn(key):
    """Get pretty display name for a model key."""
    return MODEL_DISPLAY_NAMES.get(key, key.replace('_', ' ').title())


# ── Data Loading ──
@st.cache_data
def load_all_results():
    """Load all model results from JSON files."""
    models_data = {}
    for json_file in sorted(RESULTS_DIR.glob("*_results.json")):
        try:
            key = json_file.stem.replace("_results", "")
            models_data[key] = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"Could not load {json_file.name}: {e}")
    return models_data


def create_comparison_df(models_data):
    """Create a comparison DataFrame sorted by F1."""
    rows = []
    for key, data in models_data.items():
        rows.append({
            "Model": dn(key),
            "Accuracy": data.get("accuracy", 0),
            "Precision": data.get("precision", 0),
            "Recall": data.get("recall", 0),
            "F1 Score": data.get("f1_score", 0),
            "Val F1": data.get("val_f1_score", 0),
        })
    df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>🔬 ForensicID — File Type Identification</h1>
    <p>Classify unknown binary fragments into 22 file types using 10 ML models · 1M+ training fragments · 317 engineered features</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to",
        ["📊 Compare Models", "🔍 Analyze Model", "📁 Upload & Predict"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This dashboard lets you compare ML models "
        "trained on binary file fragments for digital forensic file-type "
        "identification. Upload any file to see which type it is."
    )
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#9ca3af;font-size:0.7rem;'>"
        "ForensicID · Built with Streamlit & Plotly"
        "</div>",
        unsafe_allow_html=True
    )

# Load data
models_data = load_all_results()
if not models_data:
    st.error("❌ No model results found in `results/` directory. Train some models first.")
    st.stop()


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
if page == "📊 Compare Models":
    df = create_comparison_df(models_data)

    # Top-line stats
    best_f1 = df.iloc[0]
    best_acc = df.loc[df["Accuracy"].idxmax()]
    n_models = len(df)

    st.markdown('<div class="section-title">Performance Overview</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="label">Best F1 Score</div>
            <div class="value">{best_f1['F1 Score']*100:.1f}%</div>
            <div class="sub">{best_f1['Model']}</div>
        </div>
        <div class="stat-card">
            <div class="label">Best Accuracy</div>
            <div class="value">{best_acc['Accuracy']*100:.1f}%</div>
            <div class="sub">{best_acc['Model']}</div>
        </div>
        <div class="stat-card">
            <div class="label">Models Trained</div>
            <div class="value">{n_models}</div>
            <div class="sub">across 4 paradigms</div>
        </div>
        <div class="stat-card">
            <div class="label">File Types</div>
            <div class="value">22</div>
            <div class="sub">classes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Leaderboard table
    st.markdown('<div class="section-title">Leaderboard</div>', unsafe_allow_html=True)
    
    # Add rank column
    leaderboard = df.copy()
    leaderboard.insert(0, "Rank", [f"#{i+1}" for i in range(len(leaderboard))])
    
    st.dataframe(
        leaderboard.style.format({
            "Accuracy": "{:.1%}",
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "F1 Score": "{:.1%}",
            "Val F1": "{:.1%}",
        }).highlight_max(subset=["Accuracy", "F1 Score"], color='#d1fae5')
        .highlight_min(subset=["F1 Score"], color='#fee2e2'),
        use_container_width=True,
        hide_index=True,
    )

    # Charts
    st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)
    tab_bar, tab_radar = st.tabs(["📊 Bar Chart", "🎯 Radar Chart"])

    with tab_bar:
        st.plotly_chart(plot_comparison_bars(df), use_container_width=True, key='bars')

    with tab_radar:
        st.plotly_chart(plot_radar_comparison(df), use_container_width=True, key='radar')


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — INDIVIDUAL MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Analyze Model":
    
    # Model selector
    sorted_keys = sorted(models_data.keys(), key=lambda k: models_data[k].get('f1_score', 0), reverse=True)
    selected = st.selectbox(
        "Choose a model",
        sorted_keys,
        format_func=lambda x: f"{dn(x)}  —  F1: {models_data[x].get('f1_score',0)*100:.1f}%"
    )
    
    data = models_data[selected]
    name = dn(selected)
    
    st.markdown(f'<div class="section-title">{name} — Performance Metrics</div>', unsafe_allow_html=True)
    display_model_metrics(data, name)

    # Model parameters (if available)
    if "parameters" in data:
        params = data["parameters"]
        st.markdown(f'<div class="section-title">Architecture</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Parameters", f"{params.get('total_params', 0):,}")
        c2.metric("Trainable", f"{params.get('trainable_params', 0):,}")
        c3.metric("Model Size", f"{params.get('model_size_mb', 0):.2f} MB")

    # Training curves
    st.markdown(f'<div class="section-title">Training History</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        fig = plot_training_history(data, name)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key='loss')
    with col_r:
        fig = plot_accuracy_history(data, name)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key='acc')

    # Confusion matrix
    st.markdown(f'<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    fig = plot_confusion_matrix(data, name)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key='cm')

    # Per-class
    st.markdown(f'<div class="section-title">Per-Class F1 Scores</div>', unsafe_allow_html=True)
    fig = plot_per_class_metrics(data, name)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key='perclass')

    if "per_class_metrics" in data:
        with st.expander("📋 Detailed per-class metrics table", expanded=False):
            pc_df = pd.DataFrame([
                {
                    "File Type": cls.upper(),
                    "Precision": m.get("precision", 0),
                    "Recall": m.get("recall", 0),
                    "F1": m.get("f1", 0),
                    "Samples": m.get("support", 0),
                }
                for cls, m in data["per_class_metrics"].items()
            ]).sort_values("F1", ascending=False)
            st.dataframe(
                pc_df.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — UPLOAD & PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "📁 Upload & Predict":

    st.markdown('<div class="section-title">Upload a File for Type Identification</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload any file (or a `.bin` fragment). The system will fragment it, "
        "extract features, and run **all available models** to predict the file type."
    )

    # Load models once
    with st.spinner("Loading models…"):
        loaded_models, class_labels = load_models()

    if not loaded_models:
        st.warning("No models found in `saved_models/`. Train models first.")
        st.stop()

    # Upload
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        upload_mode = st.radio(
            "Upload mode",
            ["Any file", "Binary fragment (.bin)"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    if upload_mode == "Binary fragment (.bin)":
        uploaded = st.file_uploader("Drop a .bin fragment here", type=["bin"], key="bin_up")
    else:
        uploaded = st.file_uploader("Drop any file here", type=None, key="any_up")

    if uploaded is None:
        # Show available models while waiting
        st.info("👆 Upload a file above to get started.")
        st.markdown('<div class="section-title">Available Models</div>', unsafe_allow_html=True)
        avail = []
        for k in sorted(models_data.keys(), key=lambda k: models_data[k].get('f1_score',0), reverse=True):
            d = models_data[k]
            avail.append({
                "Model": dn(k),
                "F1 Score": f"{d.get('f1_score',0)*100:.1f}%",
                "Accuracy": f"{d.get('accuracy',0)*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(avail), use_container_width=True, hide_index=True)
        st.stop()

    # ── File uploaded ──
    file_bytes = uploaded.read()
    file_ext = uploaded.name.rsplit('.', 1)[-1].upper() if '.' in uploaded.name else "—"

    st.markdown('<div class="section-title">File Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("File Name", uploaded.name[:35])
    c2.metric("Size", f"{len(file_bytes)/1024:.1f} KB")
    c3.metric("Extension", file_ext)

    # Fragmentation
    with st.spinner("Creating fragments…"):
        fragments = create_fragments(file_bytes, chunk_size=4096, num_fragments=5)

    # Preprocessing (direct files only)
    if upload_mode != "Binary fragment (.bin)":
        with st.spinner("Detecting file type & cleaning headers/footers…"):
            cleaned_data, detection, clean_stats = clean_file_data(file_bytes, uploaded.name)

        if clean_stats['bytes_removed'] > 0:
            with st.expander("🧹 Preprocessing details", expanded=False):
                p1, p2, p3 = st.columns(3)
                p1.metric("Original", f"{clean_stats['original_size']:,} B")
                p2.metric("Cleaned", f"{clean_stats['cleaned_size']:,} B")
                p3.metric("Removed", f"{clean_stats['bytes_removed']:,} B ({clean_stats['removal_percentage']:.1f}%)")
    else:
        cleaned_data = file_bytes

    # ── Run Predictions ──
    st.markdown('<div class="section-title">Predictions</div>', unsafe_allow_html=True)

    with st.spinner("Running all models…"):
        predictions = predict_file(file_bytes, loaded_models, class_labels, cleaned_data=cleaned_data)

    if not predictions:
        st.error("Could not generate predictions. Check if the file is valid.")
        st.stop()

    # ── Best prediction highlight ──
    best_name, best_info = max(predictions.items(), key=lambda x: x[1]['confidence'])
    conf = best_info['confidence']

    st.markdown(f"""
    <div class="result-card">
        <h3>🎯 Most Confident Prediction</h3>
        <div class="big">.{best_info['predicted_class'].upper()}</div>
        <p style="margin:0.25rem 0 0;color:#166534;">
            by <strong>{dn(best_name)}</strong> · Confidence: <strong>{conf*100:.1f}%</strong>
            {"  ✅ High" if conf > 0.8 else "  ⚠️ Low" if conf < 0.5 else ""}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Voting summary ──
    from collections import Counter
    votes = Counter(p['predicted_class'] for p in predictions.values())
    consensus, consensus_n = votes.most_common(1)[0]
    
    v1, v2, v3 = st.columns(3)
    v1.metric("🗳️ Consensus", consensus.upper())
    v2.metric("Agreement", f"{consensus_n}/{len(predictions)} models")
    v3.metric("Agreement %", f"{consensus_n/len(predictions)*100:.0f}%")

    # ── All model predictions table ──
    st.markdown('<div class="section-title">All Model Predictions</div>', unsafe_allow_html=True)
    
    pred_rows = []
    for mname, pinfo in sorted(predictions.items(), key=lambda x: x[1]['confidence'], reverse=True):
        pred_rows.append({
            "Model": dn(mname),
            "Prediction": pinfo['predicted_class'].upper(),
            "Confidence": pinfo['confidence'],
        })
    df_pred = pd.DataFrame(pred_rows)
    st.dataframe(
        df_pred.style.format({"Confidence": "{:.1%}"})
        .highlight_max(subset=["Confidence"], color="#d1fae5"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Confidence bar chart ──
    st.plotly_chart(plot_confidence_comparison(df_pred.rename(columns={"Confidence": "Confidence Score"})), use_container_width=True, key='conf_bars')

    # ── Per-model detail tabs ──
    with st.expander("🔬 Detailed per-model probabilities", expanded=False):
        tabs = st.tabs([dn(m) for m in predictions.keys()])
        for tab, (mname, pinfo) in zip(tabs, predictions.items()):
            with tab:
                top_idx = np.argsort(pinfo['probabilities'])[-5:][::-1]
                top_data = []
                for rank, idx in enumerate(top_idx, 1):
                    if idx < len(class_labels):
                        top_data.append({
                            "Rank": rank,
                            "File Type": class_labels[idx].upper(),
                            "Probability": pinfo['probabilities'][idx],
                        })
                if top_data:
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.dataframe(
                            pd.DataFrame(top_data).style.format({"Probability": "{:.2%}"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                    with c2:
                        labels = [r['File Type'] for r in top_data]
                        scores = [r['Probability'] for r in top_data]
                        st.plotly_chart(plot_prediction_pie(labels, scores), use_container_width=True, key=f"pie_{mname}")


# ── Footer ──
st.markdown(
    '<div class="footer">ForensicID · File Type Identification Dashboard · Built with Streamlit & Plotly</div>',
    unsafe_allow_html=True
)
