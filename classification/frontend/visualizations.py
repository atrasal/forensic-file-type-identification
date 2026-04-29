"""
Visualization and plotting functions for the dashboard.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def plot_comparison_bars(df):
    """Create comparison bar charts"""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df[metric],
            name=metric,
            text=[f"{v*100:.1f}%" for v in df[metric]],
            textposition="auto",
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_radar_comparison(df):
    """Create radar chart for model comparison"""
    fig = go.Figure()
    
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    for idx, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=metrics,
            fill='toself',
            name=row["Model"],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=600,
        title="Model Performance - Radar Comparison"
    )
    
    return fig


def plot_training_history(model_data, model_name):
    """Plot training and validation loss curves"""
    history = model_data.get("training_history", {})
    
    if not history:
        st.warning(f"No training history available for {model_name}")
        return None
    
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    fig = go.Figure()
    
    if "train_loss" in history:
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history["train_loss"],
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='#1f77b4', width=2),
        ))
    
    if "val_loss" in history:
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history["val_loss"],
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#ff7f0e', width=2),
        ))
    
    fig.update_layout(
        title=f"{model_name} - Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_accuracy_history(model_data, model_name):
    """Plot validation accuracy over epochs"""
    history = model_data.get("training_history", {})
    
    if "val_accuracy" not in history:
        return None
    
    epochs = range(1, len(history["val_accuracy"]) + 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=history["val_accuracy"],
        mode='lines+markers',
        name='Validation Accuracy',
        fill='tozeroy',
        line=dict(color='#2ca02c', width=2),
    ))
    
    fig.update_layout(
        title=f"{model_name} - Validation Accuracy Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode='x',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_confusion_matrix(model_data, model_name):
    """Plot confusion matrix as heatmap"""
    confusion_mat = model_data.get("confusion_matrix")
    
    if not confusion_mat:
        st.warning(f"No confusion matrix available for {model_name}")
        return None
    
    cm = np.array(confusion_mat)
    
    per_class = model_data.get("per_class_metrics", {})
    labels = sorted(per_class.keys())
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 8},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=f"{model_name} - Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=600,
        width=800,
    )
    
    return fig


def plot_per_class_metrics(model_data, model_name):
    """Plot per-class precision, recall, and F1 score"""
    per_class = model_data.get("per_class_metrics", {})
    
    if not per_class:
        st.warning(f"No per-class metrics available for {model_name}")
        return None
    
    rows = []
    for cls, metrics in per_class.items():
        rows.append({
            "Class": cls,
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
            "Support": metrics.get("support", 0)
        })
    
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=df["Class"], y=df["Precision"], name="Precision"))
    fig.add_trace(go.Bar(x=df["Class"], y=df["Recall"], name="Recall"))
    fig.add_trace(go.Bar(x=df["Class"], y=df["F1"], name="F1 Score"))
    
    fig.update_layout(
        title=f"{model_name} - Per-Class Metrics",
        xaxis_title="File Type",
        yaxis_title="Score",
        barmode='group',
        height=400,
        hovermode='x unified',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    return fig


def display_model_metrics(model_data, model_name):
    """Display key metrics as formatted cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    def format_percentage(value):
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%"
    
    with col1:
        st.metric(
            "Accuracy",
            format_percentage(model_data.get("accuracy")),
            delta=format_percentage(model_data.get("val_accuracy", model_data.get("accuracy")))
        )
    
    with col2:
        st.metric(
            "Precision",
            format_percentage(model_data.get("precision")),
            delta=format_percentage(model_data.get("val_precision", model_data.get("precision")))
        )
    
    with col3:
        st.metric(
            "Recall",
            format_percentage(model_data.get("recall")),
            delta=format_percentage(model_data.get("val_recall", model_data.get("recall")))
        )
    
    with col4:
        st.metric(
            "F1 Score",
            format_percentage(model_data.get("f1_score")),
            delta=format_percentage(model_data.get("val_f1_score", model_data.get("f1_score")))
        )


def plot_confidence_gauge(confidence_score):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_score * 100,
        title={'text': "Confidence %"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    fig.update_layout(height=300, width=300)
    return fig


def plot_confidence_comparison(df_predictions):
    """Bar chart of prediction confidence across models"""
    fig = go.Figure()
    
    conf_col = 'Confidence Score' if 'Confidence Score' in df_predictions.columns else 'Confidence'
    text_col = 'Confidence' if 'Confidence' in df_predictions.columns else None
    text_vals = df_predictions[text_col] if text_col and df_predictions[text_col].dtype == object else [f"{v:.1%}" for v in df_predictions[conf_col]]
    
    fig.add_trace(go.Bar(
        x=df_predictions['Model'],
        y=df_predictions[conf_col],
        text=text_vals,
        textposition='auto',
        marker=dict(
            color=df_predictions[conf_col],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        hovertemplate='<b>%{x}</b><br>Confidence: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Prediction Confidence by Model",
        xaxis_title="Model",
        yaxis_title="Confidence Score",
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_prediction_pie(labels_top, scores_top):
    """Mini pie chart for top predictions"""
    fig = go.Figure(data=[go.Pie(
        labels=labels_top,
        values=scores_top,
        textposition='inside',
        textinfo='label+percent'
    )])
    
    fig.update_layout(height=300, width=300)
    return fig
