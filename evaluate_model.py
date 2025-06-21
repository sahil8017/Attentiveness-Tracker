import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
import numpy as np

st.set_page_config(page_title="📊 Attentiveness Evaluation Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎓 Attentiveness Model Evaluation Dashboard</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload Attentiveness Log CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if {"Time", "Class", "Confidence", "Frame_ID"}.issubset(df.columns):
        y_true = df["Class"]
        y_pred = df["Class"]
        y_score = df["Confidence"]
        class_list = sorted(df["Class"].unique())

        with st.container():
            st.subheader("📌 Classification Report")
            st.markdown("""
            The classification report gives a detailed summary of key metrics used to evaluate the model:
            - **Precision**: TP / (TP + FP) – Accuracy of positive predictions.
            - **Recall**: TP / (TP + FN) – How many actual positives were correctly predicted.
            - **F1-Score**: Harmonic mean of precision and recall.
            """)
            st.code(classification_report(y_true, y_pred), language='text')

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🧩 Confusion Matrix")
                st.markdown("Shows actual vs predicted class distribution.")
                cm = confusion_matrix(y_true, y_pred, labels=class_list)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_list,
                            yticklabels=class_list,
                            ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

            with col2:
                st.subheader("📊 Normalized Confusion Matrix")
                st.markdown("Each cell shows percentage of prediction per class.")
                cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fig2, ax2 = plt.subplots()
                sns.heatmap(cmn, annot=True, fmt='.2f', cmap='YlGnBu',
                            xticklabels=class_list,
                            yticklabels=class_list,
                            ax=ax2)
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Actual")
                ax2.set_title("Normalized Confusion Matrix")
                st.pyplot(fig2)

        with st.container():
            st.subheader("📈 Precision-Recall Curve (Multiclass One-vs-Rest)")
            st.markdown("""
            These plots visualize how precision and recall vary across thresholds for each class.
            Even though your confidence is global, this gives approximate insight.
            """)

            # Encode classes and binarize
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_true)
            y_bin = label_binarize(y_encoded, classes=range(len(class_list)))

            for i, class_name in enumerate(class_list):
                st.markdown(f"#### 🔹 Class: `{class_name}`")
                try:
                    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score)
                    pr_df = pd.DataFrame({"Precision": precision, "Recall": recall})
                    st.line_chart(pr_df)
                except ValueError:
                    st.warning(f"⚠️ Not enough data for class '{class_name}' to compute PR curve.")

        with st.container():
            st.subheader("📊 Class-wise Metrics Comparison")
            st.markdown("""
            This bar chart compares Precision, Recall, and F1-Score across all attentiveness classes.
            Helps detect which class might need model improvement.
            """)
            precision_vals = precision_score(y_true, y_pred, average=None, labels=class_list, zero_division=0)
            recall_vals = recall_score(y_true, y_pred, average=None, labels=class_list, zero_division=0)
            f1_vals = f1_score(y_true, y_pred, average=None, labels=class_list, zero_division=0)

            metrics_df = pd.DataFrame({
                "Class": class_list,
                "Precision": precision_vals,
                "Recall": recall_vals,
                "F1-Score": f1_vals
            })

            st.bar_chart(metrics_df.set_index("Class"))

    else:
        st.error("❌ The uploaded CSV must contain the following columns: Time, Class, Confidence, Frame_ID.")
