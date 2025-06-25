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
import io # Required to read uploaded file as string buffer

st.set_page_config(page_title="📊 Attentiveness Evaluation Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎓 Attentiveness Model Evaluation Dashboard</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload Attentiveness Log CSV File", type=["csv"],
                                 help="Upload a CSV with 'Time', 'Class', 'Confidence', 'Frame_ID' columns.")

if uploaded_file:
    try:
        # Read the uploaded CSV file
        # Using io.StringIO to handle the uploaded file content correctly
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))

        # Define expected columns
        expected_cols = ["Time", "Class", "Confidence", "Frame_ID"]

        # If the dataframe has 4 columns and no explicit header was provided by the user in the file
        # (i.e., the first row looks like data), then we manually assign column names.
        if len(df.columns) == 4 and not all(col in df.columns for col in expected_cols):
             df.columns = expected_cols

        # Validate if required columns are present after reading/potential renaming
        if not all(col in df.columns for col in expected_cols):
            st.error("❌ The uploaded CSV must contain the following columns: Time, Class, Confidence, Frame_ID.")
            st.info("Please ensure your CSV has a header row with these exact names, or that the first row contains data in that order.")
            st.stop() # Stop execution if columns are missing

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded CSV file is empty or contains no data.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while reading the CSV: {e}")
        st.stop()

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # --- Ensure data quality for metrics ---
    if df.empty:
        st.warning("The uploaded CSV contains headers but no data rows to evaluate.")
        st.stop()

    # In a real model evaluation, you would have 'y_true' (ground truth) and 'y_pred' (model's prediction)
    # Since your log currently stores the *detected* class, we're using that for both true and predicted
    # for illustrative purposes of the evaluation metrics.
    # For a true evaluation, you'd need a dataset with actual labels.
    y_true = df["Class"]
    y_pred = df["Class"] # Assuming 'Class' is both the true label and the model's prediction from the log
    y_score = df["Confidence"]

    # Get unique classes and sort them for consistent plotting
    class_list = sorted(df["Class"].unique())

    # --- Classification Report ---
    with st.container():
        st.subheader("📌 Classification Report")
        st.markdown("""
        The classification report gives a detailed summary of key metrics used to evaluate the model:
        - **Precision**: TP / (TP + FP) – Accuracy of positive predictions.
        - **Recall**: TP / (TP + FN) – How many actual positives were correctly predicted.
        - **F1-Score**: Harmonic mean of precision and recall.
        """)
        # Ensure that the classification_report handles cases where a class might not be in y_true or y_pred
        # by passing `labels=class_list` and `zero_division=0`
        st.code(classification_report(y_true, y_pred, labels=class_list, zero_division=0), language='text')

    # --- Confusion Matrices ---
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧩 Confusion Matrix")
            st.markdown("Shows actual vs predicted class distribution.")
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=class_list)
            fig, ax = plt.subplots(figsize=(7, 6)) # Set size for consistency
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_list,
                        yticklabels=class_list,
                        ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig) # Display plot in Streamlit

        with col2:
            st.subheader("📊 Normalized Confusion Matrix")
            st.markdown("Each cell shows percentage of predictions per class, normalized by true class count.")
            # Calculate normalized confusion matrix
            cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig2, ax2 = plt.subplots(figsize=(7, 6)) # Set size for consistency
            sns.heatmap(cmn, annot=True, fmt='.2f', cmap='YlGnBu',
                        xticklabels=class_list,
                        yticklabels=class_list,
                        ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            ax2.set_title("Normalized Confusion Matrix")
            st.pyplot(fig2) # Display plot in Streamlit

    # --- Precision-Recall Curve ---
    with st.container():
        st.subheader("📈 Precision-Recall Curve (Multiclass One-vs-Rest)")
        st.markdown("""
        These plots visualize how precision and recall vary across thresholds for each class.
        Even though your confidence is global, this gives approximate insight.
        """)

        # Encode classes and binarize for multiclass PR curve
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_true)
        y_bin = label_binarize(y_encoded, classes=range(len(class_list)))

        # Handle cases where a class might only have one instance or constant score
        for i, class_name in enumerate(class_list):
            st.markdown(f"#### 🔹 Class: `{class_name}`")
            # Check if the class is actually present in the data to avoid errors
            if class_name not in y_true.unique():
                st.warning(f"⚠️ No data available for class '{class_name}' to compute PR curve.")
                continue

            # Filter y_score to only include scores relevant to this class, if possible.
            # Currently, y_score is global confidence. For true PR curves per class,
            # you'd need per-class confidence scores from your model.
            # As a workaround using the current data structure, we pass the global y_score
            # but understand its limitations for individual class PR curves.
            try:
                # Ensure y_score is treated as a 1D array for precision_recall_curve
                # and y_bin[:, i] is also 1D.
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score)
                pr_df = pd.DataFrame({"Precision": precision, "Recall": recall})
                st.line_chart(pr_df)
            except ValueError as ve:
                st.warning(f"⚠️ Could not compute PR curve for class '{class_name}': {ve}. This often happens if only one unique value is present in y_true or y_score for this class.")
            except Exception as e:
                st.warning(f"⚠️ An unexpected error occurred for class '{class_name}': {e}")


    # --- Class-wise Metrics Comparison ---
    with st.container():
        st.subheader("📊 Class-wise Metrics Comparison")
        st.markdown("""
        This bar chart compares Precision, Recall, and F1-Score across all attentiveness classes.
        Helps detect which class might need model improvement.
        """)
        # Compute metrics for each class
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

