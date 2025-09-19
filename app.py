import streamlit as st
import numpy as np
import scipy.io
import pywt
from scipy.fft import rfft
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# ------------------ Load trained models ------------------
rf_model = joblib.load("models/random_forest.pkl")
svm_model = joblib.load("models/svm.pkl")

FIXED_LEN = 1024

def load_mat_signal(file):
    mat = scipy.io.loadmat(file)
    return mat['sig'].squeeze()

def pad_or_crop(signal, target_len=FIXED_LEN):
    if len(signal) >= target_len:
        return signal[:target_len]
    else:
        return np.pad(signal, (0, target_len-len(signal)), 'constant')

def extract_fft_features(signal):
    yf = np.abs(rfft(signal))
    return [np.mean(yf), np.std(yf), np.max(yf)]

def extract_wavelet_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    feats = []
    for c in coeffs:
        feats.extend([np.mean(c), np.std(c)])
    return feats

def extract_features(signal):
    feats = []
    feats.extend(extract_fft_features(signal))
    feats.extend(extract_wavelet_features(signal))
    return np.array(feats).reshape(1, -1)

# ------------------ Fault Mapping ------------------
fault_mapping = {
    "LLG": "Line-to-Line-to-Ground Fault",
    "LLL": "Three-Phase Fault",
    "SLG": "Single-Line-to-Ground Fault",
    "normal": "No Fault (Normal Condition)"
}

# ------------------ Streamlit UI ------------------
st.title("⚡ AI-Based Fault Detection in Transmission Lines")
st.write("Upload a `.mat` file (signal data) to predict fault type.")

uploaded_file = st.file_uploader("Upload .mat file", type=["mat"])

if uploaded_file is not None:
    # Load signal
    signal = load_mat_signal(uploaded_file)
    signal = pad_or_crop(signal)

    # Plot uploaded signal
    st.subheader("📈 Uploaded Signal")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signal[:500])  # show first 500 samples
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Extract features
    features = extract_features(signal)

    # Make predictions
    rf_pred = rf_model.predict(features)[0]
    svm_pred = svm_model.predict(features)[0]

    # Show predictions with full meaning
    st.success(f"### 🟢 Random Forest Prediction: {rf_pred} → {fault_mapping[rf_pred]}")
    st.success(f"### 🔵 SVM Prediction: {svm_pred} → {fault_mapping[svm_pred]}")

    # ------------------ Demo Evaluation ------------------
    st.subheader("📊 Model Performance (Sample Demo)")

    # Example true/predicted values (replace with real validation if needed)
    y_true = ["LLG","LLG","LLL","SLG","normal","SLG","LLL","LLG"]
    y_pred = ["LLG","LLG","LLL","SLG","normal","SLG","LLL","LLG"]

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=["LLG","LLL","SLG","normal"])
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["LLG","LLL","SLG","normal"],
                yticklabels=["LLG","LLL","SLG","normal"], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Confusion Matrix (Sample)")
    st.pyplot(fig_cm)

    # Classification Report
    report = classification_report(y_true, y_pred,
                                   target_names=["LLG","LLL","SLG","normal"],
                                   output_dict=True)
    st.write(pd.DataFrame(report).transpose())
