import sys
import joblib
import scipy.io
import numpy as np
import pandas as pd
import pywt
from scipy.fft import rfft

MODEL_PATH = "models/random_forest.pkl"
FIXED_LEN = 1024

def load_mat_signal(path):
    mat = scipy.io.loadmat(path)
    return mat['sig'].squeeze()

def pad_or_crop(signal, target_len=FIXED_LEN):
    if len(signal) >= target_len:
        return signal[:target_len]
    else:
        return np.pad(signal, (0, target_len-len(signal)), 'constant')

def extract_features(signal):
    # FFT features
    yf = np.abs(rfft(signal))
    feats = [np.mean(yf), np.std(yf), np.max(yf)]
    # Wavelet features
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    for c in coeffs:
        feats.extend([np.mean(c), np.std(c)])
    return pd.DataFrame([feats])

def predict(path):
    model = joblib.load(MODEL_PATH)
    sig = load_mat_signal(path)
    sig = pad_or_crop(sig)
    X = extract_features(sig)
    return model.predict(X)[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/infer.py <file.mat>")
    else:
        print("Prediction:", predict(sys.argv[1]))
